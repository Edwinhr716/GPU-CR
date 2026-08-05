# Background Primer: the OS and GPU concepts GPU-CR is built on

> **Audience.** You know C++ but have not worked with Linux systems programming
> (dynamic linking, signals, shared memory) or with CUDA/PyTorch. Read this before
> the per-subsystem references (`02`–`06`); they assume these concepts and only add
> short reminders. Every concept below is used somewhere concrete in this repo, and
> each section says where.

---

## 1. Dynamic linking and `LD_PRELOAD` — how GPU-CR gets inside another program

When a C/C++ program is compiled, calls to library functions like `cudaMalloc` are
usually *not* copied into the executable. Instead the executable records "I need the
symbol `cudaMalloc` from `libcudart.so`", and at program start a system component
called the **dynamic linker** (`ld.so`) finds each needed shared library (`.so` file,
the Linux equivalent of a Windows DLL) and wires the calls up.

The dynamic linker honors an environment variable, **`LD_PRELOAD`**: any library
listed there is loaded *first*, and when two libraries define the same symbol, the
one loaded first wins. So if our library `vGPU-NVIDIA.so` defines its own
`cudaMalloc`, and you run

```bash
LD_PRELOAD=/path/to/vGPU-NVIDIA.so python3 serving_vllm_nvidia.py
```

then every call the Python process (and PyTorch, and vLLM inside it) makes to
`cudaMalloc` lands in **our** function instead of NVIDIA's. This technique is called
**symbol interposition** and it is the foundation of the whole project: it is how
GPU-CR observes and records every GPU memory allocation *without modifying the
application at all*.

Two supporting mechanisms you will see everywhere in the code:

- **`dlsym(RTLD_NEXT, "cudaMalloc")`** — our fake `cudaMalloc` still needs to call
  the *real* one. `dlsym` is a function that looks up a symbol at runtime;
  `RTLD_NEXT` means "find the next definition after mine in load order", i.e. the
  genuine CUDA one. The typical hook pattern is: look up the real function once,
  do our bookkeeping, forward the call.
- **`__attribute__((constructor))`** — a GCC/Clang attribute marking a function that
  the dynamic linker runs automatically when the library is loaded, before `main()`.
  `init()` at the bottom of `src/vGPU.cpp` uses this; it is the library's entry
  point. Nothing in the application calls it — the *loader* does.

Where you'll meet this: `src/vGPU.cpp` (allocation hooks + constructor),
`src/ipc_hooks.cpp` (hooks for the CUDA driver's low-level memory API),
`src/nccl_hooks.cpp` (hooks for NCCL calls).

## 2. Processes, PIDs, and `/proc`

Every running program is a **process** with a numeric **PID**. GPU-CR is split
across processes on purpose:

- the **target** process: the GPU application (e.g. a vLLM server) with our library
  preloaded inside it;
- the **coordinator** process: a small CLI (`cr_client` or `multi_cr_client`) that a
  human or an orchestration agent runs to say "checkpoint PID 12345 now".

Linux exposes information about every process under the virtual directory
**`/proc/<pid>/`** — e.g. `/proc/12345/maps` lists that process's memory regions.
It's a "filesystem" backed by the kernel, not the disk. Some tooling around GPU-CR
reads it when other channels fail.

## 3. Signals — how the coordinator pokes the target

A **signal** is the kernel's built-in way for one process to interrupt another:
process A calls `kill(pid, SIGUSR1)` and the kernel forces process B to stop what it
is doing and run a function B registered in advance, a **signal handler**. It is the
only cross-process mechanism that works without the target's cooperation-in-the-moment,
which is exactly what a checkpointer needs — the target is busy running a model and
isn't listening on any socket for us.

Things to know when reading the handlers in `src/vGPU.cpp`:

- Signals are identified by number. Classic ones like `SIGUSR1`/`SIGUSR2`
  ("user-defined" signals) carry no payload; **real-time signals** (`SIGRTMAX`,
  `SIGRTMAX-1`, …) are a second block of user-definable signal numbers. GPU-CR's
  assignments are in `src/common.h` (`CR_CKPT_SIGNAL = SIGUSR1`,
  `CR_INIT_SIGNAL = SIGRTMAX`, the multi-GPU IPC phases on `SIGRTMAX-1..3`).
  Because a signal carries no data, the *payload* (e.g. which memory regions to
  selectively checkpoint) travels separately through shared memory (§4).
- A handler runs **asynchronously** — the target might be interrupted in the middle
  of `malloc` or of holding a lock. Code inside handlers is therefore heavily
  restricted in theory ("async-signal-safety"); GPU-CR's handlers do far more than
  the rules allow (they call CUDA, `printf`, take mutexes), which works in practice
  because the target is quiesced, but it is a permanent maintenance hazard — see the
  gotchas in `02-core-preload-library.md`.
- Note from production (docs/future-improvements.md §6): Python runtimes internally
  use SIGUSR1/2, so deployments patch GPU-CR's signals to `SIGRTMAX-7/8`. If a
  checkpoint "does nothing", signal-number mismatch between CLI and library is a
  prime suspect.

## 4. Shared memory, `mmap`, and hugepages — how bulk data and commands move

Signals say *"go"*; the actual data goes through **shared memory**: a region of RAM
mapped into two processes at once, so a write by one is instantly visible to the
other.

- **`mmap`** is the system call that maps a file (or anonymous memory) into a
  process's address space. Mapping *the same file* in two processes yields shared
  memory. GPU-CR creates files under a mount like `/mnt/huge-ckpt/` and both the
  coordinator and the target map them.
- GPU-CR uses two kinds of shared regions (see `src/comm/share_mem.cpp` and
  `src/backend/mmap_backend.cpp`):
  1. a tiny **control channel** holding a `signal_controls` struct — the command
     word (`CKPT_MSG`, `RESTORE_MSG`, …) plus the selective-region list — which the
     CLI writes and the library's handler reads, and through which the library acks
     completion (`FINISH_MSG`);
  2. a huge **staging buffer** (`SHM_SIZE`, default 25 GiB, compile-time
     `-DSHM_SIZE_GB`) that receives the GPU memory dump during checkpoint and feeds
     the restore.
- **Hugepages**: normal memory pages are 4 KiB; the CPU's address-translation cache
  (TLB) has limited entries, so copying tens of GiB through 4 KiB pages is slow.
  Linux offers 2 MiB pages via a special filesystem, **hugetlbfs** (mounted at
  `/mnt/huge-ckpt` in our setup), which must be *reserved* ahead of time
  (`echo N > /proc/sys/vm/nr_hugepages`). GPU-CR stages the VRAM dump on hugepages
  for DMA speed. Caveat from production: the code does not verify the mount is
  really hugetlbfs and silently degrades on a plain directory
  (docs/future-improvements.md §5).

## 5. File descriptors, Unix domain sockets, and `SCM_RIGHTS`

A **file descriptor (fd)** is a small integer a process uses to refer to an open
kernel object (file, socket, pipe — or, importantly here, a *GPU memory handle*).
Fds are per-process: the number 7 in process A means nothing in process B.

A **Unix domain socket (UDS)** is a socket between two processes on the same
machine, addressed by a filesystem path instead of an IP address. Its superpower is
**`SCM_RIGHTS`**: a message type that transfers *a copy of an fd itself* to the
other process — the kernel installs a new fd in the receiver referring to the same
object.

Why GPU-CR needs this: CUDA can export a chunk of GPU memory as a **POSIX fd** so
another process can map it (that is how multi-GPU workers share buffers). After a
restore, those shared mappings must be rebuilt, which means worker A must hand a
fresh fd to worker B — `src/ipc_fd_exchange.cpp` implements exactly this UDS +
`SCM_RIGHTS` exchange. See `03-ipc-interception.md`.

## 6. GPU programming in one page (CUDA, and where ROCm differs)

A GPU is a separate processor with its own memory (**VRAM**, "device memory") apart
from the CPU's RAM ("host memory"). A CUDA program:

1. allocates device memory (`cudaMalloc`),
2. copies input host→device (`cudaMemcpy` / `cudaMemcpyAsync`),
3. launches **kernels** — functions that run on the GPU in parallel,
4. copies results device→host.

Concepts the codebase leans on:

- **Runtime API vs driver API.** CUDA has two layers: the high-level *runtime* API
  (`cuda*` functions, e.g. `cudaMalloc`) and the low-level *driver* API (`cu*`
  functions, e.g. `cuMemCreate`). PyTorch uses both. GPU-CR hooks both:
  runtime-level allocation in `src/vGPU.cpp` / `src/GPUs/NVIDIA/nv.cpp`,
  driver-level VMM in `src/ipc_hooks.cpp`.
- **VMM (Virtual Memory Management), the key to "free VRAM but keep pointers".**
  The driver API splits allocation into steps: `cuMemAddressReserve` (claim a
  *virtual* address range — just numbers, no memory), `cuMemCreate` (allocate
  *physical* VRAM), `cuMemMap` + `cuMemSetAccess` (connect the two). Because the
  steps are separate, GPU-CR can **unmap and release the physical VRAM while
  keeping the virtual address reserved**. The application still holds pointers, and
  they become valid again when restore re-creates and re-maps physical memory at
  the same addresses. This is the single most important trick in the repository —
  it is why a checkpointed app's VRAM usage really drops to zero without the app
  noticing.
- **Streams and events.** A *stream* is an ordered queue of GPU work; *async*
  operations return immediately and you synchronize later. An *event* is a marker
  in a stream you can wait on. GPU-CR uses several streams + multi-threaded copies
  to move tens of GiB quickly (`memcpy_multi`, the backends in `src/GPUs/`).
- **Pinned (page-locked) host memory.** `cudaHostRegister` pins host RAM so the GPU
  can DMA into it directly (~2× faster). GPU-CR tries to pin its staging buffer;
  note that pinning currently *fails* on hugetlbfs mappings and the code continues
  unpinned (docs/future-improvements.md §5).
- **Contexts.** A driver-API notion of "this thread's connection to the GPU". You
  will see `pushContext`/`popContext` in the NVIDIA backend because helper threads
  must attach to the application's context before making driver calls.
- **ROCm/HIP** is AMD's CUDA equivalent (`hipMalloc`, `hipStream_t`, …). The
  abstract class in `src/GPUs/GPU.h` exists so the core logic never says "cuda" or
  "hip"; vendor code lives in `src/GPUs/NVIDIA/` and `src/GPUs/AMD/`.

## 7. The external checkpoint tools GPU-CR drives

GPU-CR moves the *bulk data* (VRAM contents) itself, but delegates the *opaque GPU
control state* (contexts, streams, driver state it can't see) to vendor tools:

- **`cuda-checkpoint`** (NVIDIA, vendored in `cuda-checkpoint/`): a driver utility
  that can lock a process's CUDA state, checkpoint it into host memory, restore it,
  and unlock. GPU-CR invokes the binary against the target PID during
  whole-process checkpoint (`src/GPUs/NVIDIA/nv.cpp`). Because GPU-CR has already
  dumped and *freed* almost all VRAM through the VMM trick, `cuda-checkpoint` only
  has to move the small remainder — that split (Data vs Control in the README
  charts) is the performance win.
- **CRIU** ("Checkpoint/Restore In Userspace", used for AMD): a Linux project that
  freezes an entire process and serializes it to disk; an AMD plugin adds GPU
  support. GPU-CR's AMD backend shells out to a custom-built `criu` (not included;
  see README §III) and needs `AMD_CKPT_DIR` plus sometimes the parent PID (`-m`).
- The **selective** path (checkpoint only named memory regions, added v0.2.0) uses
  *neither* tool — it never stops the process. See
  `docs/design-memory-address-checkpoint.md`.

## 8. Multi-GPU: NCCL, NVSHMEM, and why IPC must be torn down

- **NCCL** is NVIDIA's collective-communication library: when a model is split
  across N GPUs (vLLM tensor/pipeline parallelism), the worker processes exchange
  tensors through NCCL "**communicators**". Internally NCCL allocates GPU buffers
  and shares them *across processes* via the fd-export mechanism from §5/§6.
- **The problem:** memory that process B imported from process A cannot be
  checkpointed by A alone — the driver refuses while cross-process references
  exist, and blindly restoring would leave B pointing at dead memory. So before a
  multi-GPU checkpoint, all such IPC links must be **torn down** in a coordinated
  order across every worker, and **rebuilt** (with fresh fds re-exchanged) after
  restore. That is the phased dance run by `multi_cr_client`
  (`05-coordinator-clis-and-tools.md`) and implemented by `src/ipc_hooks.cpp`
  (`03-ipc-interception.md`) with help from the NCCL adapter
  (`06-adapters-and-build-system.md`).
- **NVSHMEM** is another NVIDIA multi-GPU memory-sharing library. It happens to
  allocate through the same driver VMM/fd path GPU-CR already hooks, so it needs no
  patches — only examples exist under `adapters/nvshmem/`.

## 9. PyTorch and vLLM — the workloads

- **PyTorch** is the dominant ML framework; its **caching allocator** grabs big
  VRAM slabs from CUDA and sub-allocates tensors internally, which hides individual
  tensors from GPU-CR. The selective-checkpoint deployment therefore runs with
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (one CUDA allocation per tensor — visible, but
  slower; see docs/future-improvements.md §3/§7 for the planned fix).
- **vLLM** is a popular LLM inference server built on PyTorch. It preallocates
  most free VRAM as a **KV cache** (per-token attention state), which is why
  multi-GPU dump sizes are ~36 GiB regardless of model size at
  `gpu_memory_utilization=0.9`. **Tensor parallelism (TP)** splits every layer
  across GPUs; **pipeline parallelism (PP)** puts different layers on different
  GPUs; either way you get one worker process per GPU, and `multi_cr_client` must
  coordinate all of them. The scripts in `apps/vllm/` are the reference workloads.
- **LoRA fine-tuning** (motivates the selective feature): tenants share one big
  frozen base model and each own small adapter + optimizer tensors. Selective C/R
  parks one tenant's tensors while the process keeps serving the others.

## 10. Checkpoint/Restore vocabulary used in this repo

| Term | Meaning here |
|---|---|
| **Checkpoint (ckpt)** | Copy GPU state to host staging and *free the physical VRAM* (usage → 0), keeping the process alive and its pointers reserved. |
| **Restore** | Re-allocate physical VRAM, re-map it at the original virtual addresses, copy staged bytes back, hand control state back via the vendor tool. |
| **Selective C/R** | Same, but only for caller-named `pointer:size` regions, without stopping the process or invoking cuda-checkpoint/CRIU. |
| **Data vs Control** | Data = bulk VRAM moved by GPU-CR itself; Control = opaque driver state moved by cuda-checkpoint/CRIU. |
| **Staging buffer** | The hugepage-backed host region the dump lands in (`SHM_SIZE`). |
| **Teardown/Rebuild** | The multi-GPU IPC phases that unlink and relink cross-process GPU memory around a checkpoint. |
| **Park / evict / swap** | Consumer-side synonyms for selectively checkpointing a tenant's state out of VRAM. |

## Where to go next

Return to [`00-start-here.md`](00-start-here.md) for the architecture map and
reading order.
