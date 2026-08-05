# 04 — GPU Backends (NVIDIA / AMD) and the NCCL Hooks

## What this subsystem does

The core checkpoint/restore engine in `src/vGPU.cpp` is deliberately vendor-neutral: it
copies GPU memory to a host staging buffer, releases the physical GPU memory behind each
allocation, and later re-materializes physical memory and copies the data back. To do
that it never calls CUDA or ROCm directly. Instead it calls through an abstract C++
interface, `class GPU`, declared in `src/GPUs/GPU.h`. Two concrete subclasses implement
that interface:

- `class nv` (`src/GPUs/NVIDIA/nv.h`, `src/GPUs/NVIDIA/nv.cpp`) — the NVIDIA/CUDA backend.
- `class amd` (`src/GPUs/AMD/amd.h`, `src/GPUs/AMD/amd.cpp`) — the AMD/ROCm (HIP) backend.

A tiny factory, `createGPU()` in `src/GPUs/gpu_factory.cpp`, decides at runtime which
subclass to instantiate.

The backend `.cpp` files also contain something that is *not* part of the class at all:
**LD_PRELOAD interception hooks** for the application's own allocation calls
(`cudaMalloc`/`cudaFree` on NVIDIA, `hipMalloc`/`hipFree` on AMD). These hooks are what
make checkpointing possible in the first place — they force every allocation through
the driver's *virtual memory management* (VMM) API so that the physical memory behind a
pointer can later be released and re-created without the pointer value changing.

Finally, `src/nccl_hooks.h` / `src/nccl_hooks.cpp` intercept the application's NCCL
communicator creation/destruction calls so that GPU-CR always knows which communicators
exist in a multi-GPU job.

> **Background: LD_PRELOAD.** On Linux, when a program starts, the dynamic linker
> (`ld.so`) resolves every function the program imports from shared libraries. The
> `LD_PRELOAD` environment variable names a shared library that the linker loads
> *before* all others, so any function it defines "shadows" the same-named function in
> the real libraries. GPU-CR is built as `vGPU-NVIDIA.so` / `vGPU-AMD.so`; running an
> application as `LD_PRELOAD=/path/to/vGPU-NVIDIA.so python train.py` means that when
> the application (or PyTorch inside it) calls `cudaMalloc`, it actually calls *our*
> `cudaMalloc` defined in `nv.cpp`. Our version can do extra bookkeeping and then
> either implement the call itself or forward to the real one.

> **Background: environment variables.** An environment variable is a named string
> (e.g. `GPU_VENDOR=NVIDIA`) attached to a process and inherited by every child process
> it spawns. C code reads them with `getenv("NAME")`. GPU-CR uses several
> (`GPU_VENDOR`, `AMD_CKPT_DIR`, `CR_NCCL_LIB`, `EXPORT_FILE_PATH`) as its runtime
> configuration mechanism, because an LD_PRELOAD library has no command line of its own.

### NVIDIA vs AMD strategy at a glance

| Aspect | NVIDIA (`nv`) | AMD (`amd`) |
|---|---|---|
| External process checkpoint tool | `cuda-checkpoint --toggle --pid <pid>` (freezes/unfreezes CUDA driver state for a process) | CRIU (`criu dump` / `criu restore`) — checkpoints the *entire Linux process*, with the AMD GPU plugin |
| Who actually runs that tool today | The coordinator CLI (`coordinator/cr_client.cpp:207-218`, `267-277`), **not** `nv::externalCheckpoint` (dead code path) | The coordinator CLI (`coordinator/cr_client.cpp:181-205`, `222-247`), **not** `amd::externalCheckpoint` (dead code path) |
| Allocation interception | `cudaMalloc`/`cudaFree` hooks re-implement allocation with the CUDA driver VMM API (`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`) | `hipMalloc`/`hipFree` hooks do the same with the HIP VMM API (`hipMemAddressReserve` + `hipMemCreate` + `hipMemMap`) |
| Release / remap of physical memory | `cuMemUnmap` + `cuMemRelease`, later `cuMemCreate` + `cuMemMap` at the *same* virtual address; sizes re-rounded to 2 MB internally | `hipMemUnmap` + `hipMemRelease`, later `hipMemCreate` + `hipMemMap`; sizes are stored *already rounded* by the hook |
| Special quirks | Must manage CUDA *contexts* explicitly (`pushContext`/`popContext`, captured PyTorch context `g_pytorch_context`); tracks allocation handles and erases them on release | No context management (HIP hides contexts); needs `AMD_CKPT_DIR` set and CRIU run as root; `releasePhysicalMemory` does **not** erase the stale handle from its handle map |
| Extra hook subsystems built in | `src/ipc_hooks.cpp` (CUDA IPC + P2P), `src/nccl_hooks.cpp` (NCCL tracking) | Neither — both are excluded from the AMD build in `CMakeLists.txt` |

> **Background: cuda-checkpoint.** A small NVIDIA-provided utility. Running
> `cuda-checkpoint --toggle --pid <P>` asks the NVIDIA driver to *suspend* all CUDA
> state of process `P`: running kernels are drained, device memory contents are moved
> to host memory by the driver, and the process's connection to the GPU is detached.
> Running the same command again *resumes* it. It operates on another process — the
> target application does not call it on itself.

> **Background: CRIU ("Checkpoint/Restore In Userspace").** A Linux tool that can
> freeze a whole process (memory, open files, sockets, threads) and write it to disk as
> image files, then later re-create the process exactly as it was. AMD ships a CRIU
> plugin that additionally saves/restores the ROCm GPU state, which is why the AMD path
> uses CRIU instead of a cuda-checkpoint-style tool. CRIU needs root privileges (the
> commands here are prefixed with `sudo`).

---

## `src/GPUs/GPU.h` — the abstract interface

This header defines vendor-neutral type aliases, two enums, and the pure-virtual `GPU`
class. Everything `vGPU.cpp` knows about a GPU goes through this class via the global
`GPU* gpu` pointer initialized in `init_CR()` (`src/vGPU.cpp:701`).

The type aliases (`GPU.h:9-11`) are all `void*`:

```cpp
typedef void* GPUStream;   // CUDA: cudaStream_t, ROCm: hipStream_t
typedef void* GPUEvent;    // CUDA: cudaEvent_t,  ROCm: hipEvent_t
typedef void* GPUDevice;   // CUDA: CUdevice,     ROCm: hipDevice_t
```

This works because CUDA and HIP stream/event handles are themselves opaque pointers;
each backend just casts them back. `GPUVendor` (`NVIDIA`/`AMD`/`UNKNOWN`) and
`GPUMemcpyKind` (`HostToDevice`/`DeviceToHost`/`DeviceToDevice`) are the only other
shared vocabulary.

> **Background: streams and events.** A CUDA/HIP *stream* is a work queue on the GPU:
> operations submitted to the same stream run in order; different streams may overlap.
> An *event* is a marker you can drop into a stream (`recordEvent`) and later wait on
> from the CPU (`synchronizeEvent`) — "block until the GPU has finished everything
> submitted before this marker." `vGPU.cpp` uses one stream + one event to implement
> double-buffered (pipelined) copies: while the GPU fills staging buffer B, the CPU
> drains staging buffer A, and the event tells the CPU when a buffer is safe to drain.

### Memory management methods

#### `virtual int allocate(void** ptr, size_t size)` / `virtual int deallocate(void* ptr)`
- **Purpose:** in principle, allocate/free tracked GPU memory through the backend
  object.
- **Called by:** *nobody*. Verified by grep — `vGPU.cpp` never calls `gpu->allocate` or
  `gpu->deallocate`. Both vendor implementations simply `throw std::runtime_error`
  (`nv.cpp:81-87`, `amd.cpp:57-63`), because real allocations are intercepted at the
  `cudaMalloc`/`hipMalloc` hook level, before any `GPU` object is even involved.
- **Why it matters:** the interface slot exists so a future backend *could* route
  allocations through the object, but today it is a deliberate trap: if any code path
  ever reaches these methods, something is wired wrong and you get an immediate
  exception rather than silent mis-tracking.

#### `virtual std::map<void*, size_t>& getMemoryMap()`
- **Purpose:** expose the map of tracked allocations (device pointer → byte size).
- **Called by:** *nobody today* (grep of `src/` finds no `getMemoryMap` caller outside
  the backends). `vGPU.cpp` instead reads the global `std::map<void*, size_t>
  allocated_memory` directly (declared `extern` in `src/common.h:61`, defined once per
  backend in `nv.cpp:13` / `amd.cpp:11`, and iterated at e.g. `src/vGPU.cpp:106`,
  `196`, `226`).
- **Why it matters:** it is the *intended* clean accessor for the allocation table, but
  beware — the AMD implementation returns the wrong map (see the AMD section and the
  gotchas). If you refactor `vGPU.cpp` to use `getMemoryMap()`, fix AMD first.

#### `virtual int memcpyAsync(void* dst, const void* src, size_t size, GPUMemcpyKind kind, GPUStream stream)`
- **Purpose:** enqueue an asynchronous memory copy on a stream, in the given direction.
  "Asynchronous" means the call returns immediately; the copy happens whenever the GPU
  gets to it, and completion is observed via events/stream sync.
- **Called by:** `vGPU.cpp` through the `GPU*` interface, in all four data-movement
  loops: full checkpoint `ckpt()` (`src/vGPU.cpp:135`), selective checkpoint
  `ckpt_selective()` (`:319`), full restore `restore_ptr_and_content()` (`:472`), and
  selective restore `restore_ptr_and_content_selective()` (`:590`).
- **Why it matters:** this is the actual data path of every checkpoint and restore —
  gigabytes of model weights flow through it. It is asynchronous specifically so the
  engine can overlap GPU→host DMA with the CPU-side `memcpy_multi()` into the shared
  memory "filesystem".

### Synchronization methods (grouped: thin wrappers)

`createStream`, `destroyStream`, `createEvent`, `destroyEvent`, `recordEvent`,
`synchronizeEvent`, `synchronizeStream`:

- **Purpose:** create/destroy the stream and event used for pipelined copies, mark
  progress points, and block the CPU until the GPU catches up. Each is a one-line
  wrapper over the corresponding `cudaXxx`/`hipXxx` call in the backends.
- **Called by:** `vGPU.cpp` via the `GPU*` interface at the start and end of each of
  the four copy loops — e.g. `createStream`/`createEvent` at `src/vGPU.cpp:90-95`
  (`ckpt`), `:260-265` (`ckpt_selective`), `:438-442` (`restore_ptr_and_content`),
  `:556-560` (selective restore); `recordEvent` at `:100`, `:161`, `:270`, `:446`,
  `:564`; `synchronizeEvent` at `:148`, `:167`, `:332`, `:489`, `:607`;
  `synchronizeStream` at `:180`, `:365`, `:506`, `:624`; `destroyStream`/`destroyEvent`
  at `:190-191`, `:375-376`, `:510-511`, `:628-629`.
- **Why it matters:** correctness of the double-buffer pipeline depends entirely on
  these. If you ever see corrupted checkpoint data, suspect a missing
  `synchronizeEvent` between the GPU filling a staging buffer and the CPU reading it.

#### `virtual int syncAllKernels()`
- **Purpose:** device-wide barrier (`cudaDeviceSynchronize` /
  `hipDeviceSynchronize`) — wait for *every* kernel and copy the application has in
  flight, on all streams.
- **Called by:** `vGPU.cpp` right before any checkpoint touches memory:
  `cr_signal_handler` before selective ckpt (`src/vGPU.cpp:763`) and full ckpt
  (`:783`), and `cr_ipc_signal_handler` before IPC teardown (`:849`).
- **Why it matters:** the checkpoint is triggered *asynchronously by a signal* while
  the application may be mid-training-step. Snapshotting memory while a kernel is
  still writing it would capture garbage. This is the "quiesce the GPU" step.

#### `virtual int registerHostMemory(void* ptr, size_t size)`
- **Purpose:** pin (page-lock) an existing host buffer so the GPU can DMA into it at
  full PCIe bandwidth.
- **Called by:** `init_CR()` (`src/vGPU.cpp:714`), on the hugepage-backed staging
  buffer obtained from the shared-memory backend.
- **Why it matters:** unpinned (pageable) host memory forces CUDA to bounce copies
  through an internal pinned buffer, roughly halving throughput. Note the caller
  treats failure as non-fatal: registration of hugepage memory often fails, and the
  code just logs "will use regular memory" and continues.

> **Background: pinned / registered host memory.** Normal process memory can be paged
> out by the OS at any moment, so the GPU's DMA engine cannot safely read it directly.
> `cudaHostRegister`/`hipHostRegister` tell the OS "lock these pages in RAM" and tell
> the driver about them, enabling direct DMA. The trade-off is that pinned memory is a
> scarce system resource.

### Checkpoint/restore memory management

#### `virtual int releasePhysicalMemory(void* ptr)`
- **Purpose:** free the *physical* GPU memory backing an allocation while keeping its
  *virtual address* reserved, so the pointer value stays valid-looking and can be
  re-populated later.
- **Called by:** `ckpt()` (`src/vGPU.cpp:198`) and `ckpt_selective()` (`:382`), via the
  `GPU*` interface, after the data has been copied out.
- **Why it matters:** this is the core trick of GPU-CR. The application keeps holding
  raw device pointers (in Python objects, tensors, graphs...) across the checkpoint.
  If we freed memory the normal way, restore would hand back *different* addresses and
  every stored pointer in the application would dangle. Releasing only the physical
  backing frees the GPU for other tenants while preserving the address space contract.

#### `virtual int remapPhysicalMemory(void* ptr, size_t size)`
- **Purpose:** the inverse — allocate fresh physical GPU memory and map it at exactly
  the previously reserved virtual address `ptr`.
- **Called by:** `restore_ptr_and_content()` (`src/vGPU.cpp:427`) and
  `restore_ptr_and_content_selective()` (`:545`), via the `GPU*` interface, before data
  is copied back in.
- **Why it matters:** together with `releasePhysicalMemory` this implements
  "same-address restore." It only works because the allocation hooks used the VMM API
  in the first place — the classic `cudaMalloc` gives you no way to choose an address.

> **Background: `cudaMalloc` vs the cuMem VMM API.** Classic `cudaMalloc(&p, n)` is a
> single opaque operation: the driver picks a virtual address and physical memory
> together, and `cudaFree` destroys both. The CUDA *Virtual Memory Management* driver
> API (and its HIP mirror) splits this into four steps you control individually:
> 1. `cuMemAddressReserve` — reserve a range of GPU *virtual* addresses (no memory yet);
> 2. `cuMemCreate` — allocate a chunk of *physical* GPU memory, returned as a handle;
> 3. `cuMemMap` — bind the physical chunk to the reserved virtual range;
> 4. `cuMemSetAccess` — make the mapping readable/writable.
> Because the steps are separate, you can later `cuMemUnmap` + `cuMemRelease` (drop the
> physical memory) *without* `cuMemAddressFree` (keep the address), and re-run steps
> 2-4 at the same address. VMM granularity is 2 MB, hence the pervasive
> `ROUND_UP_2MB()` macro from `src/common.h:23`.

### External tool interfaces

#### `virtual int externalCheckpoint(int pid)` / `virtual int externalRestore(int pid)`
- **Purpose:** shell out to the vendor's external checkpoint tool (cuda-checkpoint or
  CRIU) against process `pid`.
- **Called by:** **nobody in the current tree.** Grep across `src/` and `coordinator/`
  finds only the definitions. The external tools are actually invoked by the
  coordinator CLI `cr_client` (`coordinator/cr_client.cpp:207-218` and `:267-277` for
  cuda-checkpoint; `:181-205` and `:222-247` for CRIU), i.e. from a *separate process*,
  not from inside the checkpointed application. The comments at `src/vGPU.cpp:800-803`
  confirm this division ("External checkpoint ... is called from cr_client, not
  here").
- **Why it matters:** it has to be an external process. cuda-checkpoint freezes the
  target's entire CUDA state and CRIU freezes the target's entire process — a process
  cannot meaningfully do either to itself. Keep these methods in mind as legacy /
  reserved API; if you change the shell commands, the live copies to update are in
  `cr_client.cpp`, and the near-duplicates here will silently drift (they already
  have: `cr_client`'s CRIU command adds `--ghost-limit 50M --ext-unix-sk`, the ones in
  `amd.cpp` do not).

#### `virtual int pushContext()` / `virtual int popContext()`
- **Purpose:** make the right CUDA *context* current on the signal-handler thread
  before doing GPU work, and restore the previous one after. These have default no-op
  implementations in `GPU.h:76-77`; only NVIDIA overrides them (AMD/HIP has no
  user-visible context stack).
- **Called by:** `cr_signal_handler` (`src/vGPU.cpp:758` and `:820`), bracketing the
  entire ckpt/restore dispatch, via the `GPU*` interface.
- **Why it matters:** see the Background note below and `nv::pushContext` — getting
  this wrong makes every CUDA call in the checkpoint path fail with "invalid context"
  or, worse, operate on a different context than the one holding the allocations.

> **Background: CUDA contexts.** A CUDA context is like a per-process GPU "session":
> allocations, streams, and loaded kernels belong to a context, and each CPU thread has
> a *current context* (managed as a small stack: `cuCtxPushCurrent` /
> `cuCtxPopCurrent`). PyTorch creates and uses its own context. GPU-CR's checkpoint
> runs inside a signal handler on whatever thread got the signal — that thread's
> current context may be nothing at all, so GPU-CR must explicitly push the context
> that owns the application's memory before touching it.

#### `virtual GPUVendor getVendor() const` / `virtual std::string getVendorName() const`
- **Purpose:** identify the backend.
- **Called by:** `getVendorName` is used for logging in `init_CR()`
  (`src/vGPU.cpp:702`). `getVendor` currently has no callers (grep-verified) — it
  exists for future vendor-conditional logic.
- **Why it matters:** trivial, but it is the only runtime introspection the neutral
  code has.

#### `GPU* createGPU()` (free function declaration, `GPU.h:89`)
Documented with its implementation below. The header comment notes the caller owns the
returned object; in practice `vGPU.cpp` stores it in the global `gpu` and never deletes
it (the library lives as long as the process).

---

## `src/GPUs/gpu_factory.cpp` — the factory

### `GPU* createGPU()`
- **Purpose:**
  1. Read the `GPU_VENDOR` environment variable; abort with a clear message if unset
     (`gpu_factory.cpp:14-20`).
  2. Under `#ifdef __HIP_PLATFORM_AMD__` (a *compile-time* switch), accept
     `AMD`/`ROCM`/`HIP` (case-insensitive) and return `new amd()`; otherwise accept
     `NVIDIA`/`CUDA` and return `new nv()`.
  3. If the runtime string does not match the compiled-in backend, abort — e.g. an
     AMD-built `.so` refuses `GPU_VENDOR=NVIDIA` (`gpu_factory.cpp:27`, `:35`).
- **Called by:** `init_CR()` at `src/vGPU.cpp:701` — exactly once per process, lazily,
  the first time a checkpoint/restore signal arrives.
- **Why it matters:** note that only *one* backend exists in any given binary. The
  factory is not choosing between two compiled backends; it is a sanity check that the
  environment agrees with the build. Both knobs must match (see "How vendor selection
  works").

---

## `src/GPUs/NVIDIA/nv.h` and `nv.cpp` — the NVIDIA backend

`nv.h` declares the `nv` subclass (device, context, lazy-init flag) plus `extern "C"`
prototypes for the hooked `cudaMalloc`/`cudaFree`. `nv.cpp` holds three kinds of code:
the class implementation, file-level global state, and the LD_PRELOAD hooks.

Global state (`nv.cpp:12-18`):
- `std::map<void*, size_t> allocated_memory` — pointer → **unrounded** requested size.
  This is the single source of truth for "what to checkpoint"; `vGPU.cpp` iterates it
  directly via the `extern` in `common.h`.
- `std::map<void*, int> allocated_memory_type` — 0 = classic cudaMalloc, 1 = VMM (in
  practice everything the hook produces is 1).
- `static std::map<void*, CUmemGenericAllocationHandle> global_handle_map` — pointer →
  physical-memory handle from `cuMemCreate`, needed to `cuMemRelease` later.
- `CUcontext g_pytorch_context` — the application's (PyTorch's) CUDA context, captured
  by the `cudaMalloc` hook. Shared with `src/ipc_hooks.cpp` (extern at
  `ipc_hooks.cpp:43`), where kernel-launch hooks push it whenever they detect a context
  mismatch (`ipc_hooks.cpp:933-1197`).

Error-handling macros: `CU_CHECK` and `CUDA_CHECK_RET` (`nv.cpp:28-44`) both print and
**`exit(EXIT_FAILURE)`** on error — they do not return an error code despite the name.
Keep that in mind: any method built on them kills the whole application on failure.

### `nv::nv()` / `nv::~nv()` / `nv::ensureCudaInitialized()`
- **Purpose:** the constructor only logs; real initialization is deferred to
  `ensureCudaInitialized()` (`nv.cpp:54-77`), which calls `cuInit(0)`, then either
  adopts the thread's current context or retains device 0's *primary context*
  (`cuDevicePrimaryCtxRetain`) if none exists.
- **Called by:** the constructor runs from `createGPU()`
  (`src/GPUs/gpu_factory.cpp:33`) and from the `cudaMalloc` hook's private
  `hook_gpu = new nv()` (`nv.cpp:351`). `ensureCudaInitialized` is called by
  `registerHostMemory` (`nv.cpp:148`) and `pushContext` (`nv.cpp:293`).
- **Why it matters:** laziness is deliberate — the object may be constructed before
  the application has initialized CUDA, and initializing CUDA too early inside an
  LD_PRELOAD library can subtly change application behavior.

### `nv::allocate` / `nv::deallocate`
Throw `std::runtime_error` unconditionally (`nv.cpp:81-87`). All allocation goes
through the `cudaMalloc`/`cudaFree` hooks below. No callers (grep-verified).

### `nv::getMemoryMap()`
Returns the global `allocated_memory` (`nv.cpp:89-91`) — correct, but currently
uncalled (see interface section).

### Stream/event/copy/sync wrappers (grouped)
`createStream`, `destroyStream`, `createEvent`, `destroyEvent`, `recordEvent`,
`synchronizeEvent`, `memcpyAsync`, `synchronizeStream`, `syncAllKernels`
(`nv.cpp:95-145`) are each a single CUDA *runtime* API call wrapped in
`CUDA_CHECK_RET` (so any failure exits the process). `memcpyAsync` translates
`GPUMemcpyKind` to `cudaMemcpyKind` and returns -1 for an unknown kind. Called by
`vGPU.cpp` via the `GPU*` interface as listed in the interface section above.

### `nv::registerHostMemory(void* ptr, size_t size)` (`nv.cpp:147-158`)
- **Purpose:** `cudaHostRegister(ptr, size, cudaHostRegisterMapped |
  cudaHostRegisterPortable)` on the staging buffer.
- **Called by:** `init_CR()` (`src/vGPU.cpp:714`) via `GPU*`.
- **Why it matters:** this is the one CUDA wrapper that deliberately does *not* exit on
  failure. Hugepage-backed memory frequently cannot be registered; it logs, calls
  `cudaGetLastError()` to clear the sticky error (important — CUDA errors otherwise
  poison later calls), and returns -1 so the caller can proceed unpinned.

### `nv::releasePhysicalMemory(void* ptr)` (`nv.cpp:162-201`)
- **Purpose:** step-by-step:
  1. Look `ptr` up in `allocated_memory`; warn and return -1 if unknown.
  2. Re-round the stored (unrounded) size with `ROUND_UP_2MB`.
  3. `cuMemUnmap(ptr, aligned_size)` — detach physical memory from the virtual range.
  4. Look up the physical handle in `global_handle_map`; `cuMemRelease` it and erase
     the map entry.
  5. Deliberately **skip** `cuMemAddressFree` so the virtual range stays reserved.
- **Called by:** `ckpt()` (`src/vGPU.cpp:198`) and `ckpt_selective()` (`:382`) via
  `GPU*`.
- **Why it matters:** this is what actually gives the GPU memory back to the system at
  checkpoint time while keeping application pointers stable. Unlike the wrapper
  methods it returns -1 instead of exiting, but note its *caller* in `ckpt()` treats
  -1 as fatal and exits anyway.

### `nv::remapPhysicalMemory(void* ptr, size_t size)` (`nv.cpp:204-265`)
- **Purpose:** step-by-step:
  1. Verify `ptr` is tracked in `allocated_memory` (warn + -1 otherwise).
  2. If a handle already exists for `ptr` (the address is currently mapped — e.g. a
     restore onto a live allocation), unmap and release the old physical memory first
     (`nv.cpp:213-234`).
  3. `cuMemCreate` a fresh 2 MB-aligned physical chunk on `device_` (pinned,
     device-located).
  4. `cuMemMap` it at the *existing* virtual address `ptr`, `cuMemSetAccess` RW.
  5. Record the new handle in `global_handle_map`.
- **Called by:** `restore_ptr_and_content()` (`src/vGPU.cpp:427`) and
  `restore_ptr_and_content_selective()` (`:545`) via `GPU*`.
- **Why it matters:** steps 3-5 use `CU_CHECK`, so a failure here (typically GPU OOM at
  restore time) kills the process. Also note `device_` is only set by
  `ensureCudaInitialized()`; the restore path reaches this via
  `pushContext()` → `ensureCudaInitialized()` in the signal handler, so ordering
  matters if you refactor.

### `nv::externalCheckpoint(int pid)` / `nv::externalRestore(int pid)` (`nv.cpp:269-290`)
- **Purpose:** build the string `cuda-checkpoint --toggle --pid <pid>` and run it with
  `system()`. Restore just calls checkpoint again, because `--toggle` flips state in
  both directions.
- **Called by:** nobody (grep-verified; the live invocation is
  `coordinator/cr_client.cpp:207-218` / `:267-277`, which even resolves a bundled
  binary path relative to its own executable via `/proc/self/exe`).
- **Why it matters:** dead-but-plausible code — see the interface section and gotchas.

> **Background: `system()` and `/proc`.** `system("cmd")` asks the OS to run a shell
> command as a *child process* and waits for it to finish — it is the simplest way for
> C++ code to invoke an external tool. `/proc` is a virtual filesystem the Linux
> kernel exposes: `/proc/<pid>/...` describes each running process, and the symlink
> `/proc/self/exe` points at the current process's own executable file (used by
> `cr_client` to find the bundled `cuda-checkpoint` next to itself).

### `nv::pushContext()` / `nv::popContext()` (`nv.cpp:292-322`)
- **Purpose:** `pushContext` ensures CUDA is initialized, then pushes
  `g_pytorch_context` if the `cudaMalloc` hook ever captured one, otherwise its own
  `context_`, onto the calling thread's context stack (`cuCtxPushCurrent`).
  `popContext` pops whatever is current.
- **Called by:** `cr_signal_handler` (`src/vGPU.cpp:758`, `:820`) via `GPU*`,
  bracketing the whole checkpoint/restore dispatch.
- **Why it matters:** the allocations being checkpointed live in the *application's*
  context. Preferring the captured PyTorch context over GPU-CR's own is what makes
  checkpointing PyTorch/vLLM work at all. These return -1 on failure rather than
  exiting.

### LD_PRELOAD hooks: `cudaMalloc` / `cudaFree` (`extern "C"`, `nv.cpp:326-512`)
- **Purpose (`cudaMalloc`, step-by-step):**
  1. Take `gpu_mem_mutex` (shared with `vGPU.cpp`, defined at `src/vGPU.cpp:38`) so an
     in-progress checkpoint never races an allocation.
  2. Capture the caller's current context into `g_pytorch_context` if not yet captured
     (`nv.cpp:333-339`) — this is where the "PyTorch context" comes from.
  3. Handle `size == 0` by returning `nullptr`/success.
  4. Lazily `cuInit`, and get-or-retain a context/device (function-local statics,
     `nv.cpp:359-400`).
  5. Perform the four-step VMM allocation (`cuMemAddressReserve` → `cuMemCreate` →
     `cuMemMap` → `cuMemSetAccess`) with the size rounded up to 2 MB, unwinding
     partial state on each failure and returning `cudaErrorMemoryAllocation`.
  6. Record the pointer in `global_handle_map`, `allocated_memory` (with the
     **original unrounded** size) and `allocated_memory_type` (=1).
- **Purpose (`cudaFree`):** under the same mutex, if the pointer is tracked, fully tear
  down the VMM allocation (`cuMemUnmap` + `cuMemRelease` + `cuMemAddressFree` — note
  this one *does* free the address, unlike `releasePhysicalMemory`) and erase all
  tracking; if untracked, forward to the real `cudaFree` found via
  `dlsym(RTLD_NEXT, "cudaFree")` (`nv.cpp:476-482`).
- **Called by:** the *application* (or CUDA runtime users inside it such as PyTorch),
  transparently, because `vGPU-NVIDIA.so` is LD_PRELOADed and its `cudaMalloc`
  symbol shadows `libcudart`'s. No GPU-CR code calls these directly.
- **Why it matters:** this is the foundation of the whole system — an allocation that
  did *not* go through this hook is invisible to checkpointing and cannot be
  released/remapped. It is also why `NCCL_CUMEM_ENABLE` and PyTorch allocator settings
  matter operationally: allocations made through other driver paths bypass the hook.

> **Background: `dlopen` / `dlsym` / `RTLD_NEXT`.** `dlopen("lib.so", flags)` loads a
> shared library at runtime and returns a handle; `dlsym(handle, "name")` looks up a
> function address in it. The pseudo-handle `RTLD_NEXT` means "search the libraries
> that come *after* the one making the call" — inside an LD_PRELOAD hook, that finds
> the *real* implementation the hook is shadowing, which is how a hook forwards calls
> it does not want to handle itself.

---

## `src/GPUs/AMD/amd.h` and `amd.cpp` — the AMD backend

Structured as a mirror of the NVIDIA backend with HIP (`hip*`) APIs. Global state
(`amd.cpp:11-13`): the same trio — `allocated_memory`, `global_handle_map` (of
`hipMemGenericAllocationHandle_t`), `allocated_memory_type` — **plus** a private member
`amd::memory_map_` that nothing ever writes to. The `HIP_CHECK` macro
(`amd.cpp:28-34`) exits on failure like `CU_CHECK`, but is in fact almost unused; most
AMD methods check `hipError_t` manually and return -1, so the AMD backend is generally
*less* fatal than NVIDIA's.

### `amd::amd()` / `amd::~amd()` / `amd::ensureHipInitialized()`
- **Purpose:** the constructor calls `hipSetDevice(0)` (warning on failure);
  `ensureHipInitialized` calls `hipInit(0)` once, tolerating failure.
- **Called by:** constructor from `createGPU()` (`gpu_factory.cpp:25`).
  `ensureHipInitialized` currently has **no callers** in `amd.cpp` (grep-verified) —
  unlike NVIDIA, nothing invokes it; HIP initializes implicitly on first API call.
- **Why it matters:** there is no HIP context management at all — HIP hides contexts —
  which is why `amd` does not override `pushContext`/`popContext` and inherits the
  no-op defaults from `GPU.h:76-77`.

### `amd::allocate` / `amd::deallocate`
Throw, exactly like the NVIDIA versions (`amd.cpp:57-63`). Never called.

### `amd::getMemoryMap()` — returns the wrong map
Returns the private, always-empty `memory_map_` (`amd.cpp:65-67`), *not* the global
`allocated_memory` that the `hipMalloc` hook populates. Harmless only because nothing
calls `getMemoryMap()` today; a landmine for the first refactor that does. (Compare
`nv::getMemoryMap`, which returns the global.)

### `amd::releasePhysicalMemory(void* ptr)` (`amd.cpp:69-103`)
- **Purpose:** look up `ptr` in `allocated_memory` (note: **returns 0 and skips** if
  untracked, where NVIDIA returns -1), find its handle, `hipMemUnmap` +
  `hipMemRelease`, keep the virtual address. The stored size is used as-is — it is
  already 2 MB-rounded because the hook stores the aligned size (see below).
- **Called by:** `ckpt()` (`src/vGPU.cpp:198`) and `ckpt_selective()` (`:382`) via
  `GPU*` — same call sites as NVIDIA; the neutral code cannot tell the difference.
- **Why it matters:** two asymmetries vs NVIDIA: (1) it does **not**
  `global_handle_map.erase(...)` after releasing, so a stale handle stays in the map —
  calling release twice on the same pointer would `hipMemRelease` an already-released
  handle; (2) untracked pointers are silently tolerated. Both are worth fixing if you
  touch this file.

### `amd::remapPhysicalMemory(void* ptr, size_t size)` (`amd.cpp:105-148`)
- **Purpose:** `hipMemCreate` (pinned, device 0) → `hipMemMap` at the existing address
  → `hipMemSetAccess` RW, unwinding on failure; then (re)record `ptr` in all three
  tracking maps.
- **Called by:** `restore_ptr_and_content()` (`src/vGPU.cpp:427`) and the selective
  variant (`:545`) via `GPU*`.
- **Why it matters:** unlike the NVIDIA version it does **not** first check whether
  `ptr` is already mapped and release the old backing — remapping a still-mapped
  address will fail (or leak) rather than being handled gracefully. It also trusts the
  caller's `size` without rounding; the callers in `vGPU.cpp` pass already-rounded
  sizes, so this works, but it is an implicit contract.

### Stream/event/copy/sync wrappers (grouped)
`createStream` ... `syncAllKernels` and `memcpyAsync` (`amd.cpp:150-250`) mirror the
NVIDIA set with `hip*` calls, but every one returns -1 on failure instead of exiting.
`memcpyAsync` maps unknown kinds to `hipMemcpyDefault` instead of erroring. Called by
`vGPU.cpp` via `GPU*` at the same lines listed in the interface section.

### `amd::registerHostMemory` (`amd.cpp:252-259`)
`hipHostRegister(ptr, size, hipHostRegisterDefault)`; -1 on failure. Called by
`init_CR()` (`src/vGPU.cpp:714`).

### `amd::externalCheckpoint(int pid)` / `amd::externalRestore(int pid)` (`amd.cpp:261-312`)
- **Purpose:** read `AMD_CKPT_DIR` from the environment (hard `exit` if unset), then
  `system()` a `sudo criu dump -t <pid> -D <dir> ...` (checkpoint) or
  `sudo criu restore -D <dir> ...` command, with `LD_LIBRARY_PATH` pointed at the
  AMD GPU userspace libraries and `-L /usr/local/lib/criu` selecting the CRIU plugin
  directory (where the AMD GPU plugin lives). Logs land in `<dir>/dump.log` /
  `<dir>/restore.log`.
- **Called by:** nobody (grep-verified). The live CRIU invocations are in
  `coordinator/cr_client.cpp:181-205` (dump) and `:222-247` (restore), which carry
  additional flags (`--ghost-limit 50M --ext-unix-sk`, `--restore-detached`,
  `--pidfile`) that these methods lack.
- **Why it matters:** same "dead but plausible" warning as NVIDIA — and doubly so
  here, because the flag sets have already diverged from the real ones.

### LD_PRELOAD hooks: `hipMalloc` / `hipFree` (`extern "C"`, `amd.cpp:315-424`)
- **Purpose:** same four-step VMM dance as the CUDA hook
  (`hipMemAddressReserve` → `hipMemCreate` → `hipMemMap` → `hipMemSetAccess`), with
  cleanup on each failure. `hipFree` tears down tracked VMM allocations completely
  (including `hipMemAddressFree`) and forwards untracked pointers to the real
  `hipFree` via `dlsym(RTLD_NEXT, ...)`.
- **Called by:** the application, transparently via LD_PRELOAD of `vGPU-AMD.so`.
- **Why it matters — differences from the CUDA hook:**
  - **No mutex.** The CUDA hook locks `gpu_mem_mutex`; the HIP hook does not, so a
    concurrent allocation during a checkpoint is a real race on AMD.
  - **Stores the aligned size**: `allocated_memory[*devPtr] = aligned_size`
    (`amd.cpp:377`), whereas NVIDIA stores the unrounded request. Downstream code
    compensates (NVIDIA re-rounds in release/remap; the selective-checkpoint code in
    `vGPU.cpp:285-291` explicitly documents handling of rounding), but any new code
    reading `allocated_memory` must remember the semantics differ per vendor.
  - **No context capture** (no HIP equivalent needed).

---

## `src/nccl_hooks.h` and `src/nccl_hooks.cpp` — NCCL communicator tracking

> **Background: NCCL and communicators.** NCCL (NVIDIA Collective Communications
> Library, pronounced "nickel") is the library PyTorch and vLLM use to move tensors
> *between* GPUs — all-reduce for gradient averaging, all-gather for tensor
> parallelism, and so on. A **communicator** (`ncclComm_t`) is NCCL's handle for one
> group of participating GPUs/processes: creating one (`ncclCommInitRank`) opens
> network/IPC connections and allocates GPU buffers shared across ranks. A
> "collective" operation is one that *every* rank in the communicator must call at the
> same time, or everyone hangs.

**Historical note (important for reading this code):** the header's big comment block
(`nccl_hooks.h:1-23`) describes an older design where checkpoint suspended
communicators with a custom `ncclCommSuspend`/`ncclCommResume` (requiring a patched
NCCL and `NCCL_CUMEM_ENABLE=1`). That design is **retired**: the `.cpp` header comment
(`nccl_hooks.cpp:1-12`) and `src/common.h:41-48` ("These replace the old NCCL
suspend/resume signals") record that multi-GPU IPC state is now torn down/rebuilt by
`src/ipc_hooks.cpp` instead. What survives here is *communicator tracking only*, kept
for diagnostics and future use. On AMD builds this file is excluded entirely
(`CMakeLists.txt:76`).

The tracked state is a mutex-protected `std::vector<ncclComm_t> g_tracked_comms`
(`nccl_hooks.cpp:56-57`). NCCL types are re-declared locally (an opaque `ncclComm_t`
pointer and a 128-byte `ncclUniqueId` struct) so the build needs no `nccl.h` — the
declarations only have to match NCCL's C ABI.

### `find_real_nccl_sym(const char* name)` (static, `nccl_hooks.cpp:77-191`)
- **Purpose:** find the address of the *real* NCCL function that a hook is shadowing.
  Five strategies, in order:
  1. `dlsym(RTLD_NEXT, name)` — works if `libnccl.so` was linked in normally. The
     comment stresses this must come first: `RTLD_DEFAULT` would find our *own* hook
     and recurse forever.
  2. `dlopen("libnccl.so.2", RTLD_NOLOAD)` — `RTLD_NOLOAD` means "give me a handle
     only if this library is *already* loaded." This catches PyTorch, which loads NCCL
     itself via `dlopen` *after* our preload library, with `RTLD_LOCAL` (symbols not
     visible to `RTLD_NEXT`). Retried on every call because the load may happen late.
  3. `dlopen(getenv("CR_NCCL_LIB"))` — an explicit path to a locally built NCCL,
     historically the one with `ncclCommSuspend` (tried once).
  4. Path relative to our own `.so`: `dladdr()` on this very function reveals where
     `vGPU-NVIDIA.so` lives on disk, and `<that dir>/../nccl_install/lib/libnccl.so.2`
     is tried (once).
  5. Plain `dlopen("libnccl.so.2")` from the normal library search path (last resort).
- **Called by:** each of the four NCCL hooks below, lazily on their first invocation.
- **Why it matters:** this function encodes the hardest-won knowledge in the file —
  the dynamic-linking order problems of hooking a library that the application loads
  lazily and privately. If NCCL interception "mysteriously" stops working after a
  PyTorch upgrade, start here.

### `nccl_register_comm` / `nccl_unregister_comm` / `nccl_get_comm_count` (`nccl_hooks.cpp:199-225`)
- **Purpose:** mutex-protected, idempotent add/remove/count on `g_tracked_comms`.
- **Called by:** `nccl_register_comm` is called by the `ncclCommInitRank` and
  `ncclCommInitRankConfig` hooks (`nccl_hooks.cpp:271`, `:295`) and by the exported
  `cr_register_nccl_comm` (`:336`); `nccl_unregister_comm` by the `ncclCommDestroy`
  hook (`:312`) and `cr_unregister_nccl_comm` (`:340`). `nccl_get_comm_count` has no
  in-tree callers (grep-verified) — it is a diagnostics accessor.
- **Why it matters:** this list is the system's only inventory of live communicators.

### `nccl_suspend_all_comms` / `nccl_resume_all_comms` / `nccl_suspend_available` (`nccl_hooks.cpp:227-243`)
- **Purpose:** deprecated stubs. `nccl_suspend_available()` hard-returns `false`; the
  suspend/resume functions print "deprecated — use ipc_teardown_all_imports() /
  ipc_rebuild_all_imports()" and return 1 (the documented "nothing tracked / no-op"
  code).
- **Called by:** nobody — `src/vGPU.cpp` includes `nccl_hooks.h` (`vGPU.cpp:22`) but
  calls none of these (grep-verified); the multi-GPU signal handler
  `cr_ipc_signal_handler` (`vGPU.cpp:828`) calls the `ipc_*` functions from
  `ipc_hooks.cpp` instead.
- **Why it matters:** they exist so any stale external caller fails soft with a
  pointer to the replacement, instead of breaking the link.

### LD_PRELOAD hooks: `ncclCommInitRank`, `ncclCommInitRankConfig`, `ncclCommDestroy`, `ncclCommFinalize` (`extern "C"`, `nccl_hooks.cpp:253-331`)
- **Purpose:** each resolves the real function via `find_real_nccl_sym` (cached in a
  function-local static), logs the interception, forwards the call, and:
  - the two `Init` hooks register the new communicator *after* a successful return;
  - `ncclCommInitRankConfig` falls back to plain `ncclCommInitRank` if the config
    variant does not exist (older NCCL);
  - `ncclCommDestroy` unregisters *before* forwarding (the handle is about to die);
  - `ncclCommFinalize` deliberately does **not** unregister — finalize flushes a
    communicator but does not destroy it; only `ncclCommDestroy` does
    (`nccl_hooks.cpp:328-330`).
- **Called by:** the application's NCCL usage (PyTorch's `ProcessGroupNCCL`, vLLM),
  intercepted via LD_PRELOAD — no GPU-CR code calls these.
- **Why it matters:** if a hook cannot resolve the real symbol, `ncclCommInitRank` and
  `ncclCommDestroy` return 3 (`ncclInternalError`) — the application will see NCCL
  failing, which is the correct loud failure; `ncclCommFinalize` instead skips quietly.

### `cr_register_nccl_comm(void*)` / `cr_unregister_nccl_comm(void*)` (`nccl_hooks.cpp:335-341`)
- **Purpose:** `extern "C"` escape hatch so an application can register communicators
  *manually* — e.g. from Python via `ctypes` — when interception is impossible (NCCL
  statically linked, or symbols unreachable even by `find_real_nccl_sym`).
- **Called by:** no in-tree caller; intended for application/test code loading
  `vGPU-NVIDIA.so` and calling in by symbol name.
- **Why it matters:** it is the documented fallback for exotic link setups; keep the C
  linkage and `void*` signature stable, since out-of-tree callers bind to it by name.

---

## How vendor selection works

Vendor selection happens **twice**, and both answers must agree:

1. **Build time (CMake `GPU_VENDOR`).** `CMakeLists.txt:14-38` resolves `GPU_VENDOR`
   from `-DGPU_VENDOR=...`, else the `GPU_VENDOR` environment variable, else defaults
   to `NVIDIA`. For NVIDIA it points includes/libs at `/usr/local/cuda` and links
   `cudart cuda`; for AMD it uses `/opt/rocm`, links `amdhip64`, and — crucially —
   adds the compile definition `__HIP_PLATFORM_AMD__=1` (`CMakeLists.txt:34`). That
   macro is the *only* compile-time vendor switch in the C++ sources
   (`gpu_factory.cpp:3-7`, `:22`, plus `#if !defined(__HIP_PLATFORM_AMD__)` guards in
   `vGPU.cpp` and `cr_client.cpp`). The source list is then filtered
   (`CMakeLists.txt:68-78`): an NVIDIA build excludes `src/GPUs/AMD/`, and an AMD
   build excludes `src/GPUs/NVIDIA/` *and* the CUDA-only `ipc_hooks.cpp`,
   `ipc_fd_exchange.cpp`, and `nccl_hooks.cpp`. The output library is named
   `vGPU-${GPU_VENDOR}.so` — so a given `.so` contains exactly one backend.

2. **Run time (`createGPU()` factory).** When the first checkpoint signal triggers
   `init_CR()` (`src/vGPU.cpp:701`), `createGPU()` reads the `GPU_VENDOR` environment
   variable of the *application process* and instantiates the one compiled-in backend
   — or `exit(EXIT_FAILURE)`s if the variable is unset or names the other vendor
   (`gpu_factory.cpp:14-38`).

Practical consequence: `export GPU_VENDOR=NVIDIA` (or `AMD`) must be in the
application's environment at launch, matching the `.so` you preloaded. Getting it
wrong does not fall back — it kills the application from inside a signal handler,
which can look like a mysterious crash if you are not watching stderr.

---

## Gotchas for maintainers

**Error-handling conventions are inconsistent — know which regime you are in.**
- NVIDIA wrapper methods built on `CU_CHECK`/`CUDA_CHECK_RET` call
  `exit(EXIT_FAILURE)` on any CUDA error (`nv.cpp:28-44`); the process dies, mid
  signal handler if that is where you are. AMD methods mostly log and return -1.
- Exceptions to the NVIDIA rule: `registerHostMemory` (soft-fails, and clears the
  sticky CUDA error with `cudaGetLastError()` — copy that pattern if you add
  tolerated failures), `releasePhysicalMemory` (returns -1), and
  `pushContext`/`popContext` (return -1).
- Return codes are inconsistent across vendors for the same situation:
  `releasePhysicalMemory` on an untracked pointer is -1 on NVIDIA but 0 ("skip") on
  AMD. `vGPU.cpp` exits on -1 from release, so the same bug is fatal on one vendor
  and silent on the other.
- `allocate`/`deallocate` throw C++ exceptions — from a `void*`-returning C-ish call
  chain there is nothing to catch them, so reaching them is `std::terminate`.

**NVIDIA context push/pop discipline.**
- All GPU work in the signal handler is bracketed by `gpu->pushContext()` ...
  `gpu->popContext()` (`vGPU.cpp:758`, `:820`). Any early `return` you add between
  them leaks a pushed context on that thread; keep the bracket intact.
- `pushContext` prefers `g_pytorch_context`, captured opportunistically by the
  `cudaMalloc` hook (`nv.cpp:333-339`, `:381`, `:398`). If the application never calls
  `cudaMalloc` through the hook (custom allocator, pure driver-API app), the capture
  never happens and checkpoint runs in GPU-CR's own primary context — allocations made
  in another context will not be visible. The kernel-launch hooks in `ipc_hooks.cpp`
  (`:933-1197`) also push this captured context to fix launch-time mismatches; the
  two files must keep agreeing on what `g_pytorch_context` means.
- AMD has no contexts; the inherited no-op defaults (`GPU.h:76-77`) are correct there.

**AMD / CRIU environment requirements.**
- `AMD_CKPT_DIR` must be set in the environment of whichever process runs CRIU —
  today that is `cr_client` (`cr_client.cpp:181-186`, `:223-228`); both it and the
  (dead) `amd::externalCheckpoint` hard-exit without it.
- CRIU runs under `sudo` with `LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu` and
  plugins from `-L /usr/local/lib/criu`; all three of those paths are hard-coded and
  environment-specific.
- CRIU dumps a **process tree** rooted at the pid given to `-t`. `cr_client` therefore
  takes two pids: `-p <pid>` (the process that receives GPU-CR's signals / owns the
  shared-memory channel) and `-m <criu_pid>` (the tree root handed to `criu dump -t`,
  defaulting to `-p`'s value — `cr_client.cpp:102-104`, `:125`, `:195`). For a Python
  app with worker children you typically must pass the *parent* pid as `-m`, or CRIU
  will refuse / miss children. Restore requires the original pids to be free.
- The duplicated CRIU command lines in `amd.cpp` have already drifted from the live
  ones in `cr_client.cpp` (missing `--ghost-limit`, `--ext-unix-sk`,
  `--restore-detached`); do not "fix" a CRIU flag in only one place — or better,
  delete the dead methods and route through one implementation.

**Allocation-tracking semantics differ per vendor.**
- NVIDIA stores the **unrounded** requested size in `allocated_memory` and re-rounds
  with `ROUND_UP_2MB` inside release/remap; AMD stores the **already-rounded** size.
  Code that mixes the two assumptions will over- or under-copy. (The selective
  checkpoint path in `vGPU.cpp:285-291` documents its own handling of this.)
- The CUDA hooks serialize against checkpoints with `gpu_mem_mutex`; the HIP hooks
  take **no lock**, so allocation-vs-checkpoint races are possible on AMD.
- `amd::releasePhysicalMemory` leaves the released handle in `global_handle_map`
  (no `erase`); `amd::remapPhysicalMemory` does not handle an already-mapped pointer
  (NVIDIA's does). And `amd::getMemoryMap()` returns the empty private `memory_map_`
  instead of the global `allocated_memory` — currently harmless only because nothing
  calls `getMemoryMap()` anywhere.
- `cudaFree`/`hipFree` free the virtual address too (`cuMemAddressFree`), unlike
  `releasePhysicalMemory`. An application that frees memory *between* checkpoint and
  restore invalidates the address the restore path is about to remap; the ordering
  guarantees come only from the coordinator protocol.

**NCCL communicator lifecycle pitfalls.**
- Registration happens only if interception works. PyTorch loads NCCL by `dlopen`
  after our preload, so `RTLD_NEXT` alone fails; `find_real_nccl_sym`'s strategies 2-5
  exist for exactly this. If tracked-comm counts stay at 0 under PyTorch, check which
  strategy fired in the stderr log, `CR_NCCL_LIB`, and whether NCCL is statically
  linked (then only `cr_register_nccl_comm` via ctypes can help).
- `ncclCommFinalize` must **not** unregister (the comm still exists until
  `ncclCommDestroy`); preserve that asymmetry.
- The suspend/resume API is deprecated no-ops returning 1 — never re-wire the
  multi-GPU flow to it; the real teardown/rebuild lives in `ipc_hooks.cpp`
  (`ipc_teardown_all_imports` / `ipc_rebuild_all_imports`, driven by
  `cr_ipc_signal_handler` in `vGPU.cpp:828`). Signals named `CR_NCCL_*` in
  `common.h:47-48` are legacy aliases for the IPC signals.
- Anything collective (in the old design, `ncclCommSuspend`) must be invoked on **all
  ranks simultaneously** or every rank hangs; the coordinator signals all processes
  before waiting for any. Keep that pattern for any future per-communicator
  operation you add.
- On AMD builds none of this file exists (`CMakeLists.txt:76`) — do not add
  unconditional calls to `nccl_*` from shared code without an `#ifdef` or a stub.

**Dead-but-plausible code.** `externalCheckpoint`/`externalRestore` (both vendors),
`allocate`/`deallocate`, `getMemoryMap`, `getVendor`, `nccl_get_comm_count`, and
`amd::ensureHipInitialized` all currently have zero callers (grep-verified as of this
writing). When you need the functionality, check the coordinator (`cr_client.cpp`)
first — that is where the live version usually is.
