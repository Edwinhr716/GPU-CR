# 02 — The Core Preload Library (`vGPU.cpp` and its support files)

## What this subsystem does

This is the heart of GPU-CR: the code that lives *inside* the GPU application's own
process and does the actual checkpoint and restore work. When you start a GPU program
(for example vLLM) with `LD_PRELOAD=vGPU-NVIDIA.so`, this library is loaded into that
process. It registers signal handlers so that an external CLI (`cr_client` or
`multi_cr_client`) can poke the process from the outside, reads a "what do you want me
to do" message from a shared control file, and then either **checkpoints** (copies every
tracked GPU allocation into a big host-memory dump area and frees the physical VRAM
behind it, while keeping the GPU virtual addresses reserved) or **restores** (re-creates
the physical VRAM mappings at the same addresses and copies the data back). The files
covered here are the orchestration layer: `src/vGPU.cpp` (signal handlers and the
copy pipelines), `src/common.h` (shared constants and on-disk/in-memory layouts),
`src/comm/` (the tiny control-word protocol between CLI and library), and
`src/backend/` (the large memory-mapped dump buffer). The actual CUDA/HIP calls and the
interception of `cudaMalloc`/`cudaFree` live in `src/GPUs/` and are documented
elsewhere; here they appear only through the abstract `GPU` interface.

> **Background: LD_PRELOAD.** On Linux, when a program starts, a system component
> called the *dynamic linker* loads the shared libraries (`.so` files) the program
> needs. The environment variable `LD_PRELOAD` tells the dynamic linker to load an
> extra library *first*, before all others. Because symbol lookup takes the first
> definition it finds, a preloaded library can define its own `cudaMalloc` and have
> the application call *ours* instead of the real one — without changing a single
> line of the application. That is how this library "injects" itself.

### Checkpoint control flow at a glance

```
  cr_client (separate process)                GPU application process (vLLM, ...)
  ---------------------------                 -----------------------------------
  1. write CKPT_MSG into the        mmap'd    vGPU-NVIDIA.so, loaded via LD_PRELOAD
     shared control file  ------------------>   control->signal == CKPT_MSG
  2. kill(pid, SIGUSR1)   -- POSIX signal -->  cr_signal_handler()      [vGPU.cpp]
  3. poll control file                            |  reads msg via comm->recv_msg()
     until it reads FINISH_MSG                    |  gpu->syncAllKernels()
          ^                                       v
          |                                    ckpt()                    [vGPU.cpp]
          |                                       |  for each tracked allocation:
          |                                       |    gpu->memcpyAsync(GPU -> staging_buf)
          |                                       |    memcpy_multi()  <- 4 worker threads
          |                                       |      (staging_buf -> shared_mem_fs dump)
          |                                       |  gpu->releasePhysicalMemory(ptr)
          |                                       v
          +--------- comm->send_msg(FINISH_MSG)  done
                                                  |
  4. run cuda-checkpoint / CRIU                   |
     to freeze the rest of the process           GPU VRAM is now free; virtual
                                                 addresses are still reserved
```

Two things to notice, because they surprise most newcomers:

* The signal itself carries **no data**. It is only a doorbell. The actual command
  (`CKPT_MSG`, `RESTORE_MSG`, `SELECTIVE_CKPT_MSG`, ...) and any arguments (the list of
  selective regions) travel through a small memory-mapped control file that both
  processes open (`ShareMemComm`).
* All of the heavy work — CUDA calls, gigabytes of memcpy — runs **inside the signal
  handler itself**, on whatever application thread happened to receive the signal. The
  only extra threads are the short-lived `memcpy_multi` copy workers. (See "Gotchas"
  for why this is a deliberate but fragile choice.)

> **Background: POSIX signals and signal handlers.** A *signal* is the operating
> system's way of interrupting a running process asynchronously — think of it as a
> software interrupt with a small integer number (e.g. `SIGUSR1`). Any process with
> permission can send one with the `kill(pid, signum)` system call ("kill" is a
> misleading historical name; most signals do not kill anything). A process can
> register a *signal handler* — a function the kernel will call, pausing whatever the
> process was doing, whenever that signal arrives. That is the mechanism `cr_client`
> uses to make code run inside the GPU application without the application's
> cooperation.

---

## `src/vGPU.cpp` — signal handlers, checkpoint/restore pipelines, initialization

This is the main translation unit of the preload library. It owns the global
singletons (`comm`, `backend`, `gpu`, the staging buffers), defines the two signal
handlers registered at library load, and implements the four data-movement pipelines:
full checkpoint, selective checkpoint, full restore, selective restore. Everything in
this file runs inside the *target application's* process.

### `init()` (the `__attribute__((constructor))` at the bottom of the file)

**Purpose:** Registers the signal handlers. It installs `cr_signal_handler` for
`CR_INIT_SIGNAL` (`SIGRTMAX`), `CR_CKPT_SIGNAL` (`SIGUSR1`) and `CR_RESTORE_SIGNAL`
(`SIGUSR2`); installs `cr_ipc_signal_handler` for the multi-GPU signals
`CR_IPC_TEARDOWN_SIGNAL` (`SIGRTMAX-1`) and `CR_IPC_REBUILD_SIGNAL` (`SIGRTMAX-2`);
and installs a small diagnostic lambda for `CR_IPC_VALIDATE_SIGNAL` (`SIGRTMAX-3`)
that just calls `ipc_validate_all_mappings`. It deliberately does *not* initialize
the checkpoint machinery — that is deferred to `init_CR()` so that a process which is
never checkpointed pays no cost.

> **Background: constructor attributes.** A function marked
> `__attribute__((constructor))` is called automatically by the dynamic linker when
> the shared library is loaded — before the application's `main()` runs. Nobody in
> this codebase calls `init()` by name; the loader does it. This is the standard way
> for an `LD_PRELOAD` library to set itself up.

> **Background: real-time signals.** `SIGRTMAX`, `SIGRTMAX-1`, etc. are "real-time"
> signals — a block of extra signal numbers Linux provides beyond the classic named
> ones, precisely so applications can define their own meanings without colliding
> with signals the application already uses (Python, for example, uses `SIGUSR` far
> less often than one might fear, but RT signals are safer still).

**Called by:** the dynamic linker, at library load time (see Background above). No
call site exists in the source, by design.

**Why it matters:** without it, `kill()` from `cr_client` would hit the default signal
disposition — for `SIGUSR1`/`SIGUSR2` that default is *terminate the process*. So a
missing/failed constructor doesn't just disable checkpointing; the first checkpoint
attempt would kill the inference server.

### `cr_signal_handler(int signum)` (vGPU.cpp:728)

**Purpose:** The single-GPU control entry point. Step by step:

1. Ignores any signal that is not one of the three CR signals (defensive check).
2. For `CR_INIT_SIGNAL`: runs `init_CR()` if not already initialized, replies
   `FINISH_MSG`, and returns. This lets an operator pre-pay the expensive setup
   (mapping ~27 GB of hugepages) before the first real checkpoint.
3. For checkpoint/restore signals: lazily runs `init_CR()` if needed, then reads the
   real command with `comm->recv_msg()` — remember, the signal number alone only
   distinguishes "checkpoint-ish" from "restore-ish"; the message narrows it to
   `CKPT_MSG` vs `SELECTIVE_CKPT_MSG`, or `RESTORE_MSG` vs `SELECTIVE_RESTORE_MSG`.
4. Wraps the work in `gpu->pushContext()` / `gpu->popContext()` (making sure the
   CUDA driver context is current on the signal-handling thread).
5. For checkpoints it first calls `gpu->syncAllKernels()` — wait for all in-flight
   GPU work to finish so the memory being dumped is quiescent — then calls `ckpt()`
   or `ckpt_selective()`. For the full-checkpoint path on NVIDIA it additionally
   calls `ipc_disable_all_peer_access()` (defined in `src/ipc_hooks.cpp`) because
   peer-to-peer GPU access is driver state that the external `cuda-checkpoint` tool
   cannot re-create.
6. For restores it calls `restore_ptr_and_content()` or
   `restore_ptr_and_content_selective()`, then (NVIDIA, full path) re-enables P2P.
7. Finally writes `FINISH_MSG` back through `comm->send_msg()`, which is what
   releases the polling loop in the CLI.

**Called by:** the kernel, when an external process sends a signal. The real senders
are `coordinator/cr_client.cpp:134/150/165/173/251/263` (each does
`comm->send_msg(...)` then `kill(pid, CR_*_SIGNAL)`) and
`coordinator/multi_cr_client.cpp:292-293` via its `broadcast_and_wait()` helper
(used at `multi_cr_client.cpp:576`, `:627`, `:653`). It is registered as a handler at
`src/vGPU.cpp:1084-1086`.

**Why it matters:** this function *is* the external API of the preload library. Every
checkpoint and restore in the single-GPU flow funnels through it; the
`send_msg(FINISH_MSG)` at its end is the only completion notification the CLI ever
gets.

### `cr_ipc_signal_handler(int signum)` (vGPU.cpp:828)

**Purpose:** The multi-GPU companion handler, used when several processes share GPU
memory via CUDA IPC (inter-process communication handles, e.g. under NCCL). CUDA IPC
mappings cannot survive a freeze/restore, so they must be torn down before checkpoint
and rebuilt after restore. The handler reads one of three messages:

* `IPC_TEARDOWN_MSG` — synchronizes the GPU, then delegates to the `ipc_*` helpers
  (from `src/ipc_hooks.cpp`) to unmap all imported IPC memory, save the *contents* of
  exported allocations into an anonymous `mmap`'d host buffer
  (`g_ipc_export_data_buf`), tear the exports down, tear down IPC events, do the same
  save-and-teardown for non-exported `cuMem` allocations
  (`g_local_alloc_data_buf`), and finally disable P2P access. Extensive timing
  logging brackets each phase.
* `IPC_EXPORT_MSG` — restore phase 3a: rebuilds the exported allocations at their
  original addresses from `g_ipc_export_data_buf`, rebuilds local allocations,
  publishes the new export handles into a reserved `IpcRebuildShmBlock` slot inside
  the shared dump buffer (so peer processes can find them), and starts a Unix-domain
  socket server (`uds_fd_server_start()`, from `src/ipc_fd_exchange.cpp`) that can
  hand file descriptors to peers.
* `IPC_IMPORT_MSG` — restore phase 3b: reads the *peer's* `IpcRebuildShmBlock` from
  the shared buffer, re-imports those mappings, stops the UDS server, validates all
  mappings, and re-enables P2P.

In every case it ends with `comm->send_msg(FINISH_MSG)`.

**Called by:** the kernel, on signals sent by `coordinator/multi_cr_client.cpp:569`
(`IPC_TEARDOWN_MSG` + `CR_IPC_TEARDOWN_SIGNAL`), `:633` (`IPC_EXPORT_MSG` +
`CR_IPC_REBUILD_SIGNAL`) and `:639` (`IPC_IMPORT_MSG` + `CR_IPC_REBUILD_SIGNAL`).
Registered at `src/vGPU.cpp:1089-1090`. `cr_client` (single-GPU) never triggers it.

**Why it matters:** without this teardown/rebuild sequence, `cuda-checkpoint` fails or
the restored processes crash the moment NCCL touches a stale IPC mapping. It is the
piece that makes multi-GPU (tensor-parallel) checkpointing possible without modifying
NCCL.

### `init_CR()` (vGPU.cpp:673)

**Purpose:** One-time lazy initialization of the whole checkpoint stack, guarded by
the `CR_initialized` flag. Steps:

1. `get_id()` assigns this process a small integer CR ID (see below).
2. Writes a `pid_map_<pid>` file (containing the ID) into the control directory
   (`$EXPORT_FILE_PATH`, default `/mnt/huge-ckpt`) so external tooling can translate
   a PID into a CR ID.
3. Constructs and sets up the three singletons: `comm = new ShareMemComm(getpid())`
   (control channel, keyed by PID), `backend = new ShareMem(id)` (the big dump
   buffer, keyed by CR ID), and `gpu = createGPU()` (vendor autodetection —
   `src/GPUs/gpu_factory.cpp:13` — returning the NVIDIA or AMD implementation of the
   abstract `GPU` interface in `src/GPUs/GPU.h`).
4. Takes the backend's 2 GB host buffer (`backend->get_host_buffer()`), tries to
   register it with the GPU driver as *pinned* memory
   (`gpu->registerHostMemory(...)`; failure is tolerated, just slower), and slices
   it into the two 1 GB entries of the global `staging_buf` array.

> **Background: pinned memory.** GPUs copy data to/from host RAM using DMA hardware,
> which needs the memory to stay at a fixed physical address. "Pinning" (page-locking)
> tells the OS never to move or swap those pages, which lets the GPU driver use its
> fast asynchronous DMA path. Unpinned memory still works but the driver has to copy
> through an internal bounce buffer.

**Called by:** `cr_signal_handler` (`src/vGPU.cpp:741` on `CR_INIT_SIGNAL`, `:754`
lazily before ckpt/restore) and `cr_ipc_signal_handler` (`src/vGPU.cpp:834`). It is
*not* run at library load; a process that never gets a CR signal never allocates the
dump buffer.

**Why it matters:** every other function in this file dereferences `comm`, `backend`,
`gpu` or `staging_buf` without null checks. If `init_CR()` were skipped or reordered,
the first checkpoint would segfault. Its lazy-but-idempotent design is also why the
`-i` "init" CLI flag exists: to move the multi-second hugepage mapping cost out of the
critical checkpoint path.

### `get_id()` (vGPU.cpp:647)

**Purpose:** Allocates a process-unique small integer by opening (creating if needed)
the file `<ctl_dir>/control`, sizing it with `ftruncate`, mapping it with `mmap`, and
performing an atomic `fetch_add(1)` on the first `int` in the mapping. Because the
mapping is `MAP_SHARED` and file-backed, every GPU process on the machine increments
the *same* counter, so each gets a distinct ID (0, 1, 2, ...). The ID selects which
dump file this process will use (`ckpt-<id>.data` / `/mnt/huge-ckpt/<id>`).

> **Background: mmap and shared memory.** `mmap` asks the kernel to place a file's
> contents directly into the process's address space, so reading/writing memory reads/
> writes the file — no `read()`/`write()` calls needed. If two processes map the same
> file with `MAP_SHARED`, they see each other's writes immediately: the file becomes
> a piece of *shared memory*. `ftruncate` sets the file's size first, because
> touching a mapped page beyond the end of the file raises a bus error. This
> file-backed-mmap pattern is the backbone of all communication in GPU-CR.

**Called by:** `init_CR()` (`src/vGPU.cpp:680`) only. (`multi_cr_client.cpp:209`
mentions it in a comment — the CLI must *not* assume PIDs and CR IDs are ordered the
same way, precisely because this counter hands out IDs in signal-arrival order.)

**Why it matters:** if two processes ever received the same ID they would write their
checkpoints into the same dump file and destroy each other's data. The atomic counter
in shared memory is the only thing preventing that.

### `ckpt()` (vGPU.cpp:65)

**Purpose:** The full checkpoint pipeline. Dumps *every* allocation recorded in the
global `allocated_memory` map into the backend dump buffer, then frees the physical
VRAM. Step by step:

1. Gets the dump buffer (`backend->get_tmp_buf()`) and treats its start as a
   `shared_mem_fs` header (see `common.h` below); resets `file_num` to 0 and
   `current_offset` to `ROUND_UP_2MB(sizeof(shared_mem_fs))` — the payload area
   starts at the first 2 MB boundary after the header.
2. Takes the global `fs_mutex` (manually) and `gpu_mem_mutex` (via `lock_guard`),
   then creates a dedicated GPU stream and event.

   > **Background: CUDA streams and events.** A *stream* is a work queue on the GPU:
   > operations submitted to the same stream run in order, and `memcpyAsync` returns
   > immediately, letting the CPU do other work while the copy happens. An *event* is
   > a marker you drop into a stream; `synchronizeEvent` blocks the CPU until the GPU
   > has passed that marker. Together they let this code overlap GPU-to-host DMA with
   > CPU-side memcpy — a classic double-buffering pipeline.

3. For each `(ptr, size)` entry in `allocated_memory`: rounds the size up with
   `ROUND_UP_2MB` (note: the *full* checkpoint still rounds; only the selective path
   was changed to unrounded — see Gotchas), appends a `shared_mem_file` record to the
   header (`ptr`, `start_offset`, `size`), bounds-checks against `SHM_SIZE` and
   `MAX_FILE_NUM` (hard `exit(-1)` on overflow), and streams the data out in chunks:
   `gpu->memcpyAsync(...DeviceToHost...)` into the *current* 1 GB staging buffer;
   whenever a staging buffer fills, it waits (via the event) for the *previous*
   buffer's DMA to complete and `memcpy_multi`'s that previous buffer into the dump
   area while the GPU keeps filling the other one. The `current_buf & 1` indexing is
   the ping-pong between `staging_buf[0]` and `staging_buf[1]`.
4. After the loop: drains the last full buffer, `synchronizeStream`, copies the final
   partial buffer (`buf_offset` bytes), and asserts
   `des_offset + buf_offset == fs->current_offset` (the pipeline's book-keeping and
   the header must agree exactly).
5. Calls `gpu->releasePhysicalMemory(ptr)` for every allocation — this frees the
   VRAM pages while keeping the GPU *virtual* address reservation, so pointers held
   by the application (PyTorch tensors) stay valid-looking and can later be remapped
   to fresh physical memory at the same addresses.
6. Prints a timing breakdown and returns the total bytes dumped.

**Called by:** `cr_signal_handler` only (`src/vGPU.cpp:786`, on `CKPT_MSG`).

**Why it matters:** this is the function that actually gets the model weights and KV
cache off the GPU. Its release step is what makes the VRAM reusable by another tenant
while the checkpointed process sleeps; its address-preserving trick is what makes
restore transparent to PyTorch.

### `ckpt_selective(const selective_cr_request* req)` (vGPU.cpp:236) and `find_containing_allocation(...)` (vGPU.cpp:225)

**Purpose:** Same pipeline as `ckpt()`, but only for a caller-chosen subset of
memory. The CLI passes up to `MAX_SELECTIVE_REGIONS` (4096) `ptr:size` pairs through
the control file (`scomm->control->selective_req`). For each requested pointer,
`find_containing_allocation()` linearly scans `allocated_memory` to find the
allocation block that *contains* it (the caller may pass an interior tensor address,
not the block base); unknown pointers are logged and skipped. The matched base
pointers are deduplicated in a `std::set`, then each whole block is dumped and its
physical memory released, exactly as in `ckpt()`.

One deliberate difference (the recent "unrounded dumps" change, commit `7a131b4`,
design doc `docs/design-unrounded-dumps.md`): the dumped `size` is the **raw**
`alloc_size` from the tracking map, *not* `ROUND_UP_2MB(alloc_size)`. The rounding
padding was never visible to the application, and both `releasePhysicalMemory` and
`remapPhysicalMemory` re-round internally, so dumping it was pure waste — up to ~48×
amplification for LoRA-scale tensors.

**Called by:** `ckpt_selective` from `cr_signal_handler` (`src/vGPU.cpp:766`, on
`SELECTIVE_CKPT_MSG`); `find_containing_allocation` from `ckpt_selective`
(`src/vGPU.cpp:277`) only.

**Why it matters:** selective checkpoint is what makes fast tenant-switching (swap out
one LoRA adapter's state, not 20 GB of shared base weights) economical. Without
`find_containing_allocation`, callers would need to know exact `cudaMalloc` base
addresses, which PyTorch does not expose.

### `restore_ptr_and_content()` (vGPU.cpp:409) and `restore_ptr_and_content_selective()` (vGPU.cpp:528)

**Purpose:** The reverse pipelines. Both read the `shared_mem_fs` header that the
matching checkpoint wrote (they trust it completely — nothing is passed in; the
selective variant does not even look at `selective_req`, it simply restores whatever
the last selective checkpoint recorded). Steps:

1. For every file entry, `gpu->remapPhysicalMemory(ptr, size)` — allocate fresh
   physical VRAM and map it at the *original* virtual address (the GPU vendor layer
   rounds the size back up to 2 MB internally, which is why unrounded header sizes
   are safe).
2. Then the mirror-image double-buffered pipeline: `memcpy_multi` from the dump area
   into a staging buffer, `gpu->memcpyAsync(...HostToDevice...)` out of it, event
   synchronization to overlap the two, final `synchronizeStream`.
3. Print a timing breakdown, return total bytes.

The two functions are near-verbatim copies of each other (only log prefixes differ) —
a refactoring opportunity, and a hazard: fixes applied to one must be mirrored in the
other.

**Called by:** `cr_signal_handler` only — `src/vGPU.cpp:806` (`RESTORE_MSG`) and
`src/vGPU.cpp:775` (`SELECTIVE_RESTORE_MSG`).

**Why it matters:** these functions are the reason a restored process can keep using
its old pointers. If remap happened at a different address, every tensor in the
application would dangle; if the data copy were skipped, the model would compute
garbage.

### `memcpy_multi(void* dest, void* src, size_t size)` (vGPU.cpp:48)

**Purpose:** A parallel `memcpy`: splits the range into `NUM_COPY_THREADS` (4) equal
chunks, spawns one `std::thread` per chunk, and joins them. Plain single-threaded
`memcpy` cannot saturate the bandwidth between the pinned staging buffer and the
hugepage dump area; four threads roughly quadruple it, and this copy sits on the
critical path of every checkpoint and restore. These are the only "worker threads" in
the subsystem — short-lived, spawned per copy.

**Called by:** `ckpt()` (`src/vGPU.cpp:153/172/185`), `ckpt_selective()`
(`:337/357/370`), `restore_ptr_and_content()` (`:462/494`), and
`restore_ptr_and_content_selective()` (`:580/612`). Declared in `src/common.h:69` but
no other translation unit uses it.

**Why it matters:** checkpoint latency is dominated by host-side copies; the design
doc's GB/s numbers assume this parallel copy. Removing it would multiply switch times.

---

## `src/common.h` — constants, layouts, and the shared vocabulary

This header is included by both the preload library and the coordinator CLIs, which
makes it the *protocol definition*: any struct or constant here is implicitly an ABI
contract between two different binaries reading the same shared memory. There are no
functions here (only the `memcpy_multi` declaration), so this section describes the
groups of definitions.

### Sizes and rounding: `HUGE_PAGE_SIZE`, `ROUND_UP_2MB`, `SHM_SIZE`, `STAGING_BUF_*`, `MAX_FILE_NUM`

`HUGE_PAGE_SIZE` is 2 MB and `ROUND_UP_2MB(x)` rounds `x` up to the next multiple of
2 MB using bit masking.

> **Background: hugepages.** The CPU normally manages memory in 4 KB pages, and every
> page needs a translation entry. For a 25 GB buffer that is ~6.5 million entries —
> slow to set up and hard on the TLB cache. Linux "hugepages" are 2 MB pages: 512×
> fewer entries, meaningfully faster large copies. They must be reserved by the
> administrator ahead of time and are typically exposed via a special filesystem
> (*hugetlbfs*) mounted at a path like `/mnt/huge-ckpt`; creating and mmap'ing a file
> there gives you hugepage-backed shared memory. That is exactly what the backend
> does. 2 MB is also the granularity of CUDA's virtual memory management, which is
> why the same constant shows up in GPU-side rounding.

`SHM_SIZE` is the per-process dump buffer size, `SHM_SIZE_GB << 30` bytes with a
default of 25 GB, overridable at compile time (`cmake .. -DSHM_SIZE_GB=40`). It is
consumed by `ShareMem::setup()` (buffer creation, `src/backend/mmap_backend.cpp:44/63`),
by the overflow checks in `ckpt()`/`ckpt_selective()` (`src/vGPU.cpp:118/302`), and by
`multi_cr_client.cpp:183` which maps the same files from the CLI side.
`STAGING_BUF_SIZE` (1 GB) × `STAGING_BUF_NUM` (2) sizes the pinned ping-pong buffers.
`MAX_FILE_NUM` (4096) caps how many allocations one dump can hold. `COPY_THRESHOLD`
and `NUM_COPY_THREADS` tune the copy pipeline.

### Signal numbers: `CR_INIT_SIGNAL`, `CR_CKPT_SIGNAL`, `CR_RESTORE_SIGNAL`, `CR_IPC_*_SIGNAL`

The agreed doorbell numbers: `SIGRTMAX` for init, `SIGUSR1` for checkpoint-flavored
requests, `SIGUSR2` for restore-flavored, and `SIGRTMAX-1/-2/-3` for the multi-GPU
IPC teardown/rebuild/validate flow (`CR_NCCL_SUSPEND/RESUME_SIGNAL` are legacy
aliases for the first two). Used by the handlers' registration in `src/vGPU.cpp:1084-1096`
and by every `kill()` in `coordinator/cr_client.cpp` and
`coordinator/multi_cr_client.cpp`. Change one side without the other and signals
either get ignored or terminate the target.

### `allocated_memory`, `allocated_memory_type`, `gpu_mem_mutex` (extern declarations)

`allocated_memory` is the master map of live GPU allocations (`ptr -> requested
size`). It is *defined* in the vendor layers (`src/GPUs/NVIDIA/nv.cpp:13`,
`src/GPUs/AMD/amd.cpp:11`) and *written* by the intercepted `cudaMalloc`/`cudaFree`
(`nv.cpp:458/507`) and their HIP equivalents.

> **Background: dlsym interception.** The preload library defines its own
> `extern "C" cudaMalloc` (`src/GPUs/NVIDIA/nv.cpp:326`). Because of `LD_PRELOAD`
> ordering, the application binds to this one. Inside, the hook uses
> `dlsym(RTLD_NEXT, "cudaFree")`-style lookups to find the *real* CUDA function
> further down the library search order, so it can do its book-keeping and then
> delegate. `RTLD_NEXT` literally means "the next library after me that defines this
> symbol". This is how the map gets populated without the application knowing.

`allocated_memory_type` records how each pointer was allocated (0 = plain runtime
malloc, 1 = CUDA/HIP virtual-memory-management), which the vendor layer needs to
release/remap correctly. `gpu_mem_mutex` serializes the map against the pipelines
(see "Key global state").

### Layout structs: `shared_mem_file`, `shared_mem_fs`, `selective_cr_region`, `selective_cr_request`, `signal_controls`

`shared_mem_fs` is the header at offset 0 of the dump buffer — a tiny table-of-
contents "filesystem": `file_num`, `current_offset` (end of payload), and an array of
`MAX_FILE_NUM` `shared_mem_file` entries (`ptr`, `start_offset`, `size`).
`selective_cr_request` is the argument block for selective operations: up to
`MAX_SELECTIVE_REGIONS` (4096) `{ptr, size}` pairs plus a count. `signal_controls` is
the entire content of the per-PID control file: one `uint32_t signal` word (the
message/handshake slot) followed by an embedded `selective_cr_request`. `cr_client`
writes the request directly into the mapping (`coordinator/cr_client.cpp:148/163`)
and the handler reads it in place (`src/vGPU.cpp:761`) — no copy, no serialization.

**Why this file matters:** it is compiled into at least three binaries
(`vGPU-*.so`, `cr_client`, `multi_cr_client`). Any layout or constant change here
requires rebuilding *all* of them together, or the shared-memory structures silently
misalign.

---

## `src/comm/comm.h` and `src/comm/share_mem.cpp` — the control channel

This pair defines how the CLI and the preload library exchange the one word of
control state that the signal cannot carry. `comm.h` declares an abstract `Comm`
interface plus the message constants (`INIT_MSG`, `CKPT_MSG`, `RESTORE_MSG`,
`FINISH_MSG`, `IPC_TEARDOWN_MSG`, `IPC_EXPORT_MSG`, `IPC_IMPORT_MSG`,
`SELECTIVE_CKPT_MSG`, `SELECTIVE_RESTORE_MSG`, and legacy NCCL aliases);
`share_mem.cpp` provides the only real implementation, `ShareMemComm`. The design is
intentionally primitive: a single shared `uint32_t`, written by whichever side is
"speaking", polled by whichever side is waiting.

### `Comm::*` base-class methods (share_mem.cpp:3-21)

**Purpose:** All are empty stubs (`recv_msg` returns 0, `is_finished` returns
`false`). The base class exists purely as an interface so other transports could be
added; note the methods are not pure-virtual, so forgetting an override fails
silently at runtime rather than at compile time. Grouped here because they are
trivial.

**Called by:** only through the derived class — every construction site instantiates
`ShareMemComm` (`src/vGPU.cpp:697`, `coordinator/cr_client.cpp:127`,
`coordinator/multi_cr_client.cpp:153`).

**Why it matters:** mostly as a trap — if a new call site ever holds a plain `Comm`
constructed directly, all messaging becomes a silent no-op and every wait loop hangs.

### `ShareMemComm::ShareMemComm(pid_t pid)` and `setup()` (share_mem.cpp:23-49)

**Purpose:** The constructor just stores the PID. `setup()` opens (creating if
necessary) the control file `<ctl_dir>/control-<pid>` where `ctl_dir` is
`$EXPORT_FILE_PATH` or `/mnt/huge-ckpt`, `ftruncate`s it to `HUGE_PAGE_SIZE`, and
`mmap`s it `MAP_SHARED` as a `signal_controls*` stored in the public `control`
member. Because both the CLI and the target process construct a `ShareMemComm` for
the *same PID*, they map the *same file* and thus share the same `signal_controls`
struct. Any failure is fatal (`exit`).

**Called by:** constructed and set up in `init_CR()` (`src/vGPU.cpp:697-698`), in
`coordinator/cr_client.cpp:127-128`, and per-worker in
`coordinator/multi_cr_client.cpp:153` (followed by its `setup()`).

**Why it matters:** this is the rendezvous point. If the two sides compute different
paths (e.g. `EXPORT_FILE_PATH` set in one environment but not the other), they map
*different* files: the library never sees the message and the CLI polls forever.

### `send_msg(uint32_t)`, `recv_msg()`, `is_finished()` (share_mem.cpp:51-61)

**Purpose:** Grouped because they are one-liners. `send_msg` stores the value into
`control->signal`; `recv_msg` loads it; `is_finished` is `recv_msg() == FINISH_MSG`
(and `FINISH_MSG` is 0, which is also what a freshly created, zero-filled control
file contains — see Gotchas). The protocol for one operation is: CLI writes the
command word, sends the signal, then spins on `is_finished()` with `usleep(1000)`
(`coordinator/cr_client.cpp:135/152/167/175/253/264`,
`multi_cr_client.cpp:303`); the handler reads the word (`src/vGPU.cpp:757`,
`:837`), does the work, and overwrites the same word with `FINISH_MSG`
(`src/vGPU.cpp:746/821/1071`), which the CLI's next poll observes.

**Called by:** listed above; there are no other users.

**Why it matters:** the whole system's request/response semantics hang on this single
word. It is also the least robust part of the design (no atomics, no memory barriers,
request and response share one slot) — see Gotchas.

---

## `src/backend/backend.h` and `src/backend/mmap_backend.cpp` — the dump buffer

The "backend" owns the big staging area that a checkpoint writes and a restore reads:
one `SHM_SIZE` (default 25 GB) region laid out as `shared_mem_fs` header + payload,
plus a separate 2 GB host buffer that becomes the pinned `staging_buf` pair.
`backend.h` declares an abstract `Backend` (stub base methods, mirroring the `Comm`
pattern) and the concrete `ShareMem`; `mmap_backend.cpp` implements them. Because the
region is a `MAP_SHARED` file mapping, the *external* world — `multi_cr_client`, the
snapshot agent — can open the same file and read the dump without any help from the
(possibly frozen) GPU process. That property is the entire point of this design.

### `ShareMem::ShareMem(int id)` and `Backend` base stubs (mmap_backend.cpp:13-26)

**Purpose:** Trivial: the base-class constructor/`setup()` are empty, and
`ShareMem`'s constructor stores the CR `id` that selects this process's files.
Grouped as trivia.

**Called by:** `init_CR()` (`src/vGPU.cpp:699`, `backend = new ShareMem(id)`).

### `ShareMem::setup()` (mmap_backend.cpp:28)

**Purpose:** Creates and maps the two shared regions. Step by step:

1. Chooses a backend mode by checking `EXPORT_FILE_PATH`. If set, it uses a plain
   file `"$EXPORT_FILE_PATH/ckpt-<id>.data"` (any filesystem — used in deployments
   where the path itself points at hugetlbfs or tmpfs); if unset, it uses
   `/mnt/huge-ckpt/<id>`, which is expected to be a hugetlbfs mount (the error path
   even prints "Check if hugepages are configured properly").
2. `open(O_CREAT|O_RDWR)` + `ftruncate(fd, SHM_SIZE)` + `mmap(..., MAP_SHARED, ...)`
   → `tmp_buf`, the 25 GB dump region. Note the full `SHM_SIZE` is reserved up
   front regardless of how small actual dumps are (hugetlbfs reservations are
   all-or-nothing here; see `docs/design-unrounded-dumps.md` §5.4).
3. Initializes the `shared_mem_fs` header in place: `file_num = 0`,
   `current_offset = ROUND_UP_2MB(sizeof(shared_mem_fs))`, briefly holding the
   *member* mutex `ShareMem::fs_mutex` (which — careful — is a different object from
   the global `fs_mutex` in `vGPU.cpp`; see Gotchas).
4. Creates the second file (`ckpt-<id>-host.data` or `/mnt/huge-ckpt/<id>-host`),
   sizes it to `STAGING_BUF_SIZE * STAGING_BUF_NUM` (2 GB), maps it as
   `host_buf_ptr`. This is the memory `init_CR()` later pins and splits into
   `staging_buf[0..1]`.

**Called by:** `init_CR()` (`src/vGPU.cpp:700`) only.

**Why it matters:** every byte of every checkpoint lives in memory this function
maps. If the hugetlbfs mount is missing or under-reserved, `mmap` fails here and the
process exits — the most common deployment failure mode. It also fixes the file
*names* that external consumers (snapshot agent, `multi_cr_client.cpp:183`) must
mirror exactly.

### `ShareMem::get_tmp_buf()` and `ShareMem::get_host_buffer()` (mmap_backend.cpp:114-120)

**Purpose:** Plain accessors returning `tmp_buf` and `host_buf_ptr`. Grouped as
trivia.

**Called by:** `get_tmp_buf` from all four pipelines and both IPC rebuild phases
(`src/vGPU.cpp:74/246/415/534/998/1025`); `get_host_buffer` from `init_CR()`
(`src/vGPU.cpp:706`). Always through the `Backend*` global, so keep the virtual
signatures in sync with `backend.h` (note `get_host_buffer` is pure-virtual in the
base but the `ShareMem` declaration forgot the `override` keyword — harmless today,
but it means a future signature drift would not be caught by the compiler flag).

---

## Key global state

All of the following live in `src/vGPU.cpp` unless noted, and are process-globals
inside the *target application*:

* **`allocated_memory`** (`std::map<void*, size_t>`, defined
  `src/GPUs/NVIDIA/nv.cpp:13` / `src/GPUs/AMD/amd.cpp:11`, declared
  `src/common.h:61`). The source of truth for "what is on the GPU". **Written** by
  the intercepted allocation hooks (`nv.cpp:458` insert on `cudaMalloc`,
  `nv.cpp:507` erase on `cudaFree`, AMD equivalents). **Read** by `ckpt()` (iterate
  and release), `ckpt_selective()` / `find_containing_allocation()` (lookup), and
  the vendor `releasePhysicalMemory`/`remapPhysicalMemory` implementations. Stores
  the *unrounded* caller-requested size — a fact the unrounded-dumps change depends
  on.
* **`allocated_memory_type`** (`ptr -> 0|1`, `common.h:64`): allocation flavor
  (runtime malloc vs VMM), maintained by the same hooks, consumed by the vendor
  release/remap paths.
* **`gpu_mem_mutex`** (`vGPU.cpp:38`, extern in `common.h:66`): guards
  `allocated_memory` against concurrent mutation. Taken by all four pipelines
  (`vGPU.cpp:83/253/410/529`) and by the allocation hooks
  (`nv.cpp:327/468`). This is what stops the application from `cudaMalloc`ing in
  another thread halfway through a checkpoint iteration.
* **`fs_mutex`** (global, `vGPU.cpp:37`): guards the `shared_mem_fs` header during
  the two checkpoint pipelines. Manually locked/unlocked (not RAII) — every early
  `exit(-1)` path dutifully unlocks first. Note that `ShareMem` *also* has a member
  named `fs_mutex` (`backend.h:21`, used only in `mmap_backend.cpp:80-83`); the two
  are unrelated objects that happen to protect the same bytes at different times.
* **`comm`, `backend`, `gpu`** (`vGPU.cpp:39-41`): the singletons created by
  `init_CR()`; every pipeline and both handlers dereference them. `gpu` is the
  vendor abstraction from `src/GPUs/GPU.h` (streams, events, async memcpy,
  release/remap, context push/pop).
* **`staging_buf[STAGING_BUF_NUM]`** (`vGPU.cpp:43`): two 1 GB slices of the
  backend's pinned host buffer, filled in `init_CR()` and ping-ponged by all four
  pipelines. They are the middle hop of every byte moved: GPU ↔ `staging_buf` ↔ dump
  region.
* **`CR_initialized`** (`vGPU.cpp:45`): idempotence flag for `init_CR()`, read by
  both signal handlers. Not atomic — fine only because all CR signals are handled on
  one thread at a time in practice.
* **The `shared_mem_fs` dump layout** (in the backend's `tmp_buf`): offset 0 holds
  the header (`file_num`, `current_offset`, `files[4096]`); payload starts at
  `ROUND_UP_2MB(sizeof(shared_mem_fs))` and entries are packed back-to-back in
  header order. Since the unrounded-dumps change, selective entries' `size` and
  `start_offset` are **not** 2 MB-aligned; consumers must treat them as plain byte
  counts. The multi-GPU rebuild path additionally steals the *tail* of the rounded
  header area for two `IpcRebuildShmBlock` slots: this process's block at
  `payload_start - sizeof(IpcRebuildShmBlock)` (`vGPU.cpp:1000`) and the peer's at
  `payload_start - 2*sizeof(...)` (`vGPU.cpp:1026`) — written by `IPC_EXPORT_MSG`
  handling, read by `IPC_IMPORT_MSG` handling and by `multi_cr_client`, which maps
  the same files.
* **`signal_controls` / `selective_cr_request`** (in the per-PID control file mapped
  by `ShareMemComm::setup()`): `control->signal` is written by both sides
  (`send_msg`) and polled by both sides (`recv_msg`/`is_finished`);
  `control->selective_req` is written by `cr_client`
  (`coordinator/cr_client.cpp:148/163`) and read in place by the handler
  (`src/vGPU.cpp:761`).
* **`g_ipc_export_data_buf` / `g_local_alloc_data_buf`** (`vGPU.cpp:32-35`):
  anonymous `mmap`'d host buffers that carry exported/local `cuMem` allocation
  *contents* across the teardown→rebuild gap, written in `IPC_TEARDOWN_MSG`
  handling, consumed and unmapped in `IPC_EXPORT_MSG` handling. If the process dies
  between the phases, this data is gone — unlike the main dump, it never touches a
  file.

---

## Gotchas for maintainers

* **Everything runs in a signal handler, and almost none of it is
  async-signal-safe.** POSIX permits only a short list of functions inside signal
  handlers (roughly: simple syscalls). This code calls `fprintf`, `new`, `std::map`
  iteration, `std::thread`, mutex locks and the entire CUDA driver from handler
  context — all formally undefined behavior. It works in practice because the
  deployment contract makes the moment of delivery quiet (`CUDA_LAUNCH_BLOCKING=1`,
  eager mode, coordinator-orchestrated timing) and because the handler thread never
  interrupts *itself* holding one of these locks. But if a CR signal ever lands
  while the interrupted thread holds `gpu_mem_mutex` (e.g. mid-`cudaMalloc` hook,
  `nv.cpp:327`) or malloc's internal lock, the process deadlocks. Do not add work to
  the handlers casually, and never send two CR signals concurrently to one process.
* **The control-word protocol has no synchronization.** `control->signal` is a plain
  (non-atomic, non-volatile) `uint32_t` in shared memory, used for both request and
  response with no memory barriers. `FINISH_MSG == 0` is also the value of a freshly
  zero-filled control file, so `is_finished()` is trivially true *before* an
  operation starts — the CLIs rely on writing the command word strictly before
  `kill()` and only polling *after*. A reordering compiler/CPU or a poll inserted at
  the wrong time yields instant false completion. Treat the sequence
  write-msg → kill → poll as sacred.
* **Locking rules.** Take `fs_mutex` before `gpu_mem_mutex` (both checkpoint
  pipelines do; restores take only `gpu_mem_mutex`). The global `fs_mutex` is
  hand-unlocked on ~10 early-exit paths — if you add an error return to `ckpt()` or
  `ckpt_selective()`, you must unlock manually or convert the function to
  `lock_guard`/`unique_lock`. Also remember `ShareMem::fs_mutex` (member) and the
  global `fs_mutex` are *different mutexes with the same name*; the member one only
  matters during `setup()`.
* **Size rounding is intentionally asymmetric now.** Full `ckpt()` still dumps
  `ROUND_UP_2MB(size)` per allocation (`vGPU.cpp:110`); `ckpt_selective()` dumps the
  raw `alloc_size` (`vGPU.cpp:292`, commit `7a131b4`, rationale and validation in
  `docs/design-unrounded-dumps.md`). This is safe because `releasePhysicalMemory`
  and `remapPhysicalMemory` re-round internally and the staging pipeline is
  alignment-agnostic — but it means (a) dump size ≠ VRAM freed (release is still
  2 MB-block-granular), (b) `shared_mem_fs` consumers must never assume aligned
  sizes/offsets, and (c) the padding tail of each restored block comes back
  *uninitialized*, which is fine only because `cudaMalloc` never handed those bytes
  out.
* **`SHM_SIZE` is a hard wall, checked late.** The overflow check fires per-file
  *during* the dump and calls `exit(-1)` — killing the inference server mid-
  checkpoint with the dump half-written. If working sets grow, bump `-DSHM_SIZE_GB`
  (and the hugepage reservation: each process needs `SHM_SIZE + 2 GB`). Same for
  `MAX_FILE_NUM` (4096 allocations) and `MAX_SELECTIVE_REGIONS` (4096 regions):
  disabling the PyTorch caching allocator makes one allocation per tensor, so large
  models can approach these limits.
* **Restore trusts the dump header blindly.** `restore_ptr_and_content*()` remaps
  and copies whatever `shared_mem_fs` says, with no cross-check against
  `allocated_memory` or the selective request. A stale or corrupted dump file (wrong
  CR ID mapping, a second process reusing the file) writes garbage into GPU memory
  without complaint. The `get_id()` counter and the `pid_map_<pid>` files are the
  only defense — be careful when cleaning `/mnt/huge-ckpt` between runs (the atomic
  ID counter lives in `<ctl_dir>/control` and never resets while the file exists).
* **Environment must match on both sides.** `EXPORT_FILE_PATH` changes both the
  control-file directory (`ShareMemComm::setup`, `get_id`, `init_CR`) and the dump
  backend mode (`ShareMem::setup`). If the CLI and the target process disagree on
  it, they rendezvous on different files and the CLI polls forever.
* **Comment/code drift to verify:** the `RESTORE_MSG` branch comment says
  "cuda-checkpoint restore was already called by cr_client before this signal"
  (`vGPU.cpp:803`), but on the NVIDIA path `cr_client` actually sends the restore
  signal, waits for `FINISH_MSG`, and only *then* runs `cuda-checkpoint --toggle`
  (`coordinator/cr_client.cpp:262-277`). One of the two is stale; confirm the real
  ordering before relying on either.
* **Duplicate pipelines.** `ckpt()`/`ckpt_selective()` and the two restore functions
  are ~90 % copy-paste of each other. Any fix to the double-buffering logic must be
  applied in all four places. (Minor cosmetic bug while you're there: several error
  strings contain a literal `\\n` — e.g. `"Failed to create stream\\n"` — printing a
  backslash instead of a newline.)
* **`Comm`/`Backend` base classes are silent no-ops.** Their virtual methods are
  stubs, not pure-virtual (except the two `Backend` buffer getters), so a missed
  override or an accidentally-sliced base instance compiles fine and simply does
  nothing at runtime — the failure shows up as an infinite poll loop in the CLI.
