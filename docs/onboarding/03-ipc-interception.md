# 03 — cuMem IPC Interception and Teardown/Rebuild

## What this subsystem does

When a multi-GPU application (for example a tensor-parallel vLLM deployment) runs, each
worker process owns one GPU, and the workers share GPU memory with each other through
NCCL. NCCL allocates GPU buffers with the CUDA *Virtual Memory Management* (VMM) API,
exports each buffer as a *shareable handle* (a POSIX file descriptor), passes that fd to
the peer worker processes, and each peer *imports* the handle and maps the same physical
GPU memory into its own address space. This cross-process wiring is exactly what NVIDIA's
`cuda-checkpoint` utility cannot freeze: memory that is exported to, or imported from,
another process is "entangled" driver state that cannot be snapshotted process-by-process.

This subsystem solves that. It lives inside `vGPU-NVIDIA.so`, the LD_PRELOAD library
injected into every GPU worker, and does two jobs:

1. **Record keeping (all the time):** it intercepts every relevant CUDA driver call the
   application/NCCL makes (`cuMemCreate`, `cuMemExportToShareableHandle`,
   `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`,
   `cuMemRelease`, plus `cudaDeviceEnablePeerAccess` and the `cudaIpc*Event*` APIs) and
   records enough metadata (sizes, virtual addresses, allocation properties, access
   descriptors, fds) to be able to destroy and later exactly recreate everything.

2. **Teardown / rebuild (at checkpoint/restore time):** when the external coordinator
   (`multi_cr_client`) asks, it saves the contents of all shared buffers to host RAM,
   destroys every IPC mapping, allocation, exported fd, and peer-access link (so
   `cuda-checkpoint` sees a "clean" single-process CUDA state), and after restore it
   recreates every allocation *at the same virtual address*, refills the data, re-exports
   new fds, ships those fds to peer processes over Unix domain sockets, and re-imports
   and re-maps them — so all the raw pointers NCCL cached internally are valid again
   without modifying a single line of NCCL.

The main files:

| File | Role |
|---|---|
| `src/ipc_hooks.h` | Public API of the tracking/teardown/rebuild engine + shared-memory exchange structs |
| `src/ipc_hooks.cpp` | The engine itself (~2900 lines): interception machinery, tracking tables, teardown/rebuild logic |
| `src/ipc_fd_exchange.h` / `.cpp` | Cross-process fd transfer over Unix domain sockets (SCM_RIGHTS) |
| `src/vGPU.cpp` (driver, not documented in depth here) | Signal handlers that call into this subsystem |
| `coordinator/multi_cr_client.cpp` (driver) | External CLI that orchestrates the phases across all workers |

### Multi-GPU checkpoint/restore phase sequence

The coordinator drives everything via signals + a small shared-memory mailbox per worker
(one message word, defined in `src/comm/comm.h:23-39`). The signals are defined in
`src/common.h:37-45`: `CR_IPC_TEARDOWN_SIGNAL` (= `SIGRTMAX-1`), `CR_IPC_REBUILD_SIGNAL`
(= `SIGRTMAX-2`), `CR_IPC_VALIDATE_SIGNAL` (= `SIGRTMAX-3`), plus the single-GPU signals
`CR_CKPT_SIGNAL`/`CR_RESTORE_SIGNAL`. The rebuild signal is sent twice with two different
mailbox messages (`IPC_EXPORT_MSG`, then `IPC_IMPORT_MSG`) to split restore Phase 3 in two.

```
            multi_cr_client                worker A (vGPU-NVIDIA.so)        worker B (vGPU-NVIDIA.so)
            (coordinator)                  pid=A, GPU 0                     pid=B, GPU 1
============ CHECKPOINT (-c) ==================================================================
Phase 1     IPC_TEARDOWN_MSG + ---------->  cr_ipc_signal_handler:           (same, in parallel)
            CR_IPC_TEARDOWN_SIGNAL          - sync GPU
            to ALL workers first,           - ipc_teardown_all_imports()     unmap+release peer memory
            then wait for all               - ipc_save_and_teardown_         copy own exported buffers
                                              all_exports()                  to host RAM, close fds,
                                            - ipc_teardown_all_events()      unmap+release (keep VA!)
                                            - ipc_save_and_teardown_
                                              local_allocs()
                                            - ipc_disable_all_peer_access()
Phase 2     CKPT_MSG + CR_CKPT_SIGNAL --->  ckpt(): dump all tracked GPU allocations to shm
Phase 3     cuda-checkpoint lock+checkpoint (or --toggle) for every pid  ->  GPU state frozen

============ RESTORE (-r) =====================================================================
Phase 1     cuda-checkpoint restore+unlock (or --toggle) for every pid  ->  process runs again
Phase 2     RESTORE_MSG + CR_RESTORE_SIGNAL -> restore_ptr_and_content(): refill GPU memory
Phase 3a    IPC_EXPORT_MSG + -------------> cr_ipc_signal_handler:
            CR_IPC_REBUILD_SIGNAL           - ipc_rebuild_and_restore_all_exports()
                                              (cuMemCreate at same VA + refill + re-export -> NEW fds)
                                            - ipc_rebuild_local_allocs()
                                            - ipc_write_export_info_to_shm(my_block)
                                            - uds_fd_server_start()   <- fd server now listening
Phase 3ex   exchange_ipc_export_info():
            coordinator mmaps every worker's shm, copies worker A's export
            list into worker B's "peer_block" and vice versa
            (only METADATA moves here: pid + fd NUMBER + size; not the fd itself)
Phase 3b    IPC_IMPORT_MSG + -------------> cr_ipc_signal_handler:
            CR_IPC_REBUILD_SIGNAL           - ipc_import_from_shm_block(peer_block):
                                                for each peer pid: uds_receive_fds()
                                                   B --connect /tmp/cr_ipc_fd_A.sock--> A
                                                   B --"give me fds 87,91"-----------> A
                                                   A --SCM_RIGHTS(real fds)----------> B
                                                cuMemImportFromShareableHandle + cuMemMap
                                                at ORIGINAL VA + replay cuMemSetAccess
                                            - uds_fd_server_stop()
                                            - ipc_validate_all_mappings()
                                            - ipc_reenable_all_peer_access()
```

Key point about the fd flow: an fd *number* (e.g. "87") is only meaningful inside the
process that owns it, so the shared-memory exchange only tells worker B *which* fds to ask
worker A for. The actual fd transfer happens over a Unix domain socket using SCM_RIGHTS,
which is the only mechanism that makes the kernel duplicate the fd (with all its CUDA
metadata) into the receiving process.

---

## Core concepts (read this first)

**Background: file descriptors (fds).** On Linux, when a process opens a file, socket, or
device, the kernel gives it back a small integer called a file descriptor. The integer is
an index into a per-process table; fd 87 in process A and fd 87 in process B are
completely unrelated. An fd can refer to things that aren't files at all — CUDA shareable
memory handles are fds referring to an anonymous kernel object owned by the NVIDIA driver.

**Background: LD_PRELOAD and symbol interposition.** Linux programs call library
functions (like `cuMemCreate` in `libcuda.so`) through the *dynamic linker*, which resolves
each function name to an address at load time. The `LD_PRELOAD` environment variable tells
the dynamic linker to load a chosen shared library *before* all others; if that library
defines a function with the same name as one in a later library, the preloaded version
"wins" and every call in the program lands there instead. This is called symbol
interposition. The preloaded function can then look up the *real* implementation with
`dlsym(RTLD_NEXT, "name")` — "give me the next definition of this symbol after mine" —
and forward the call, sandwiching its own bookkeeping around it. That is exactly how
`vGPU-NVIDIA.so` works.

**Why plain interposition is not enough here — the `cuGetProcAddress` problem.** NCCL and
the CUDA runtime do not always call driver functions through the dynamic linker. Instead
they ask the driver for function pointers at runtime, via `cudaGetDriverEntryPoint`,
`cudaGetDriverEntryPointByVersion`, or `cuGetProcAddress`/`cuGetProcAddress_v2` ("give me
the address of the function named `cuMemCreate`"). A raw pointer obtained this way bypasses
any preloaded symbol. So this codebase hooks *the lookup functions themselves*: our
versions call the real lookup, and if the requested symbol is in our hook table
(`g_hook_table`, `src/ipc_hooks.cpp:720`), they save the real pointer into a
`real_cuMemXxx` global and hand the caller our hook function instead. The code even
interposes `dlsym` itself (`src/ipc_hooks.cpp:787`) so that a library which `dlsym`s
`cuGetProcAddress` out of `libcuda.so` still gets our version. This gives complete
coverage: no matter how the application, PyTorch, or NCCL obtains a `cuMem*` function,
the call lands in our hook, and we always end up holding a pointer to the real function.

**Background: CUDA VMM (Virtual Memory Management).** Classic `cudaMalloc` gives you one
opaque pointer. The VMM (a.k.a. `cuMem*`) API splits allocation into composable steps,
much like `mmap` on the CPU side:

1. `cuMemAddressReserve` — reserve a range of GPU *virtual addresses* (VA) with nothing
   behind it. Think of it as claiming street addresses before any houses are built.
2. `cuMemCreate` — allocate a chunk of *physical* GPU memory, returning an opaque
   *allocation handle* (`CUmemGenericAllocationHandle`), not a pointer.
3. `cuMemMap(va, size, offset, handle)` — connect the physical memory to the reserved VA.
   Only now can the address be used.
4. `cuMemSetAccess(va, size, descriptors)` — grant read/write permission on that VA range
   to specific devices (each descriptor names a device or NUMA node and the access flags).
   Without this the mapping exists but any access faults.

Teardown is the reverse: `cuMemUnmap` (disconnect), `cuMemRelease` (free physical memory),
`cuMemAddressFree` (give up the VA range). Crucially, you can unmap and release *without*
freeing the VA reservation, then later `cuMemCreate` + `cuMemMap` new physical memory at
the *same* VA — pointers held by other code stay numerically valid. This subsystem's whole
rebuild strategy depends on that trick.

**Shareable handles.** `cuMemCreate` can be told (via
`prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`) to make the
allocation *exportable*. `cuMemExportToShareableHandle` then converts the allocation
handle into an ordinary POSIX fd. That fd can be shipped to another process, which calls
`cuMemImportFromShareableHandle` to get its own allocation handle for the *same physical
GPU memory*, then `cuMemMap`s it into its own VA space. This is how NCCL implements fast
intra-node communication between per-GPU worker processes.

**Why IPC-tainted memory blocks checkpointing.** `cuda-checkpoint` snapshots one process's
CUDA state. An allocation that has been exported (someone else may hold a reference), an
imported mapping (the physical memory belongs to another process), or even an unexported
allocation *created with an IPC-capable handle type* carries driver-level cross-process
state that the checkpoint/restore path cannot reproduce; `cuda-checkpoint` either refuses
to freeze or fails on restore. The same is true of P2P peer access (see below). Therefore
*everything* IPC-related must be removed before the freeze and rebuilt after the thaw.

**Background: peer access.** `cudaDeviceEnablePeerAccess(peerDevice, ...)` lets kernels
running on the current GPU directly dereference pointers that live in another GPU's memory
over NVLink/PCIe ("P2P"). It is pure driver state with no user data behind it, so this
subsystem just remembers which devices were enabled, disables them all before checkpoint,
and re-enables them after restore.

**Background: Unix domain sockets and SCM_RIGHTS.** A Unix domain socket (UDS) is a
socket that exists as a path on the filesystem (here `/tmp/cr_ipc_fd_<pid>.sock`) and
connects two processes on the same machine. Besides bytes, a UDS can carry *ancillary
(control) messages*; the `SCM_RIGHTS` message type tells the kernel "duplicate these fds
of mine into the receiving process's fd table". The receiver gets new fd numbers that
refer to the same kernel objects. This is the only correct way to move a CUDA shareable
handle between processes: opening `/proc/<pid>/fd/N` (the kernel's view of another
process's fds) re-opens the underlying *device node* and loses the CUDA metadata, and
`pidfd_getfd` requires the `CAP_SYS_PTRACE` privilege. Both were tried and abandoned —
see `pidfd_copy_fd` below.

**Background: signals and the shared-memory mailbox.** A signal is an asynchronous
notification the kernel delivers to a process (e.g. `kill(pid, SIGUSR1)`); the process
runs a registered handler function. Signals carry no payload, so the coordinator first
writes a message word (e.g. `IPC_TEARDOWN_MSG`) into a shared-memory control block that
both processes have `mmap`ed (**mmap** = mapping a file or anonymous memory region
directly into a process's address space so it can be read/written like an array; if two
processes map the same file, they see each other's writes). The handler reads the message
to learn what to do, does it, and writes `FINISH_MSG` back.

---

## `src/ipc_hooks.h` — public API

The header (fully commented, worth reading) declares:

- The lifecycle functions called at checkpoint (`ipc_teardown_all_imports`,
  `ipc_get_export_data_size`, `ipc_save_and_teardown_all_exports`,
  `ipc_get_local_alloc_data_size`, `ipc_save_and_teardown_local_allocs`,
  `ipc_teardown_all_events`, `ipc_disable_all_peer_access`) and at restore
  (`ipc_rebuild_and_restore_all_exports`, `ipc_rebuild_local_allocs`,
  `ipc_write_export_info_to_shm`, `ipc_import_from_shm_block`,
  `ipc_reenable_all_peer_access`).
- Diagnostics (`ipc_dump_state`, `ipc_dump_nvidia_fds`, `ipc_validate_all_mappings`,
  count getters) and a timing snapshot struct (`IpcTimingSnapshot`) filled in by every
  phase and read back by callers for reporting.
- The shared-memory exchange structures `IpcExportShmEntry` and `IpcRebuildShmBlock`
  (`src/ipc_hooks.h:206-221`): one block per worker, up to `IPC_MAX_EXPORTS_PER_PROC`
  (64) entries, each entry = `{owner_pid, fd number, exporter's VA, size, valid}`. Both
  the workers and `multi_cr_client` include this header so the layout matches.

**Stale declarations to be aware of:** `ipc_rebuild_all_imports(const std::vector<int>&)`
(`src/ipc_hooks.h:127`) is declared but **not defined anywhere** — it was superseded by
`ipc_import_from_shm_block()`. Similarly, several header comments still say the import
path "uses pidfd_getfd"; the real mechanism is UDS + SCM_RIGHTS (see
`src/ipc_fd_exchange.cpp`). Trust the .cpp, not those comments.

---

## `src/ipc_hooks.cpp` — the engine

### Interception machinery

#### `ipc_hooks_init()` (constructor, `src/ipc_hooks.cpp:1503`)

- **Purpose:** runs automatically when the library is loaded (`__attribute__((constructor))`
  makes the dynamic linker call it before `main`). It does no hooking itself — hooking
  works purely by symbol interposition — it only prints which library provides
  `cudaGetDriverEntryPoint` / `cuGetProcAddress` under `RTLD_DEFAULT` vs `RTLD_NEXT`, as a
  sanity check that the preload actually shadows the CUDA libraries.
- **Called by:** the dynamic linker at `vGPU-NVIDIA.so` load time.
- **Why it matters:** if the log shows `DEFAULT` resolving into `libcudart.so` instead of
  `vGPU-NVIDIA.so`, interposition is broken and nothing downstream will be tracked.

#### `dlsym()` interposer (`src/ipc_hooks.cpp:787`)

- **Purpose:** replaces glibc's `dlsym` for the whole process. Step by step:
  1. Finds the *real* `dlsym` via `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")` — it must
     use `dlvsym` (versioned lookup) because calling `dlsym` to find `dlsym` would recurse
     into itself.
  2. If the caller asks for `cuGetProcAddress`, `cuGetProcAddress_v2`,
     `cudaGetDriverEntryPoint`, or `cudaGetDriverEntryPointByVersion`, it returns *our*
     hooked versions, so even lookup-functions obtained via `dlsym` are ours.
  3. If the caller asks for any symbol in `g_hook_table` directly (e.g. someone
     `dlsym`s `cuMemMap` out of a `dlopen`ed `libcuda.so`), it resolves the real address,
     stores it in the corresponding `real_*` slot, and returns the hook.
  4. Otherwise it forwards to the real `dlsym`. A thread-local `g_inside_dlsym_hook`
     flag prevents infinite recursion when the hook itself needs real lookups, and
     `handle == RTLD_NEXT` lookups are passed through untouched (that is how our own
     hooks find the real functions).
- **Called by:** any code in the process that calls `dlsym` — the CUDA runtime, NCCL's
  `cudawrap.cc` (which `dlopen`s `libcuda.so` and `dlsym`s `cuGetProcAddress`), PyTorch, etc.
- **Why it matters:** this is the deepest safety net; without it, NCCL's
  `dlsym(libcuda_handle, "cuGetProcAddress")` would fetch the raw driver function and every
  `cuMem*` call would go untracked, making checkpoint silently incomplete.

#### `cudaGetDriverEntryPoint()` / `cudaGetDriverEntryPointByVersion()` (`src/ipc_hooks.cpp:1377`, `:1406`)

- **Purpose:** interpose the CUDA *runtime*'s driver-entry-point lookup. Both call the
  real function via `dlsym(RTLD_NEXT, ...)`, then run `try_intercept(symbol, funcPtr)` on
  success to swap tracked symbols for hooks. The 4-parameter signature is deliberate:
  CUDA 12.5+'s real implementation reads the 4th argument (`driverStatus`) — the long
  comment at `src/ipc_hooks.cpp:1363-1397` explains that calling it as 3-param leaves a
  garbage register that the real implementation dereferences (segfault), so the status
  pointer is always passed straight through.
- **Called by:** the application/NCCL whenever they use the runtime API to resolve driver
  functions (NCCL's `CUDA_DRIVER_API` loading path), plus our own `dlsym` interposer
  redirects lookups of these names here.
- **Why it matters:** this is one of the three doors through which NCCL fetches `cuMem*`
  pointers; all three must be covered or tracking has holes.

#### `cuGetProcAddress()` / `cuGetProcAddress_v2()` / `cuGetProcAddress_common()` (`src/ipc_hooks.cpp:1444-1496`)

- **Purpose:** same idea for the *driver*'s lookup API. Both symbol names are provided
  (CUDA 11 calls the 4-param `cuGetProcAddress`; CUDA 12 renamed it to the 5-param
  `cuGetProcAddress_v2` via a header macro, which the file `#undef`s so it can define
  both). Each calls the real function and then the shared helper
  `cuGetProcAddress_common`, which logs and calls `try_intercept`.
- **Called by:** NCCL (`cudawrap.cc` resolves every driver symbol through here) and the
  CUDA runtime internally; also returned by our `dlsym` interposer.
- **Why it matters:** on modern stacks this is the door NCCL actually uses.

#### `try_intercept()` and `g_hook_table` (`src/ipc_hooks.cpp:748`, `:720`)

- **Purpose:** the single splice point. Given a symbol name and a pointer-to-function-
  pointer that currently holds the *real* address, it scans `g_hook_table`; on a match it
  saves the real address into the table's `real_fn_storage` slot (e.g. `real_cuMemCreate`)
  and overwrites the caller's pointer with the hook. The table covers the seven `cuMem*`
  IPC functions, a family of kernel-launch wrappers, error/synchronize helpers, and
  context functions.
- **Called by:** `cudaGetDriverEntryPoint` (`src/ipc_hooks.cpp:1401`),
  `cudaGetDriverEntryPointByVersion` (`:1428`), `cuGetProcAddress_common` (`:1452`).
- **Why it matters:** this is *how the real function pointers are obtained* in the
  lookup-interception path: we never guess them, we steal them from the driver's own
  answer at the moment the application asks.

#### `resolve_helper_fns()` (`src/ipc_hooks.cpp:223`)

- **Purpose:** lazily resolves (via `dlsym(RTLD_DEFAULT, ...)`) the driver functions the
  teardown/rebuild code needs but which the application may never have looked up itself:
  `cuMemRelease`, `cuMemAddressReserve/Free`, `cuMemcpyDtoH/HtoD` (+ `_v2` fallbacks),
  context functions (`cuCtxPush/PopCurrent`, `cuDevicePrimaryCtxRetain/Release`), and the
  async-copy set (`cuStreamCreate/Synchronize/Destroy`, `cuMemcpyDtoHAsync/HtoDAsync`,
  `cuMemHostRegister/Unregister`). Each is tried under its `_v2` name first because CUDA
  exports versioned symbols.
- **Called by:** every teardown/rebuild entry point (`ipc_teardown_all_imports`
  `src/ipc_hooks.cpp:1577`, `ipc_save_and_teardown_all_exports` `:1649`,
  `ipc_rebuild_and_restore_all_exports` `:1860`, the local-alloc pair `:2124`/`:2271`,
  `ipc_validate_all_mappings` `:2721`) and `hook_cuMemRelease` (`:531`).
- **Why it matters:** teardown may run in a process where NCCL only ever resolved the
  seven hooked functions; everything else must be found on demand.

#### Directly exported `cuMem*` wrappers (`src/ipc_hooks.cpp:918-1320`)

- **Purpose:** `cuMemCreate`, `cuMemExportToShareableHandle`,
  `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemUnmap`, `cuMemRelease`, and
  `cuMemSetAccess` are *also* exported as ordinary symbols. Each lazily resolves the real
  function with `dlsym(RTLD_NEXT, ...)` and forwards to the corresponding `hook_*`
  function. Grouped here because they are trivial one-liners.
- **Called by:** any code that links against `libcuda.so` normally (symbol interposition
  by the dynamic linker) — e.g. libraries that call `cuMemMap` directly instead of via
  `cuGetProcAddress`.
- **Why it matters:** third coverage path; all three converge on the same `hook_*`
  implementations and tracking tables.

### The tracking hooks (the `hook_cuMem*` family)

These are the functions the application/NCCL actually ends up executing whenever they
call a `cuMem*` API, via any of the three interception paths above. Every one follows the
same pattern: call the real function; if it succeeded, take `g_ipc_hook_mutex` and update
the tracking tables. NCCL calls these during `ncclCommInitRank` (buffer registration /
channel setup): each rank `cuMemCreate`s IPC-capable buffers, `cuMemMap`s +
`cuMemSetAccess`es them, `cuMemExportToShareableHandle`s them, ships fds to peers over its
bootstrap socket, and each peer `cuMemImportFromShareableHandle`s + `cuMemMap`s +
`cuMemSetAccess`es its view.

#### `hook_cuMemCreate()` (`src/ipc_hooks.cpp:355`)

- **Purpose:** after a successful create, if `prop->requestedHandleTypes != 0` (i.e. the
  allocation is IPC-*capable*, whether or not it ever gets exported), record
  `{size, *prop}` in `g_created_allocs[handle]`. The saved `CUmemAllocationProp` is what
  makes exact recreation possible later (it encodes device id, handle types, etc.).
  Non-IPC-capable creations are deliberately ignored — `cuda-checkpoint` handles those fine.
- **Called by:** the application/NCCL via interception (NCCL allocates every
  communication buffer this way when `NCCL_CUMEM_ENABLE` is on, which is the default on
  modern NCCL); also our own exported `cuMemCreate` wrapper (`src/ipc_hooks.cpp:918`).
- **Why it matters:** this is where rebuild metadata is born; an allocation missed here
  cannot be rebuilt and will crash `cuda-checkpoint`.

#### `hook_cuMemExportToShareableHandle()` (`src/ipc_hooks.cpp:379`)

- **Purpose:** creates an `IpcExportRecord`: remembers the handle (twice — see
  `app_local_handle` in the data-structures section), handle type, the exported fd (read
  out of `*shareableHandle` when the type is `POSIX_FILE_DESCRIPTOR`), the size and saved
  prop from `g_created_allocs`, and — if `cuMemMap` already happened — the VA from
  `g_handle_to_va`, registering the VA→index in `g_vaddr_to_export_idx`.
- **Called by:** NCCL via interception when it shares a buffer with peer ranks; our
  exported wrapper at `src/ipc_hooks.cpp:1276`.
- **Why it matters:** exports are the "server side" of IPC; the recorded fd is later
  closed at teardown and replaced at rebuild, and the record drives the save/restore of
  the buffer's contents.

#### `hook_cuMemImportFromShareableHandle()` (`src/ipc_hooks.cpp:443`)

- **Purpose:** just inserts the returned handle into `g_imported_handles`. No
  `IpcImportRecord` yet — at import time we don't know where it will be mapped; that
  happens in `hook_cuMemMap`.
- **Called by:** NCCL via interception on the importing rank; wrapper at
  `src/ipc_hooks.cpp:1286`.
- **Why it matters:** the set is the discriminator that lets `hook_cuMemMap` tell "mapping
  of somebody else's memory" apart from "mapping of my own allocation".

#### `hook_cuMemMap()` (`src/ipc_hooks.cpp:461`)

- **Purpose:** two independent branches after the real map succeeds:
  1. If the handle is in `g_imported_handles`: create an `IpcImportRecord`
     `{vaddr, size, handle}` and index it in `g_vaddr_to_import_idx`. This is the moment
     an import becomes trackable.
  2. If the handle is in `g_created_allocs` (our own IPC-capable memory): record
     `g_handle_to_va[handle] = ptr`, and back-fill `mapped_vaddr/has_mapping` on any
     existing export record for this handle (handles the export-before-map ordering).
- **Called by:** NCCL via interception (both sides map); wrapper at `src/ipc_hooks.cpp:1295`.
- **Why it matters:** the VA recorded here is the address NCCL bakes into its internal
  structures; teardown/rebuild must reuse exactly this VA.

#### `hook_cuMemSetAccess()` (`src/ipc_hooks.cpp:590`)

- **Purpose:** records access descriptors at three levels:
  1. `g_va_access_descs[ptr]` — for **every** cuMem VA, dedup-updated per
     `(location.type, location.id)`. This exists because NCCL calls `cuMemSetAccess`
     *before* `cuMemExportToShareableHandle`, so export records don't exist yet when
     access is set; the VA-keyed map survives that ordering.
  2. If the VA is a known import: store the first descriptor in
     `access_desc`/`access_dev` (legacy) and accumulate *all* descriptors into
     `all_access` (deduped per type+id) for replay at rebuild.
  3. If the VA is a known export: set `access_desc`/`access_dev`/`has_access`.
- **Called by:** NCCL via interception after each map; wrapper at `src/ipc_hooks.cpp:1315`.
- **Why it matters:** access descriptors are the easiest thing to lose and the failure is
  deferred: rebuild succeeds, then the first NCCL collective dies with
  `cudaErrorIllegalAddress`. Note the type+id dedup subtlety: with
  `NCCL_CUMEM_HOST_ENABLE=1`, NCCL sets both a `DEVICE` and a `HOST_NUMA` descriptor with
  the *same numeric id*; deduping by id alone would drop one (comment at
  `src/ipc_hooks.cpp:1922-1928`).

#### `hook_cuMemUnmap()` (`src/ipc_hooks.cpp:509`)

- **Purpose:** if the application itself unmaps a tracked import during normal operation
  (e.g. NCCL communicator destruction), mark the record `torn_down` so checkpoint-time
  teardown doesn't double-unmap it.
- **Called by:** application/NCCL via interception; wrapper at `src/ipc_hooks.cpp:1303`.
- **Why it matters:** without it, teardown would call `cuMemUnmap` on an already-unmapped
  VA (an error, though survivable) and `cuMemRelease` on a freed handle.

#### `hook_cuMemRelease()` (`src/ipc_hooks.cpp:530`)

- **Purpose:** the most intricate hook, because of *handle aliasing*. After a rebuild,
  the application still holds the **old** (pre-checkpoint) handle value; if it later
  releases it (NVSHMEM does this in finalize), the raw value is now meaningless. So:
  1. Look the incoming handle up in `g_rebuilt_handle_alias`; if found, substitute the
     rebuilt handle and release *that* instead.
  2. On success, run `cleanup_handle()` for both the app handle and the actual handle:
     erase from `g_handle_to_va`, `g_va_access_descs`, `g_vaddr_to_export_idx`,
     `g_created_allocs`, `g_imported_handles`; zero the handle in matching
     import/export records; mark matching local-alloc records torn down. Finally purge
     all alias entries touching either handle.
- **Called by:** application/NCCL/NVSHMEM via interception during normal cleanup; wrapper
  at `src/ipc_hooks.cpp:1309`.
- **Why it matters:** keeps the tables truthful so checkpoint-time teardown never
  double-frees, and makes post-restore application shutdown work even though every CUDA
  handle value changed under the application's feet.

### Peer-access hooks and helpers

#### `cudaDeviceEnablePeerAccess()` / `cudaDeviceDisablePeerAccess()` (`src/ipc_hooks.cpp:260`, `:288`)

- **Purpose:** classic LD_PRELOAD interposition of the *runtime* API (resolved with
  `dlsym(RTLD_NEXT, ...)` on first call). On successful enable — or on
  `cudaErrorPeerAccessAlreadyEnabled`, which is swallowed and converted to success after
  clearing the sticky error with `cudaGetLastError()` — the peer device id goes into
  `g_peer_access_devices`; disable removes it.
- **Called by:** the application (PyTorch/vLLM enable P2P between GPUs; NCCL uses the
  driver-level equivalent internally, which is handled by the cuMemSetAccess tracking
  instead).
- **Why it matters:** P2P links are per-context driver state `cuda-checkpoint` may reject;
  they must be enumerable to be disabled.

#### `ipc_disable_all_peer_access()` / `ipc_reenable_all_peer_access()` (`src/ipc_hooks.cpp:305`, `:330`)

- **Purpose:** disable saves the current set into `g_saved_peer_access_devices`, calls the
  real disable for each device (tolerating `NotEnabled`), and clears the live set;
  re-enable replays the saved set (tolerating `AlreadyEnabled`) and repopulates the live set.
- **Called by:** `src/vGPU.cpp:798` (after single-GPU data dump, pre-freeze),
  `src/vGPU.cpp:945` (end of IPC teardown phase), `src/vGPU.cpp:816` (after single-GPU
  restore), `src/vGPU.cpp:1054` (end of import phase);
  also the NCCL adapter runtime `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:319`/`:427`.
- **Why it matters:** last teardown step / last rebuild step; ordering is deliberate (see
  Gotchas).

### Old-style IPC and event hooks

#### `cudaIpcGetMemHandle()` / `cudaIpcOpenMemHandle()` / `cudaIpcCloseMemHandle()` (`src/ipc_hooks.cpp:890-916`)

- **Purpose:** detection tripwires only. The *legacy* CUDA IPC memory API (opaque 64-byte
  handles instead of fds) is **not** supported by this teardown/rebuild machinery; these
  interposers loudly warn (first 5 occurrences) and forward to the real functions.
- **Called by:** would be the application (older PyTorch multiprocessing paths); in a
  healthy vLLM+NCCL setup they should never fire.
- **Why it matters:** if you see these warnings in a worker log, the checkpoint is *not*
  safe — legacy IPC state is invisible to teardown.

#### `cudaIpcGetEventHandle()` / `cudaIpcOpenEventHandle()` (`src/ipc_hooks.cpp:1334`, `:1348`), `ipc_teardown_all_events()` (`:1535`), `ipc_get_event_count()` (`:1562`)

- **Purpose:** CUDA events (GPU synchronization markers) can also be shared across
  processes. The two interposers record every exported (`is_opened=false`) and imported
  (`is_opened=true`) event in `g_ipc_events`. `ipc_teardown_all_events` destroys
  *imported* events with `cudaEventDestroy` (the importer's view is what blocks
  checkpoint) and marks exported ones torn down without destroying them (the owner still
  needs them). There is **no rebuild** for events — after restore the application would
  need to re-share them itself; in the NCCL flow they are transient enough not to matter.
- **Called by:** teardown is called from `src/vGPU.cpp:902` (teardown phase) and
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:305`; the interposers by the
  application via the dynamic linker.
- **Why it matters:** imported IPC events are another checkpoint blocker; small but fatal.

### Checkpoint-side: save and teardown

#### `ipc_teardown_all_imports()` (`src/ipc_hooks.cpp:1575`)

- **Purpose:** for every live `IpcImportRecord`: `cuMemUnmap(vaddr, size)` +
  `cuMemRelease(handle)` (skipping the release if the application already released the
  handle). **Deliberately does NOT call `cuMemAddressFree`** — the VA reservation is kept
  so the rebuild can map new imports at the identical address. Marks records
  `torn_down`, zeroes handles, fills the `import_teardown_*` timing fields.
- **Called by:** `src/vGPU.cpp:862` (on `IPC_TEARDOWN_MSG`) and
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:293`.
- **Why it matters:** removes the "I hold someone else's memory" state. Runs *before*
  export teardown by convention (imports reference peers' exports).

#### `ipc_get_export_data_size()` (`src/ipc_hooks.cpp:1636`)

- **Purpose:** sums `mapped_size` over all live, mapped exports so the caller can size
  the host buffer.
- **Called by:** `src/vGPU.cpp:868`, `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:296`.
- **Why it matters:** the caller `mmap`s an anonymous buffer of exactly this size
  (`src/vGPU.cpp:877`) that lives across the whole freeze/thaw.

#### `ipc_save_and_teardown_all_exports()` (`src/ipc_hooks.cpp:1647`)

- **Purpose:** the export-side checkpoint, in three steps:
  1. **Save (DtoH).** Groups live exports by device id (from `saved_prop`), validates the
     buffer size, then per device: retains+pushes the device's *primary context*
     (**Background:** a CUDA context is the per-process, per-device container all CUDA
     operations run inside; copies must run in a context bound to the right device, so
     the code pushes one onto the thread's context stack and pops it after) and copies
     each export's GPU contents into the host buffer. If the host buffer can be
     *pinned* (`cuMemHostRegister` — page-locking host memory so the GPU can DMA into it
     directly), it uses a CUDA *stream* (an async work queue) to batch all copies and
     synchronize once (`cuMemcpyDtoHAsync`), else falls back to synchronous
     `cuMemcpyDtoH`. Crucially, each record's `saved_data_offset` is written here —
     because copies are grouped by device, buffer order is *not* vector order.
  2. **Close fds.** `close(rec.exported_fd)` for every export — leftover CUDA fds are
     themselves checkpoint blockers.
  3. **Teardown.** `cuMemUnmap` + `cuMemRelease` per export (again *keeping* the VA
     reservation, despite what the header comment at `src/ipc_hooks.h:84-96` says about
     `cuMemAddressFree`), mark `torn_down`, zero `local_handle`, fill timing fields.
  Returns the number torn down, or -1 if the buffer was too small / copies failed.
- **Called by:** `src/vGPU.cpp:890`, `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:301`.
- **Why it matters:** the exporter *owns* the physical memory, so its bytes must be
  preserved (peers' copies are just views of the same memory); the pattern
  save-then-destroy-then-recreate is the heart of the whole design.

#### `ipc_get_local_alloc_data_size()` (`src/ipc_hooks.cpp:2103`) and `ipc_save_and_teardown_local_allocs()` (`:2122`)

- **Purpose:** handles the third category: allocations that were `cuMemCreate`d with an
  IPC-capable handle type but **never exported** (NCCL creates such buffers internally).
  The driver still tags them with IPC state, so `cuda-checkpoint` cannot restore them.
  The size function walks `g_created_allocs`, keeps only handles that have a VA in
  `g_handle_to_va` and whose VA is *not* in the export set, and sums sizes. The
  save/teardown function repeats that walk to build `g_local_allocs` (this table is
  **populated fresh at teardown time**, unlike imports/exports which grow during normal
  operation), does the same device-grouped async/sync DtoH save, then
  `cuMemUnmap` + `cuMemRelease` per record (VA again preserved).
- **Called by:** `src/vGPU.cpp:908`/`:929`,
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:308`/`:315`.
- **Why it matters:** the least obvious blocker — nothing was ever shared, yet the freeze
  still fails without this step. Exclusion is by *VA*, not handle, because export
  teardown zeroes handles (comment at `src/ipc_hooks.cpp:2107`).

### Restore-side: rebuild

#### `ipc_rebuild_and_restore_all_exports()` (`src/ipc_hooks.cpp:1858`)

- **Purpose:** inverse of export save/teardown, in three phases:
  - **Phase A (alloc+map+access):** for each torn-down export with a saved prop:
    `cuMemCreate(new_handle, mapped_size, &saved_prop)` →
    `cuMemMap(mapped_vaddr, ...)` at the **original** VA (which still works because the
    reservation was never freed) → replay every access descriptor recorded in
    `g_va_access_descs[vaddr]` one at a time (fallbacks: the single stored
    `access_desc`, else a synthesized READWRITE descriptor for the owning device).
  - **Phase B (HtoD):** device-grouped, pinned/async-batched copy of the saved bytes back
    to the GPU, using each record's `saved_data_offset` — *not* running offsets — so
    mixed-device export sets restore from the right place.
  - **Phase C (re-export):** `cuMemExportToShareableHandle` on each new handle → **new
    fd** stored in `exported_fd`; if the application's original handle differs from the
    new one, register `g_rebuilt_handle_alias[app_local_handle] = new_handle` (so a later
    application `cuMemRelease` of the stale value works, see `hook_cuMemRelease`);
    un-mark `torn_down`; re-register the new handle in `g_created_allocs` and
    `g_handle_to_va`.
- **Called by:** `src/vGPU.cpp:969` (on `IPC_EXPORT_MSG`),
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:356`.
- **Why it matters:** after this, the exporter looks to itself exactly as before the
  checkpoint — same VAs, same data, valid fds — but the fds have **new numbers**, which is
  why the export-info exchange and re-import phases must follow.

#### `ipc_rebuild_local_allocs()` (`src/ipc_hooks.cpp:2269`)

- **Purpose:** same Phase A + Phase B as above (create at original VA, replay
  `g_va_access_descs` or synthesize a default descriptor, device-grouped HtoD), minus the
  re-export. Updates `g_created_allocs`/`g_handle_to_va` to the new handles, erases the
  old ones, and stores the new handle back into each `g_local_allocs` record.
- **Called by:** `src/vGPU.cpp:985`, `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:363`.
- **Why it matters:** NCCL's internal pointers into these buffers stay valid; skipping
  this leaves those VAs unmapped and the first kernel touching them faults.

#### `ipc_write_export_info_to_shm()` (`src/ipc_hooks.cpp:2428`)

- **Purpose:** publishes this process's post-rebuild export list: for every export with a
  valid fd, writes `{owner_pid = getpid(), fd number, VA, size, valid=1}` into the
  caller-supplied `IpcRebuildShmBlock` (capped at `IPC_MAX_EXPORTS_PER_PROC`).
- **Called by:** `src/vGPU.cpp:1002` — the worker places its block at a fixed offset near
  the end of the 2 MB-aligned header area of its checkpoint shared-memory file
  (`my_block` at `header_end - sizeof(block)`, `peer_block` at
  `header_end - 2*sizeof(block)`; `multi_cr_client.cpp:195-204` computes the same
  addresses) — and `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:373`.
- **Why it matters:** this metadata (not the fds themselves!) is what the coordinator
  cross-copies between workers in `exchange_ipc_export_info()`
  (`coordinator/multi_cr_client.cpp:218`).

#### `ipc_import_from_shm_block()` (`src/ipc_hooks.cpp:2452`)

- **Purpose:** the import-side rebuild, the most subtle function in the file:
  1. **Group by owner.** Bucket the peer-block entries by `owner_pid` so all fds from one
     peer are fetched in a single UDS connection.
  2. **Fetch fds.** For each peer, `uds_receive_fds(pid, fd_numbers, ...)` — connect to
     the peer's fd server and receive real local fds via SCM_RIGHTS.
  3. **Positional matching.** Collect this process's torn-down imports *in original
     creation order* and the valid peer exports *in block order*, and pair them by
     position: NCCL creates exports and imports in deterministic channel order, so the
     Nth peer export corresponds to the Nth torn-down import. (Size-based matching was
     abandoned because NCCL buffers routinely share the same size — comment at
     `src/ipc_hooks.cpp:2525-2532`.) A count mismatch logs a warning and matches
     `min(exports, imports)` pairs.
  4. **Rebuild each pair.** `cuMemImportFromShareableHandle(fd)` (then immediately
     `close(fd)` — CUDA has its own reference after import), `cuMemMap` at the import's
     **original** VA (reservation was kept), replay *all* recorded access descriptors from
     `all_access` (fallback: the single legacy descriptor), store the new handle, clear
     `torn_down`. Unmatched surplus fds are closed.
- **Called by:** `src/vGPU.cpp:1031` (on `IPC_IMPORT_MSG`, reading the `peer_block` that
  the coordinator filled), `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:407`.
- **Why it matters:** the final stitch: after this, both sides of every NCCL channel see
  the same physical memory at the same virtual addresses as before the checkpoint. The
  positional-matching assumption is also the subsystem's most fragile invariant — see
  Gotchas.

### Diagnostics and small helpers

#### `ipc_dump_state()` (`src/ipc_hooks.cpp:2679`), `ipc_get_import_count()` / `ipc_get_export_count()` (`:2669`/`:2674`), `ipc_reset_timing_snapshot()` / `ipc_get_timing_snapshot()` (`:126`/`:131`)

- **Purpose:** print the imports/exports tables; return table sizes (note: total records,
  including torn-down ones); reset/copy the phase-timing accumulator.
- **Called by:** `src/vGPU.cpp:857`/`:842` and
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:280-292`, `:325`, `:385`, `:431`.

#### `ipc_dump_nvidia_fds()` (`src/ipc_hooks.cpp:2700`)

- **Purpose:** scans fds 0-1023 via `readlink("/proc/self/fd/N")` (**Background:** `/proc`
  is a virtual filesystem where the kernel exposes per-process state as fake files;
  `/proc/self/fd/` contains one symbolic link per open fd, pointing at whatever the fd
  refers to) and prints any whose target mentions nvidia/cuda/anon_inode.
- **Called by:** `src/vGPU.cpp:858`/`:940` (before/after teardown),
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:290`/`:321`.
- **Why it matters:** the "did we really close everything?" check — a leftover
  anon_inode fd after teardown predicts a `cuda-checkpoint` failure.

#### `ipc_validate_all_mappings()` (`src/ipc_hooks.cpp:2719`)

- **Purpose:** probes every live export, import, and local alloc by reading 4 bytes from
  its VA with `cuMemcpyDtoH` (pushing the owning device's primary context for exports and
  locals), printing OK/FAIL per mapping and returning the error count.
- **Called by:** `src/vGPU.cpp:1047` (after import rebuild), the on-demand
  `CR_IPC_VALIDATE_SIGNAL` handler `src/vGPU.cpp:1093`, and
  `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:423`/`:441`.
- **Why it matters:** catches a broken rebuild *before* the application's next collective
  does, with a per-mapping culprit list.

#### Non-IPC hooks that live in this file (grouped)

`ipc_hooks.cpp` also hosts a family of hooks that have nothing to do with IPC teardown but
share the interception machinery: the kernel-launch wrappers `cudaLaunchKernel`,
`cuLaunchKernel`, `cuLaunchKernelEx`, `cudaLaunchKernelExC`, their `_ptsz` ("per-thread
default stream") and `__cudaLaunchKernel` variants (`src/ipc_hooks.cpp:926-1218`). Each
checks whether the thread's current CUDA context matches `g_pytorch_context` (captured in
`src/GPUs/NVIDIA/nv.cpp:18` when the application's context is first observed) and, if not,
pushes the captured context around the real launch — a workaround for restore paths where
the wrong context is current. `cudaGetLastError`/`cudaPeekAtLastError`/
`cudaStreamSynchronize` (+`_ptsz`) (`:1220-1274`) are log-only pass-throughs, and
`hook_cuCtxCreate`/`hook_cuCtxDestroy`/`hook_cuDevicePrimaryCtxRetain`/
`hook_cuDevicePrimaryCtxRelease` (`:2826-2872`) log context lifecycle. Treat them as
observability plus the context pinning fix; none of them touch the IPC tables.

---

## `src/ipc_fd_exchange.h` / `src/ipc_fd_exchange.cpp` — fd transfer over Unix sockets

This small module exists because of a hard-won lesson recorded in its header comment:
CUDA shareable fds can **only** cross process boundaries via SCM_RIGHTS. `pidfd_getfd`
needs `CAP_SYS_PTRACE`, and opening `/proc/<pid>/fd/N` re-opens the device node without
the CUDA handle metadata — the import then fails or maps garbage.

Wire protocol (both directions little-endian ints):
client → server: `num_fds` (4 bytes), then `num_fds` fd numbers; server → client:
`send_count` (4 bytes), then one 1-byte message carrying all fds as SCM_RIGHTS ancillary
data.

#### `make_sock_path()` (`src/ipc_fd_exchange.cpp:44`)

- **Purpose:** formats `/tmp/cr_ipc_fd_<pid>.sock`. The PID in the name is how importers
  find a specific peer's server — no registry needed.
- **Called by:** `uds_fd_server_start` (`:191`), `uds_receive_fds` (`:322`).

#### `send_fds()` (`src/ipc_fd_exchange.cpp:64`)

- **Purpose:** the SCM_RIGHTS send. Builds a `msghdr` with a 1-byte payload (`sendmsg`
  requires at least one real byte), attaches a control message of level `SOL_SOCKET`,
  type `SCM_RIGHTS`, whose data is the raw array of fd ints, and calls `sendmsg`. The
  kernel intercepts the control message and installs duplicates of those fds in the
  receiving process.
- **Called by:** `uds_server_thread` (`src/ipc_fd_exchange.cpp:171`).
- **Why it matters:** this one `sendmsg` call is the entire reason the module exists.

#### `uds_server_thread()` (`src/ipc_fd_exchange.cpp:101`)

- **Purpose:** the server loop, run on a dedicated pthread. Each iteration: set a 50 ms
  receive timeout on the listen socket (so `accept` wakes up periodically to check the
  `running` flag — this is the shutdown mechanism), `accept` one connection, read the
  request (`num_fds` then the fd list, both with `MSG_WAITALL`, rejecting counts outside
  1..`UDS_MAX_FDS`=64), sanity-check each requested fd with `fcntl(F_GETFD)` (invalid ones
  are *still sent* — the comment says the client handles the error; in practice the CUDA
  import of a bogus fd fails and that pair is skipped), send `send_count`, then
  `send_fds()`. Serves connections sequentially, looping so every peer can connect in turn.
- **Called by:** spawned by `uds_fd_server_start` via `pthread_create`
  (`src/ipc_fd_exchange.cpp:229`).
- **Why it matters:** during restore Phase 3b, every worker is simultaneously a server
  (for its own exports) and a client (for its imports); the background thread is what lets
  a worker answer peers while its own signal-handler thread is busy importing.

#### `uds_fd_server_start()` (`src/ipc_fd_exchange.cpp:184`)

- **Purpose:** idempotent server startup: unlink any stale socket file, create an
  `AF_UNIX`/`SOCK_STREAM` socket, `bind` it to the path, `listen`, `chmod 0777` the socket
  file, set `running=1`, spawn the thread. Every failure path cleans up and returns -1.
- **Called by:** `src/vGPU.cpp:1008` (end of the `IPC_EXPORT_MSG` phase — i.e. **after**
  new fds exist), `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:382`.
- **Why it matters:** must be running before any peer enters the import phase; the
  coordinator guarantees this by not broadcasting `IPC_IMPORT_MSG` until *all* workers
  finished the export phase (`coordinator/multi_cr_client.cpp:633-639`).

#### `uds_fd_server_stop()` (`src/ipc_fd_exchange.cpp:243`)

- **Purpose:** clear `running`, close the listen fd, `pthread_join` the thread (bounded by
  the 50 ms accept timeout), unlink the socket file.
- **Called by:** `src/vGPU.cpp:1041` (during `IPC_IMPORT_MSG`, after this worker's own
  imports are rebuilt), `adapters/nccl/runtime/gcr_checkpoint_runtime.cpp:416`.
- **Why it matters:** see Gotchas — stopping "after my own imports" is safe only because
  the coordinator's barrier means all peers fetched what they needed... which is *not*
  strictly guaranteed within phase 3b; in practice each worker's server keeps running
  until that worker processes its import message, and the sequential
  `broadcast_and_wait` barrier ends only when all workers finish, so a worker that stops
  its server early can, in a pathological schedule, race a slow peer. The 20×10 ms
  connect retry in the client absorbs the startup direction of this race.

#### `recv_fds()` (`src/ipc_fd_exchange.cpp:273`)

- **Purpose:** the SCM_RIGHTS receive: `recvmsg` with a control buffer sized for
  `num_fds` ints, verify the control message is `SOL_SOCKET`/`SCM_RIGHTS`, compute how
  many fds actually arrived from `cmsg_len`, copy them out. Returns the count received.
- **Called by:** `uds_receive_fds` (`src/ipc_fd_exchange.cpp:383`).

#### `uds_receive_fds()` (`src/ipc_fd_exchange.cpp:315`)

- **Purpose:** the client side, one call per peer: build the peer's socket path, connect
  with up to 20 attempts spaced 10 ms apart (tolerates the peer's server starting
  slightly later), send the request (`num_fds` + fd-number list), read back `send_count`,
  then `recv_fds()` into `local_fds`. Returns the number of fds received or -1.
- **Called by:** `ipc_import_from_shm_block` (`src/ipc_hooks.cpp:2509`) — one call per
  distinct `owner_pid` in the peer block.
- **Why it matters:** the received fds are local, real, CUDA-importable handles; their
  order matches the requested order, which the positional matching in
  `ipc_import_from_shm_block` relies on.

#### `pidfd_copy_fd()` (`src/ipc_fd_exchange.cpp:408`) — deprecated

- **Purpose:** the abandoned approach, kept as executable documentation: `pidfd_open` on
  the target process, then the `pidfd_getfd` syscall to duplicate its fd. Prints a loud
  warning that this does NOT work for CUDA shareable handles.
- **Called by:** nobody (verified by grep — the only references are its own
  declaration/definition and comments).
- **Why it matters:** if you are tempted to "simplify" fd transfer, this function and the
  header comments are the record of why not.

---

## Key data structures

All of the following live in `src/ipc_hooks.cpp:49-123` and (except where noted) are
guarded by `g_ipc_hook_mutex`, a **recursive** mutex — recursive because the teardown
signal handler can interrupt the same thread while it is inside a hooked `cuMem*` call
that already holds the lock (comment at `src/ipc_hooks.cpp:120-123`).

| Structure | Keyed by | Grows when | Shrinks/changes when | Survives teardown? |
|---|---|---|---|---|
| `g_ipc_imports` (vector of `IpcImportRecord`) | position (creation order) | `hook_cuMemMap` of an imported handle | never removed; `torn_down` flag toggles; `handle` replaced at rebuild | **yes — this is the rebuild recipe** |
| `g_ipc_exports` (vector of `IpcExportRecord`) | position (creation order) | `hook_cuMemExportToShareableHandle` | never removed; `torn_down`, `local_handle`, `exported_fd` mutate | **yes** |
| `g_imported_handles` (set) | handle | `hook_cuMemImportFromShareableHandle` | `hook_cuMemRelease` | stale after teardown (handles die); harmless |
| `g_created_allocs` (map handle→`{size, prop}`) | handle | `hook_cuMemCreate` with `requestedHandleTypes!=0` | `hook_cuMemRelease`; re-keyed to new handles at rebuild | re-populated by rebuild |
| `g_handle_to_va` (map) | handle | `hook_cuMemMap` of own alloc | `hook_cuMemRelease`; re-keyed at rebuild | re-populated by rebuild |
| `g_vaddr_to_import_idx`, `g_vaddr_to_export_idx` (maps VA→vector index) | VA | on map/export | export index erased on release | yes (VAs never change — that's the point) |
| `g_va_access_descs` (map VA→vector of `CUmemAccessDesc`) | VA | `hook_cuMemSetAccess` (deduped per type+id) | erased on release of the owning handle | **yes — replayed verbatim at rebuild** |
| `g_local_allocs` (vector of `CuMemLocalAllocRecord`) | position | **built fresh inside `ipc_save_and_teardown_local_allocs`** | handles swapped at rebuild | created *at* teardown, consumed at rebuild |
| `g_rebuilt_handle_alias` (map old app handle→new handle) | handle | Phase C of export rebuild | purged in `hook_cuMemRelease` | created *by* rebuild |
| `g_peer_access_devices` / `g_saved_peer_access_devices` (sets, own mutex `g_peer_access_mutex`) | device id | enable hook / `ipc_disable_all_peer_access` | disable hook / `ipc_reenable_all_peer_access` | saved-set carries state across the freeze |
| `g_ipc_events` (vector, own mutex `g_ipc_events_mutex`) | position | the two event interposers | `torn_down` flag only | torn down, never rebuilt |
| `g_timing_snapshot` (`IpcTimingSnapshot`) | — | every phase writes its section | `ipc_reset_timing_snapshot` | diagnostic only |

**Invariants worth internalizing:**

- **Vectors are append-only.** Records are never erased, only flagged. Rebuild and the
  positional-matching import logic both depend on creation order being preserved.
- **The VA is the identity.** Every record's `mapped_vaddr`/`vaddr` is immutable across
  the checkpoint; handles and fds are disposable and replaced wholesale at rebuild.
  Corollary: VA reservations must never be freed between teardown and rebuild (no
  `cuMemAddressFree` anywhere in the teardown paths — grep confirms `fn_cuMemAddressFree`
  is resolved but never called).
- **`torn_down` is the state machine.** Teardown consumes `!torn_down` records and sets
  the flag; rebuild consumes `torn_down` records and clears it. Running a phase twice is
  therefore a no-op, and skipping teardown makes rebuild a no-op.
- **Exports are identified by VA, imports by handle-at-map-time.** Local (non-exported)
  allocs are "everything in `g_created_allocs` with a VA that isn't an export VA".

---

## Gotchas for maintainers

**Phase ordering is load-bearing.**
- Checkpoint: GPU sync → import teardown → export save+teardown → event teardown → local
  alloc save+teardown → P2P disable (`src/vGPU.cpp:848-948`). Imports before exports is
  the safe order (release your view of a peer's memory before the peer releases the
  memory itself — though within one process these are independent, across the fleet the
  coordinator's barrier makes all teardowns complete before any freeze). P2P disable must
  be last-ish and *after* any DtoH copies that might traverse peer paths.
- Restore: export rebuild (fds exist) → shm publish → **UDS server start** → coordinator
  exchange → import rebuild (fds fetched) → UDS stop → validate → P2P re-enable. Starting
  the fd server before the fds exist would serve stale numbers; broadcasting
  `IPC_IMPORT_MSG` before *every* worker finished `IPC_EXPORT_MSG` would make importers
  request fds that don't exist yet. `broadcast_and_wait`
  (`coordinator/multi_cr_client.cpp:287`) enforces both barriers — it signals **all**
  workers before waiting on **any**, which also matters because rebuild phases can
  interlock across ranks.

**fd lifetime rules.**
- Export fds are owned by the export record: closed in step 2 of
  `ipc_save_and_teardown_all_exports` (`src/ipc_hooks.cpp:1794-1800`), reborn with new
  numbers in Phase C of the rebuild. Never cache an fd number across the checkpoint.
- fds received via `uds_receive_fds` are owned by the importer and must be closed right
  after `cuMemImportFromShareableHandle` (done at `src/ipc_hooks.cpp:2592`); the CUDA
  handle keeps the memory alive independently. Unmatched fds are also closed
  (`:2649-2653`). Leaking one leaves an nvidia fd that blocks the *next* checkpoint.
- The fd numbers in the shm block are only valid while the exporting process keeps them
  open — i.e. between export rebuild and that process's next teardown.

**What happens if a phase is skipped.**
- Skip IPC teardown (`-n` flag, or single-GPU flow with NCCL active): `cuda-checkpoint`
  freeze fails or restore fails with "OS call failed"; the diagnostic is leftover
  anon_inode/nvidia fds in `ipc_dump_nvidia_fds` output.
- Skip export rebuild but run import rebuild: peers fetch nothing (`num_exports==0` short-
  circuits at `src/ipc_hooks.cpp:2455`), imports stay `torn_down`, NCCL faults on first use.
- Skip import rebuild: `torn_down` import records remain; a later validate reports FAIL on
  every import VA; the *next* checkpoint's `ipc_teardown_all_imports` skips them (flag
  already set), so the failure mode is application-level, not checkpoint-level.
- Run teardown twice: safe no-op (flags). Run rebuild twice: mostly safe no-op (rebuild
  only touches `torn_down` records), but a second `IPC_EXPORT_MSG` also restarts the UDS
  server, which is idempotent (`src/ipc_fd_exchange.cpp:185-188`).

**Positional matching is an assumption, not a law.** `ipc_import_from_shm_block` pairs
peer exports to local imports by order. This holds for NCCL's deterministic channel
setup. It will silently mismap if: (a) more than one *exporting peer* contributes and the
coordinator interleaves their entries differently than the original import order, (b) the
application creates IPC mappings in a nondeterministic order (multi-threaded init), or
(c) some exports/imports were legitimately released mid-run so counts diverge (you get
the WARNING at `src/ipc_hooks.cpp:2569-2574`). A silent mismap means two buffers swap
contents/owners — data corruption, not a crash. If you touch NCCL versions or add a
second IPC-using library (NVSHMEM!), re-verify this invariant first.

**Thread-safety notes.**
- `g_ipc_hook_mutex` is recursive *specifically* because the teardown runs inside a
  signal handler that can preempt a thread mid-`cuMem*` hook on the same thread.
  Everything the signal handler transitively calls must either take only this mutex or be
  async-signal-tolerant; do not add a plain `std::mutex` lock to any path reachable from
  `cr_ipc_signal_handler` that a hook also takes (`g_peer_access_mutex` and
  `g_ipc_events_mutex` are plain mutexes — they are safe today only because the hooks
  that take them are runtime-API interposers not called from within the signal path while
  held; keep it that way).
- The interception machinery itself is *not* lock-protected: `real_*` pointer stores are
  racy-but-benign idempotent writes, and `try_intercept` mutates the caller's pointer
  in place. Fine for init-time resolution; don't extend it to hot-swap hooks at runtime.
- `uds_server_thread` runs concurrently with everything; it only reads the process fd
  table and `g_server`, so it needs no IPC-table locks. Keep it that way — taking
  `g_ipc_hook_mutex` there could deadlock against a signal-handler import in progress.

**Two drivers, one engine.** Besides the signal-driven path in `src/vGPU.cpp`, the NCCL
adapter (`adapters/nccl/runtime/gcr_checkpoint_runtime.cpp`) calls the exact same
`ipc_*`/`uds_*` API in-process (prepare/restore-export/restore-import entry points). If
you change a function's contract here, check both call sites.

**Stale artifacts.** `ipc_rebuild_all_imports` (header-only, no definition), the
`pidfd_getfd` mentions in both headers, the never-called `pidfd_copy_fd`, and the header
claim that export teardown calls `cuMemAddressFree` (it doesn't — VA is preserved) are
all fossils of earlier designs. Prefer the .cpp behavior over the .h prose, and consider
cleaning these up in a dedicated commit.
