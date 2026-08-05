# 06 — Adapters (NCCL, NVSHMEM) and the Build/Release Toolchain

This chapter covers the two library adapters that make GPU-CR work with
multi-GPU communication libraries — the NCCL adapter under `adapters/nccl/`
and the NVSHMEM examples under `adapters/nvshmem/` — plus the complete
build and release toolchain: the top-level `CMakeLists.txt`, the `Makefile`,
`Dockerfile.build`, `cloudbuild-so.yaml`, and the GitHub release workflow.

**Background: shared libraries and `.so` files.** On Linux, a `.so`
("shared object") file is a library loaded into a process at runtime rather
than copied into the executable at compile time. The dynamic linker resolves
each function name (a *symbol*) to an address when the program starts or when
the library is explicitly loaded. Two mechanisms matter constantly in this
repo: **`LD_PRELOAD`**, an environment variable that tells the dynamic linker
to load a chosen `.so` *before* everything else so that any function it
defines "shadows" (interposes on) the same-named function in later libraries;
and **`dlopen`/`dlsym`**, C functions that load a `.so` and look up a symbol
in it by string name at runtime — this is how plugins work. **Symbol
visibility** controls which functions in a `.so` are lookup-able from
outside; GPU-CR builds everything with default (public) visibility precisely
so that `dlsym` and `LD_PRELOAD` interposition can find its functions.

**Background: NCCL and communicators.** NCCL (NVIDIA Collective
Communications Library, pronounced "nickel") is the library virtually all
multi-GPU training and inference stacks (PyTorch, vLLM, ...) use to move data
between GPUs — AllReduce, Broadcast, etc. A **communicator** (`ncclComm_t`)
is NCCL's handle for one rank's membership in a group of GPUs; behind it NCCL
allocates internal GPU buffers and *shares them across processes* (rank A
maps rank B's buffer into its own address space) so kernels can write
directly into a peer's memory. Those cross-process mappings are exactly what
breaks checkpointing: NVIDIA's `cuda-checkpoint` utility refuses to freeze a
process whose driver state still contains live IPC (inter-process
communication) memory shares. So before a checkpoint, every imported peer
mapping and every exported local buffer must be torn down, and after restore
they must be rebuilt at the *same virtual addresses* so NCCL's stored
pointers remain valid. That teardown/rebuild is the entire job of this
adapter stack.

## The NCCL checkpoint call chain

There are four layers. An orchestrator (the application itself, a test
driver, or a vLLM-side controller) calls the public API that the GPU-CR
patches add to NCCL; patched NCCL dispatches into a plugin `.so`; the plugin
forwards into the GCR runtime that was LD_PRELOAD-ed into the process; the
runtime drives the cuMem IPC bookkeeping kept by `src/ipc_hooks.cpp`.

```
+---------------------------------------------------------------+
| 1. ORCHESTRATOR (application / test driver / vLLM controller) |
|    ncclCommCheckpointPrepare(comm, NCCL_CKPT_MODE_GCR_GLOBAL)  |
|    ...cuda-checkpoint freeze / swap / unfreeze happens here... |
|    ncclCommCheckpointRestore(comm, ... | RESTORE_EXPORT)       |
|    ncclCommCheckpointRestore(comm, ... | RESTORE_IMPORT)       |
+-------------------------------+-------------------------------+
                                | public C API added by nccl_patches/
                                v
+---------------------------------------------------------------+
| 2. PATCHED NCCL (libnccl.so + the 6-piece patch set)           |
|    checkpoint_api.cc: validates flags, picks NATIVE vs GCR     |
|    src/plugin/checkpoint.cc: dlopen($NCCL_CHECKPOINT_PLUGIN),  |
|      dlsym("ncclCheckpointPlugin_v1"), call init/prepare/      |
|      restore                                                   |
+-------------------------------+-------------------------------+
                                | dlopen + dlsym (plugin ABI v1)
                                v
+---------------------------------------------------------------+
| 3. GCR PLUGIN  libnccl-checkpoint-gcr.so                       |
|    gcrPluginInit: dlsym(RTLD_DEFAULT, "gcr_checkpoint_*")      |
|    gcrPluginPrepare -> gcr_checkpoint_prepare(flags)           |
|    gcrPluginRestore -> restore_export / restore_import         |
+-------------------------------+-------------------------------+
                                | plain C calls into the preloaded runtime
                                v
+---------------------------------------------------------------+
| 4. GCR RUNTIME  libgcr_preload.so (LD_PRELOAD-ed)              |
|    gcr_checkpoint_runtime.cpp orchestrates the phases;         |
|    src/ipc_hooks.cpp holds the import/export/local-alloc       |
|    records it tears down and rebuilds;                         |
|    src/ipc_fd_exchange.cpp moves the new file descriptors      |
|    between ranks over a Unix domain socket.                    |
+---------------------------------------------------------------+
```

Note the direction: control always *enters* through patched NCCL and *ends*
in the runtime. The alternative control path in this repo — `multi_cr_client`
sending signals to a process preloaded with `vGPU-NVIDIA.so` — reaches the
very same `ipc_*` machinery, but through signal handlers in `src/vGPU.cpp`
instead of through NCCL and the plugin. Pick one path per deployment; the
NCCL-plugin path uses `libgcr_preload.so`, the signal path uses
`vGPU-NVIDIA.so` (verified: no `gcr_checkpoint_*` symbol exists anywhere in
`src/` or `coordinator/`, so `vGPU-NVIDIA.so` cannot serve the plugin path).

---

## NCCL adapter — `adapters/nccl/`

### `README.md`

- **Purpose:** Explains the two-library architecture (runtime preloaded via
  `LD_PRELOAD`, plugin loaded inside NCCL via `NCCL_CHECKPOINT_PLUGIN`), the
  build flags, and the mandatory runtime environment.
- **Why it matters:** It documents the three environment settings that
  people forget most often: `NCCL_CUMEM_ENABLE=1` (route NCCL's buffer
  sharing through the `cuMem*` driver APIs GPU-CR intercepts — without it
  NCCL uses legacy CUDA IPC, GPU-CR tracks nothing, and restore fails with
  "operation not supported"), `NCCL_CUMEM_HOST_ENABLE=1` and
  `NCCL_P2P_DISABLE=1` for vLLM-style deployments, and the warning to *not*
  set `NCCL_P2P_DISABLE=1` for the bundled two-process test (its cuMem
  exports/imports come precisely from the P2P transport).

### `CMakeLists.txt` (adapter)

- **Purpose:** Builds the two adapter libraries and, optionally, the example
  binary. Requires `-DNCCL_ROOT=/path/to/nccl` (an NCCL *source* tree,
  ideally already built) and fails with a clear message otherwise. It
  discovers the generated public header `nccl.h` under
  `${NCCL_ROOT}/{include,build/include,nccl_install_stage{1,2}/include}` and
  the internal header `nccl_common.h` under `${NCCL_ROOT}/src/include`
  (overridable with `-DNCCL_PUBLIC_INCLUDE_DIR` /
  `-DNCCL_INTERNAL_INCLUDE_DIR`), and looks for `libnccl` for the example.
- **Targets produced:**
  - `gcr_preload` → `libgcr_preload.so`, built from
    `runtime/gcr_checkpoint_runtime.cpp` **plus** the core sources
    `src/ipc_hooks.cpp` and `src/ipc_fd_exchange.cpp`. So the adapter runtime
    is a *second packaging* of the core IPC tracking, without `vGPU.cpp`'s
    signal handlers. Built with `CXX_VISIBILITY_PRESET default` so
    `gcr_checkpoint_*` are visible to `dlsym(RTLD_DEFAULT, ...)`.
  - `nccl_checkpoint_gcr` → `libnccl-checkpoint-gcr.so`, from
    `plugin/nccl_checkpoint_gcr_plugin.cpp` only, linked only against `dl`.
    It compiles against the *bundled* `nccl_patches/nccl_checkpoint.h` (the
    include path lists `nccl_patches/` first), so the plugin and the patch
    always agree on the ABI struct.
  - `two_proc_nccl_test` — only if `libnccl` was found; otherwise CMake
    prints a status line and skips it.
- **Loaded by:** the top-level build via
  `add_subdirectory(adapters/nccl)` when `GPU_CR_BUILD_NCCL_ADAPTER=ON`
  (NVIDIA only; the top-level file hard-errors on AMD).
- **Why it matters:** the library *file names* are load-bearing. NCCL's
  generic plugin loader registers the prefix `libnccl-checkpoint` for this
  plugin type (see `plugin_loader.patch`), and the runtime must be a
  standalone `.so` you can put in `LD_PRELOAD`.

### `include/gcr_checkpoint.h` — the public adapter API

Four C functions, deliberately independent of NCCL:

```c
int gcr_checkpoint_prepare(int flags);
int gcr_checkpoint_restore_export(int flags);
int gcr_checkpoint_restore_import(int flags);
int gcr_checkpoint_validate(int flags);
```

- **Purpose:** The C ABI contract between the plugin (or any other caller)
  and the runtime. Return 0 on success, -1 on failure.
- **Called by (verified):** `plugin/nccl_checkpoint_gcr_plugin.cpp` resolves
  all four with `dlsym(RTLD_DEFAULT, ...)`; the NVSHMEM examples
  (`adapters/nvshmem/examples/*.c*`) resolve and call the first three the
  same way. Nothing in `src/` or `coordinator/` references them.
- **Why it matters:** because the linkage is by *string lookup at runtime*
  (`dlsym`), there is no link-time dependency between plugin and runtime.
  The plugin can live inside NCCL's process without ever being compiled
  against GPU-CR — but it also means a typo'd or hidden symbol fails only at
  runtime, and only with a stderr message.

### `runtime/gcr_checkpoint_runtime.cpp`

- **Purpose:** Implements the four `gcr_checkpoint_*` entry points. It is
  the phase orchestrator that turns the flat `ipc_*` primitives from
  `src/ipc_hooks.cpp` into the three-phase checkpoint protocol:
  - `gcr_checkpoint_prepare(flags)` — idempotent per process (a mutex plus a
    `g_prepared` latch). Sequence: `cudaDeviceSynchronize`; dump state and
    open NVIDIA fds; `ipc_teardown_all_imports()`; size and allocate a host
    staging buffer, then `ipc_save_and_teardown_all_exports()`;
    `ipc_teardown_all_events()`; second staging buffer, then
    `ipc_save_and_teardown_local_allocs()`; finally
    `ipc_disable_all_peer_access()`. Emits a loud warning if
    imports == exports == locals == 0 (the classic "you forgot
    `NCCL_CUMEM_ENABLE=1`" symptom).
  - `gcr_checkpoint_restore_export(flags)` — after `cuda-checkpoint` has
    restored the process: `ipc_rebuild_and_restore_all_exports()` and
    `ipc_rebuild_local_allocs()` from the staging buffers (which are then
    freed), writes this rank's new export handles to the shared-memory file
    named by `GCR_EXPORT_SHM_PATH` (**required** if any exports exist —
    hard error otherwise), then starts the Unix-domain-socket fd server
    (`uds_fd_server_start()`) so peers can fetch the new file descriptors.
  - `gcr_checkpoint_restore_import(flags)` — reads the *peer's* shm file
    named by `GCR_IMPORT_SHM_PATH` (required if imports exist) and
    `ipc_import_from_shm_block()`; optionally stops the fd server
    (`GCR_STOP_FD_SERVER_AFTER_IMPORT=1`, default keeps it alive for peer
    restore overlap); if `flags & 0x0400` (the value of
    `NCCL_CKPT_OPT_VALIDATE`, locally re-#defined as
    `GCR_CKPT_OPT_VALIDATE`), runs `ipc_validate_all_mappings()`; re-enables
    peer access; clears `g_prepared`.
  - `gcr_checkpoint_validate(flags)` — thin wrapper over
    `ipc_validate_all_mappings("GCR validate")`.
- **Storage backends:** the staging buffers honor `GCR_STORAGE_BACKEND` =
  `anon` (default, anonymous `mmap`), `pinned` (`cudaHostAlloc`), or
  `hugepage` (file-backed `mmap` under `GCR_STORAGE_DIR`, default
  `/mnt/huge-ckpt/nccl-gcr`). Unknown values warn and fall back to `anon`.
- **Metrics:** if `GCR_IPC_SCALING_METRICS_CSV` is set, every phase appends
  a wide CSV row (flock-protected, header written when the file is empty)
  with per-phase timing from `IpcTimingSnapshot` and the
  `GCR_IPC_SCALING_{MODE,RANK,CASE,BUFFER_GB}` labels.
- **Called by:** the plugin (NCCL path) and the NVSHMEM example binaries
  (direct `dlsym` path). Loaded into the target process only via
  `LD_PRELOAD=libgcr_preload.so`.
- **Why it matters:** this file is the *only* place the multi-process
  restore handshake (export shm → fd server → import shm) is sequenced for
  the adapter path. Order bugs here (e.g. starting the fd server before
  writing export info) show up as peers hanging in `restore_import`.

### `plugin/nccl_checkpoint_gcr_plugin.cpp`

- **Purpose:** The thin bridge NCCL loads. Exports exactly one data symbol:

  ```c
  ncclCheckpointPlugin_v1_t ncclCheckpointPlugin_v1 = {
    "gcr", gcrPluginInit, gcrPluginPrepare, gcrPluginRestore, gcrPluginFinalize
  };
  ```

  `gcrPluginInit` resolves the four runtime functions with
  `dlsym(RTLD_DEFAULT, ...)` and fails with `ncclSystemError` if any of
  `prepare`/`restore_export`/`restore_import` are missing (`validate` is
  optional). `gcrPluginPrepare` forwards flags to `gcr_checkpoint_prepare`.
  `gcrPluginRestore` is where the `NCCL_CKPT_RESTORE_EXPORT` /
  `NCCL_CKPT_RESTORE_IMPORT` bits are *actually consumed*: it dispatches to
  the matching runtime entry point and returns `ncclInvalidUsage` if neither
  bit is set. The `comm` argument is deliberately ignored — teardown is
  process-wide, not per-communicator.
- **Loaded by (verified in the patch files):** patched NCCL's
  `src/plugin/checkpoint.cc` calls `ncclOpenCheckpointPluginLib(name)` with
  the value of `$NCCL_CHECKPOINT_PLUGIN`, then
  `ncclOsDlsym(lib, "ncclCheckpointPlugin_v1")`
  (`NCCL_CHECKPOINT_PLUGIN_SYMBOL` in `nccl_checkpoint.h`).
- **Why it matters:** it decouples NCCL from GPU-CR entirely. NCCL only
  knows the v1 struct; GPU-CR only knows four C functions. Either side can
  be rebuilt independently as long as both contracts hold.

### `nccl_patches/` — the upstream NCCL patch set

Six pieces, intentionally minimal (all real work stays in GPU-CR):

| File | Lands at (in the NCCL tree) | What it adds |
|------|-----------------------------|--------------|
| `nccl.h.in.checkpoint-additions` | pasted into `src/nccl.h.in` | The public surface: `NCCL_CKPT_*` flag macros (`MODE_NATIVE 0x0001`, `MODE_GCR_GLOBAL 0x0004`, `OPT_SAVE_DATA 0x0100`, `OPT_NO_CUDA_CKPT 0x0200`, `OPT_VALIDATE 0x0400`, `RESTORE_EXPORT 0x1000`, `RESTORE_IMPORT 0x2000`) and the declarations of `ncclCommCheckpointPrepare/Restore` (+ `pnccl` twins). These values are canonical — the prebuilt plugin and examples are compiled against them. |
| `checkpoint_api.cc` | `src/checkpoint_api.cc` + add to the `LIBSRCFILES` list in `src/Makefile` | The two public entry points. `Prepare`: NATIVE mode forwards to `ncclCommSuspend` when the tree has it (`NCCL_SUSPEND_MEM` macro; upstream ≥ ~2.30), else returns `ncclInvalidUsage`; GCR_GLOBAL mode validates the flag set and calls `ncclCheckpointPluginPrepare(comm, flags)`. `Restore`: NATIVE → `ncclCommResume`; GCR_GLOBAL requires exactly one of RESTORE_EXPORT/RESTORE_IMPORT and calls `ncclCheckpointPluginRestore`. |
| `checkpoint.h` | `src/include/checkpoint.h` | Two-line internal header declaring `ncclCheckpointPluginPrepare/Restore` so `checkpoint_api.cc` can dispatch into the loader. |
| `nccl_checkpoint.h` | `src/include/plugin/nccl_checkpoint.h` | The plugin ABI: `ncclCheckpointPlugin_v1_t` (name + init/prepare/restore/finalize function pointers) and `NCCL_CHECKPOINT_PLUGIN_SYMBOL "ncclCheckpointPlugin_v1"`. The GPU-CR plugin compiles against this same file straight from `nccl_patches/`. |
| `checkpoint.cc` | `src/plugin/checkpoint.cc` | The loader. Lazily, under a mutex with a load-status latch: reads `$NCCL_CHECKPOINT_PLUGIN` (the literal `none` disables), `ncclOpenCheckpointPluginLib(name)` → `dlsym` the v1 struct → verify `init`/`prepare`/`restore` are non-null → call `init(&context, ncclDebugLog)` → register `atexit` finalize. A failed load is latched; subsequent checkpoint calls return `ncclInvalidUsage` without retrying. |
| `plugin_loader.patch` | `git apply` from the NCCL source root; touches `src/include/plugin/plugin.h` and `src/plugin/plugin_open.cc` | Registers the new plugin *type* with NCCL's generic loader: adds `ncclPluginTypeCheckpoint` to the enum, declares `ncclOpenCheckpointPluginLib`, bumps `NUM_LIBS` 5→6, and appends `"CHECKPOINT"` / `"libnccl-checkpoint"` / subsystem entries to the four parallel arrays. Generated against upstream v2.30.4-1; the hunks are small and apply with fuzz on nearby versions. |

**How the plugin gets loaded, end to end:** the first time an application
calls `ncclCommCheckpointPrepare(comm, NCCL_CKPT_MODE_GCR_GLOBAL)`,
`checkpoint_api.cc` → `ncclCheckpointPluginPrepare` → `ncclCheckpointPluginLoad`
→ `ncclOpenCheckpointPluginLib($NCCL_CHECKPOINT_PLUGIN)` (same generic
`dlopen` machinery as NCCL's net/tuner/profiler plugins, with the
`libnccl-checkpoint` name prefix registered by `plugin_loader.patch`) →
`dlsym("ncclCheckpointPlugin_v1")` → `init()`. With `NCCL_DEBUG=INFO` you see
`Successfully loaded external checkpoint plugin gcr`. In GPU-CR usage
`NCCL_CHECKPOINT_PLUGIN` is set to the full path of
`libnccl-checkpoint-gcr.so`.

**The full call chain (all four layers), concretely:**

1. Orchestrator — either the application itself (as in
   `two_proc_nccl_test`), or an external controller. In the
   signal-driven deployment, `multi_cr_client` + `vGPU-NVIDIA.so` play the
   orchestration role and reach the same `ipc_*` core via signal handlers
   in `src/vGPU.cpp`, bypassing layers 2 and 3 entirely.
2. Patched NCCL — `ncclCommCheckpointPrepare/Restore` validate flags,
   dispatch to the plugin loader.
3. Plugin — `libnccl-checkpoint-gcr.so`, `gcrPlugin{Init,Prepare,Restore}`.
4. Runtime — `gcr_checkpoint_{prepare,restore_export,restore_import}` in
   `libgcr_preload.so`, driving `ipc_*` in `src/ipc_hooks.cpp` and the fd
   exchange in `src/ipc_fd_exchange.cpp`. Between prepare and restore, the
   orchestrator runs `cuda-checkpoint` (lock/checkpoint then
   restore/unlock, or `--toggle`) against each rank's PID.

**Why `CUDARTLIB=cudart` when rebuilding NCCL is mandatory:** NCCL's default
build links the CUDA runtime *statically* (`cudart_static`). NCCL resolves
every CUDA driver function — including `cuMemCreate`,
`cuMemExportToShareableHandle`, etc. — through `cudaGetDriverEntryPoint`,
and with a static cudart that call never leaves `libnccl.so` (no PLT
indirection), so GPU-CR's `LD_PRELOAD` interception is invisible to it.
Symptom: `imports=0 exports=0` at prepare and a later
"operation not supported" from `cuda-checkpoint --action restore`. Building
with `make -j src.build CUDARTLIB=cudart` links cudart dynamically and the
preload library interposes normally.

### `examples/two_proc_nccl_test.cpp`

- **Purpose:** End-to-end demonstrator: forks two child ranks (one GPU
  each), builds a 2-rank communicator over a shared anonymous `mmap`
  (process-shared pthread barrier/mutex/cond + the `ncclUniqueId`), runs an
  AllReduce, checkpoints, restores, and re-runs the AllReduce, verifying
  every element equals 3.0 (rank values 1+2).
- **Modes:** `native` (pure `NCCL_CKPT_MODE_NATIVE`, i.e. suspend/resume
  inside NCCL — no GPU-CR involved), `gcr` (GCR prepare/restore without an
  actual process freeze), `gcr-ckpt-action` and `gcr-ckpt-toggle` (full flow:
  parent runs `cuda-checkpoint` `lock`+`checkpoint` / `restore`+`unlock`, or
  `--toggle`, against both child PIDs between prepare and restore; requires
  the `cuda-checkpoint` binary path as `argv[2]`).
- **Details worth knowing:** it sets `NCCL_CUMEM_ENABLE=1` itself (without
  overwriting an explicit user value); each rank sets `GCR_EXPORT_SHM_PATH`
  to its own `/tmp/gcr_nccl_stage2_<pid>_rank<r>.shm` before the EXPORT
  phase and `GCR_IMPORT_SHM_PATH` to the *other* rank's file before the
  IMPORT phase — that swap is the whole cross-rank handshake. The
  `cuda-checkpoint` child process unsets `LD_PRELOAD` and
  `NCCL_CHECKPOINT_PLUGIN` before `execl` so the tool itself is not
  intercepted. If the external checkpoint phase fails, the parent SIGKILLs
  the children rather than letting them hang in a half-restored CUDA state.
- **Built by:** the adapter CMakeLists, only when `libnccl` is found.
- **Why it matters:** it is the reference implementation of the orchestrator
  layer — when integrating GPU-CR into a new application, mirror this file's
  call order and environment handling.

---

## NVSHMEM adapter — `adapters/nvshmem/`

**Background: NVSHMEM.** NVSHMEM is NVIDIA's implementation of the OpenSHMEM
partitioned-global-address-space model for GPUs: every process (a "PE")
allocates from a *symmetric heap* — a region that exists at the same logical
offset on every PE — and any PE can `put`/`get` directly into any other PE's
heap, from host code or from inside a CUDA kernel. Like NCCL with
`NCCL_CUMEM_ENABLE=1`, NVSHMEM builds its cross-process mappings on the CUDA
VMM (`cuMem*`) driver APIs.

### Why there is NO patch for NVSHMEM

The NCCL adapter needs a patch only because NCCL wants an explicit public
checkpoint API on the communicator. NVSHMEM needs nothing, because GPU-CR's
interception already sits *below* it: `src/ipc_hooks.cpp` hooks the symbol
resolution functions themselves — `dlsym` (yes, GPU-CR interposes libc's
`dlsym`), `cudaGetDriverEntryPoint(ByVersion)`, and
`cuGetProcAddress`/`cuGetProcAddress_v2` — and substitutes tracking wrappers
whenever *any* library asks for `cuMemCreate`,
`cuMemExport/ImportFromShareableHandle`, `cuMemMap`, `cuMemUnmap`,
`cuMemSetAccess`, or `cuMemRelease`. NVSHMEM's symmetric-heap allocations go
through exactly these entry points, so as long as `libgcr_preload.so` (or
`vGPU-NVIDIA.so`) is preloaded, the same teardown/rebuild that serves NCCL
covers NVSHMEM with zero NVSHMEM source changes. This directory therefore
ships only demonstrators and run scripts.

### `CMakeLists.txt` (NVSHMEM)

- **Purpose:** Builds the two example binaries. Requires
  `-DNVSHMEM_ROOT=...`; finds `nvshmem.h` under
  `${NVSHMEM_ROOT}/{include,build/include}`. Library discovery knows about
  the pip-installed `nvidia-nvshmem` layout, which ships only the
  *versioned* `libnvshmem_host.so.3` with no unversioned symlink — the
  `find_library(NAMES ...)` list includes the versioned filename explicitly.
- **Targets produced:**
  - `nvshmem_host_test` (plain C++, host-side NVSHMEM calls) — only if
    `libnvshmem_host` is found.
  - `nvshmem_device_test` (`.cu`, device kernels) — only if the static
    `libnvshmem_device.a` is found; enables the CUDA language, defaults
    `CMAKE_CUDA_ARCHITECTURES` to `70;80;90` (guarded with a bool test, not
    `NOT DEFINED`, because CMake may auto-cache an empty value), and turns
    on separable compilation (required for NVSHMEM device linking).
- **Loaded by:** the top-level build via `add_subdirectory` when
  `GPU_CR_BUILD_NVSHMEM_ADAPTER=ON` (NVIDIA only).

### `examples/`

- **`nvshmem_host_test.cpp` — Purpose:** two ranks bootstrap via a
  unique-id file (rank 0 writes, rank 1 polls), `nvshmem_malloc` a
  symmetric buffer, exchange values with `nvshmem_int_p`, then walk a
  file-based control protocol (`<control-dir>/prepare`, `restore_export`,
  `restore_import` trigger files; `rank<N>.prepared` etc. acknowledgment
  files) around the checkpoint. **Called by:** an external driver script
  that touches the trigger files and runs `cuda-checkpoint` in between.
  Crucially, it calls the GCR runtime *directly* — `dlsym(RTLD_DEFAULT,
  "gcr_checkpoint_prepare")` and friends — demonstrating that the adapter
  API works without NCCL or any plugin at all.
- **`nvshmem_device_test.cu` — Purpose:** the same flow, but the data
  movement happens inside CUDA kernels (`nvshmem_int_p` / `nvshmem_int_g`
  launched via `nvshmemx_collective_launch`), proving that device-initiated
  symmetric-heap traffic also survives checkpoint/restore.
- **`run_nvshmem_host_test.sh` — Purpose:** the full driver: preloads
  `libgcr_preload.so`, launches both ranks with swapped
  `GCR_EXPORT_SHM_PATH`/`GCR_IMPORT_SHM_PATH`, orchestrates the trigger
  files, runs `cuda-checkpoint` (`action` or `toggle` mode via
  `CUDA_CKPT_MODE`), samples GPU memory via `nvidia-smi` at each phase, and
  writes a `summary.csv`. It sets the NVSHMEM environment that keeps the
  cuMem path active: `NVSHMEM_DISABLE_CUDA_VMM=0`,
  `NVSHMEM_REMOTE_TRANSPORT=none`, `NVSHMEM_DISABLE_NCCL=1`,
  `NVSHMEM_BOOTSTRAP=UID`. It refuses to run if other GPU compute processes
  exist (override `ALLOW_EXISTING_GPU_PROCS=1`) or if the hugepage mount is
  missing when `GCR_STORAGE_BACKEND=hugepage` (the script's default).
- **`run_nvshmem_device_test.sh`** — 7-line wrapper that sets
  `BENCH_NAME=nvshmem_cumem_checkpoint_device_test` and execs
  `run_nvshmem_cumem_checkpoint.sh`.
- **Why it matters / current state (verified):** the run scripts carry
  benchmark-era paths that do NOT match the CMake targets: they expect the
  binary at `build_stage2/nvshmem_cumem_checkpoint_test` (CMake builds
  `build/adapters/nvshmem/nvshmem_host_test`), the device wrapper execs
  `run_nvshmem_cumem_checkpoint.sh` which does not exist in the directory,
  and the host script's `CUDA_CKPT` default references `${GPU_CR_ROOT}`
  (never set; the script defines `GCR_ROOT`), which under `set -u` aborts
  unless you export `CUDA_CKPT` yourself. Expect to set `BUILD_DIR`-related
  variables and `CUDA_CKPT` explicitly, or fix the scripts. The README's
  mention of `nvshmrun` is also aspirational — the host script launches the
  two ranks directly with `env ... &`.

---

## How the core library interacts with the adapters (brief)

- **`src/ipc_hooks.cpp` (~2900 lines, shared verbatim by both preload
  libraries):** interposes `dlsym`, `cudaGetDriverEntryPoint(ByVersion)` and
  `cuGetProcAddress(_v2)` to substitute tracking hooks for the `cuMem*`
  family; keeps vectors of import/export/local-alloc/`cuMemSetAccess`
  records under a recursive mutex; exposes the `ipc_teardown_*` /
  `ipc_save_and_teardown_*` / `ipc_rebuild_*` / `ipc_validate_*` /
  peer-access APIs the runtime and `vGPU.cpp` call; also hooks the legacy
  `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle` purely to print loud
  warnings when an application takes the untrackable legacy path. Teardown
  frees VA space (`cuMemAddressFree`) because `cuda-checkpoint` requires
  it; rebuild re-reserves the *same* VA (`cuMemAddressReserve`) so pointers
  stored inside NCCL/NVSHMEM stay valid.
- **`src/nccl_hooks.cpp` (built into `vGPU-NVIDIA.so` only, not into
  `libgcr_preload.so`):** LD_PRELOAD shadows for `ncclCommInitRank(Config)`
  / `ncclCommDestroy` / `ncclCommFinalize` that call through to the real
  NCCL (found via a five-strategy `dlsym`/`dlopen` search: `RTLD_NEXT`,
  `RTLD_NOLOAD` on `libnccl.so.2`, `$CR_NCCL_LIB`, a path relative to the
  preload `.so`, plain `dlopen`) and merely *track* live communicators for
  diagnostics. Its former suspend/resume role is explicitly deprecated in
  favor of `ipc_hooks.cpp`.
- **`src/vGPU.cpp`:** the signal-driven orchestration (for
  `multi_cr_client`): `CR_IPC_TEARDOWN_SIGNAL` (SIGRTMAX-1) runs the same
  teardown sequence as `gcr_checkpoint_prepare`; SIGRTMAX-2 rebuilds;
  SIGRTMAX-3 validates. This is the path the released `vGPU-NVIDIA.so`
  serves.

---

## Upstream dependency contract

**NCCL versions the patches apply to.** The patch set targets the plugin-
loader conventions of upstream NCCL `master` after v2.28, and is validated
against **v2.30.x**; `plugin_loader.patch` was generated against
**v2.30.4-1** and its two hunks are small enough to apply with fuzz on
nearby versions. `NCCL_CKPT_MODE_NATIVE` additionally requires
`ncclCommSuspend`/`ncclCommResume` (upstream ≥ ~2.30, detected via the
`NCCL_SUSPEND_MEM` macro); on older trees NATIVE mode returns
`ncclInvalidUsage` while GCR_GLOBAL mode still works. Do **not** apply
`checkpoint_api.cc` to GCR's full internal NCCL tree — that tree already
defines `ncclCommCheckpointPrepare` (inside `mem_manager.cc`) and you would
get duplicate symbols.

**Symbols each side must export.**

- The **plugin** (`libnccl-checkpoint-gcr.so`) must export the data symbol
  `ncclCheckpointPlugin_v1` — a `ncclCheckpointPlugin_v1_t` with non-null
  `init`, `prepare`, `restore` (the loader rejects the plugin otherwise;
  `finalize` may be null).
- The **runtime** (`libgcr_preload.so`) must export, with default
  visibility, `gcr_checkpoint_prepare`, `gcr_checkpoint_restore_export`,
  `gcr_checkpoint_restore_import` (all required by `gcrPluginInit`) and
  optionally `gcr_checkpoint_validate`.
- **Patched NCCL** ends up exporting `ncclCommCheckpointPrepare`,
  `ncclCommCheckpointRestore` (plus `pnccl` variants via `NCCL_API`) and the
  internal `ncclCheckpointPluginPrepare/Restore`. Sanity check:
  `nm -D libnccl.so | grep -E "ncclCommCheckpoint|ncclCheckpointPlugin"`.

**Flag values are ABI.** The `NCCL_CKPT_*` values in
`nccl.h.in.checkpoint-additions` (0x0001/0x0004/0x0100/0x0200/0x0400/
0x1000/0x2000) are compiled into the prebuilt plugin, the runtime (which
re-#defines `GCR_CKPT_OPT_VALIDATE 0x0400` locally), and the examples.
Changing any value silently desynchronizes the layers.

**What breaks when upstream NCCL changes:**

- `plugin_loader.patch` edits *parallel arrays sized by `NUM_LIBS`* in
  `plugin_open.cc`. If upstream adds or reorders a plugin type, the patch
  conflicts (or worse, applies with fuzz and misaligns the arrays). Re-derive
  it against the new tree and keep `ncclPluginTypeCheckpoint` consistent
  between `plugin.h` and the array positions.
- `checkpoint.cc` uses NCCL-internal helpers (`ncclOsDlsym`,
  `ncclClosePluginLib`, `ncclGetEnv`, `ncclPluginLibPaths`, `INFO`/`WARN`,
  `NCCLCHECK`) — renames there break the loader compile.
- `nccl_checkpoint.h` includes internal `nccl_common.h` for
  `ncclDebugLogger_t`; if that type or header moves, both the patch *and*
  the plugin build (which includes the bundled copy) must be revised
  together — the ABI struct is shared, so version them in lockstep. A
  breaking struct change should become `ncclCheckpointPlugin_v2`.
- If NCCL stops resolving driver functions via
  `cudaGetDriverEntryPoint`/`cuGetProcAddress` (see
  `src/misc/cudawrap.cc` upstream), or the default build's static-cudart
  behavior changes, GPU-CR's interception model itself is affected — that is
  a `src/ipc_hooks.cpp` problem, not merely an adapter problem.

---

## Build & release runbook

### Top-level `CMakeLists.txt` — options and targets

| Option / cache var | Default | Effect |
|--------------------|---------|--------|
| `GPU_VENDOR` | `NVIDIA` (also read from the `GPU_VENDOR` env var) | Selects the backend. NVIDIA: CUDA at `CUDA_ROOT` (default `/usr/local/cuda`), links `cudart cuda`, excludes `src/**/AMD/`. AMD: ROCm at `ROCM_ROOT` (default `/opt/rocm`), links `amdhip64`, adds compile definition `__HIP_PLATFORM_AMD__=1`, excludes `src/**/NVIDIA/` **and** the CUDA-only `ipc_hooks.cpp`, `ipc_fd_exchange.cpp`, `nccl_hooks.cpp`. Anything else is a fatal error. |
| `SHM_SIZE_GB` | unset | If defined, adds compile definition `SHM_SIZE_GB=<n>` — the per-worker hugepage staging buffer size consumed by `vGPU.so` and `multi_cr_client` through `src/common.h`. Raise it (e.g. `-DSHM_SIZE_GB=40`) for vLLM at `gpu_memory_utilization` above ~0.5. |
| `GPU_CR_BUILD_NCCL_ADAPTER` | `OFF` | Adds `adapters/nccl`; requires `GPU_VENDOR=NVIDIA` and `NCCL_ROOT`. |
| `GPU_CR_BUILD_NVSHMEM_ADAPTER` | `OFF` | Adds `adapters/nvshmem`; requires `GPU_VENDOR=NVIDIA` and `NVSHMEM_ROOT`. |
| `NCCL_ROOT`, `NCCL_PUBLIC_INCLUDE_DIR`, `NCCL_INTERNAL_INCLUDE_DIR`, `NCCL_LIBRARY` | — | NCCL adapter discovery (see adapter section). |
| `NVSHMEM_ROOT`, `NVSHMEM_PUBLIC_INCLUDE_DIR`, `NVSHMEM_HOST_LIBRARY`, `NVSHMEM_DEVICE_LIBRARY`, `CMAKE_CUDA_ARCHITECTURES` | — | NVSHMEM examples discovery (see adapter section). |

Targets always produced:

- **`vGPU`** → `vGPU-${GPU_VENDOR}.so` (no `lib` prefix), from a
  `GLOB_RECURSE` of `src/*.cpp` minus `*-pre.cpp` scratch files and the
  per-vendor exclusions. Built `-fPIC -O3 -g` with default symbol
  visibility (required: its whole purpose is LD_PRELOAD interposition).
- **`cr_client`** — single-GPU checkpoint trigger CLI
  (`coordinator/cr_client.cpp` + `src/comm/share_mem.cpp`).
- **`multi_cr_client`** — multi-GPU phased orchestrator
  (`coordinator/multi_cr_client.cpp` + the same SHM control channel).

Because sources are globbed with `CONFIGURE_DEPENDS`, adding a `.cpp` under
`src/` silently joins `vGPU.so` on the next configure — including on AMD,
unless it matches an exclusion regex.

### Build each artifact locally

```bash
# Core (vGPU-NVIDIA.so, cr_client, multi_cr_client)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DGPU_VENDOR=NVIDIA
cmake --build build -j
# or equivalently, plus checksums:  make release-local

# NCCL adapter (first patch + rebuild NCCL per nccl_patches/README.md,
# with make -j src.build CUDARTLIB=cudart)
cmake -B build -S . -DGPU_VENDOR=NVIDIA \
      -DGPU_CR_BUILD_NCCL_ADAPTER=ON -DNCCL_ROOT=/path/to/nccl
cmake --build build -j
# -> build/adapters/nccl/libgcr_preload.so, libnccl-checkpoint-gcr.so,
#    two_proc_nccl_test (if libnccl found)

# NVSHMEM examples
cmake -B build -S . -DGPU_VENDOR=NVIDIA \
      -DGPU_CR_BUILD_NVSHMEM_ADAPTER=ON -DNVSHMEM_ROOT=/path/to/nvshmem
cmake --build build -j
# -> build/adapters/nvshmem/nvshmem_host_test, nvshmem_device_test
```

Compilation needs only CUDA *headers and libs*, not a GPU — a
`nvidia/cuda:*-devel` container on any machine works.

### `Makefile`

Thin convenience wrapper; variables `GPU_VENDOR?=NVIDIA`, `BUILD_DIR?=build`,
`DIST_DIR?=dist`, `CUDA_IMAGE?=nvidia/cuda:12.2.2-devel-ubuntu22.04`.

- **`make release-local`** (also `make`, `make all`, `make release`):
  configures Release, builds *only* the `vGPU` and `cr_client` targets,
  writes `build/checksums.sha256` over the two artifacts. Used verbatim by
  the GitHub release workflow and on any host that already has the CUDA
  toolkit. Note: `multi_cr_client` and the adapters are **not** part of the
  release build.
- **`make release-artifacts`**: same build, but inside a disposable
  `$(CUDA_IMAGE)` Docker container (installs cmake/build-essential, builds
  in `/tmp/build`, copies `cr_client` + `vGPU-$(GPU_VENDOR).so` +
  `checksums.sha256` into `dist/`). For hosts without CUDA installed. Runs
  as your uid/gid so `dist/` is not root-owned.
- **`make clean`**: removes `build/` and `dist/`.

### `.github/workflows/release.yml` — how a release is cut

Trigger: push of a tag matching `v*` (or manual `workflow_dispatch`). The
job runs on `ubuntu-22.04` *inside* the `nvidia/cuda:12.2.2-devel-ubuntu22.04`
container (so `make release-local` finds the CUDA toolkit without a GPU),
installs cmake/build-essential/git/ca-certificates, runs
`make release-local`, and — only for tag pushes — publishes a GitHub Release
via `softprops/action-gh-release@v2` with `build/cr_client`,
`build/vGPU-NVIDIA.so`, and `build/checksums.sha256`, with auto-generated
release notes. So cutting a release is:

```bash
git tag v0.1.1 && git push origin v0.1.1   # that's it
```

Introduced in commit `a759d54` ("Add v0.1.0 release automation workflow and
Makefile").

### `Dockerfile.build` + `cloudbuild-so.yaml` — the patched vGPU-NVIDIA.so cloud build

Commit `7489a5e` ("Add local-source Cloud Build for patched
vGPU-NVIDIA.so") added a second, deployment-flavored artifact pipeline.
`Dockerfile.build` builds `vGPU-NVIDIA.so` **from the local source tree**
(`COPY . /tmp/GPU-CR` — not a GitHub checkout) inside
`nvidia/cuda:13.0.0-devel-ubuntu22.04`, applying at build time, via `sed`,
the same two patches the deployed prebuilt carries (per the snapshot-agent
user guide, Step 2):

1. Signal remap in `src/common.h`: `CR_CKPT_SIGNAL SIGUSR1` →
   `(SIGRTMAX - 8)` and `CR_RESTORE_SIGNAL SIGUSR2` → `(SIGRTMAX - 7)`
   (frees SIGUSR1/2 for the application; typical for Python/vLLM workers).
2. Control-file permissions in `src/comm/share_mem.cpp`: the SHM control
   file is opened `0777` with an added `fchmod(fd, 0777)` instead of `0755`,
   so a differently-privileged orchestrator container can write it.

Each `sed` is verified with a `grep -q` so a drifted source line fails the
build instead of silently producing an unpatched `.so`. The final image
stage is `busybox:stable` containing exactly one file: `/vGPU-NVIDIA.so` —
the image is a delivery envelope, not a runnable service; consumers copy the
`.so` out of it (e.g. an init container `cp`).

`cloudbuild-so.yaml` is the Google Cloud Build recipe: one docker-build step
producing
`asia-southeast1-docker.pkg.dev/$PROJECT_ID/time-slicing/gpucr-so:$_TAG`
(default `_TAG=unrounded-v1`), pushed via the `images:` list, on an
`E2_HIGHCPU_32` machine with 200 GB disk and a 1-hour timeout. Because the
Docker context is the submitted source, run it from your working tree:

```bash
gcloud builds submit --config cloudbuild-so.yaml \
    --substitutions _TAG=my-feature-v1 .
```

Whatever is in your local tree — committed or not — is what gets built.
That is the point ("local-source"), and also the risk: tag deliberately.

### Release/build matrix at a glance

| Artifact | Built by | Contains patches? | Distribution |
|----------|----------|-------------------|--------------|
| `vGPU-NVIDIA.so` + `cr_client` (+ checksums) | `make release-local` in GitHub Actions on `v*` tag | No (stock SIGUSR1/2, 0755) | GitHub Release assets |
| Patched `vGPU-NVIDIA.so` | Cloud Build (`cloudbuild-so.yaml` → `Dockerfile.build`) from local source | Yes (SIGRTMAX-8/-7, 0777 control file) | Artifact Registry image `gpucr-so:$_TAG` |
| `libgcr_preload.so`, `libnccl-checkpoint-gcr.so`, examples | manual local CMake with adapter flags | n/a | not released; build per environment against your NCCL/NVSHMEM tree |

---

## Gotchas for maintainers

1. **The stock release `.so` and the cloud-built `.so` are different
   binaries.** GitHub releases signal on SIGUSR1/SIGUSR2; the Cloud Build
   image signals on SIGRTMAX-8/-7 and chmods the control file 0777. Mixing
   an orchestrator configured for one with a preload built for the other
   fails silently (signals go to the app, or nowhere).

2. **`libgcr_preload.so` vs `vGPU-NVIDIA.so` are not interchangeable.**
   Only `libgcr_preload.so` exports `gcr_checkpoint_*` (the plugin/NVSHMEM
   path); only `vGPU-NVIDIA.so` has the signal handlers and `nccl_hooks`
   (the `multi_cr_client` path). Preloading the wrong one gives
   "missing runtime symbol gcr_checkpoint_prepare" or ignored signals.

3. **Forgetting `NCCL_CUMEM_ENABLE=1` or rebuilding NCCL without
   `CUDARTLIB=cudart`** both produce the same signature: prepare logs
   `imports=0 exports=0`, and `cuda-checkpoint --action restore` later dies
   with "operation not supported". Check both when you see it.

4. **Flag values are cross-compiled ABI.** `0x0400` for VALIDATE is
   duplicated as a literal in `gcr_checkpoint_runtime.cpp`
   (`GCR_CKPT_OPT_VALIDATE`); the plugin owns the RESTORE_EXPORT/IMPORT
   dispatch. Changing any `NCCL_CKPT_*` value requires rebuilding NCCL, the
   plugin, the runtime, and the examples together.

5. **`plugin_loader.patch` edits parallel arrays.** Upstream adding a sixth
   plugin type will make it apply "successfully" with fuzz and corrupt the
   `NUM_LIBS` table alignment. After any NCCL bump, re-check
   `plugin_open.cc` by eye, then `nm -D` the built `libnccl.so` for the four
   checkpoint symbols.

6. **The GCR restore shm paths are per-rank and swapped.** Each rank writes
   its exports to its own `GCR_EXPORT_SHM_PATH` and imports from the *peer's*
   file via `GCR_IMPORT_SHM_PATH`. `restore_export` hard-fails if exports
   exist and `GCR_EXPORT_SHM_PATH` is unset; ditto for imports in
   `restore_import`. Also mind `GCR_STOP_FD_SERVER_AFTER_IMPORT` (default 0
   keeps the fd server alive for overlapped peer restore).

7. **`gcr_checkpoint_prepare` is once-per-process.** The `g_prepared` latch
   makes a second prepare a logged no-op until a successful
   `restore_import` clears it. A crashed restore leaves the process
   "prepared" with its staging buffers still allocated.

8. **The NVSHMEM run scripts are out of sync with CMake** (verified): they
   look for `build_stage2/nvshmem_cumem_checkpoint_test`, the device wrapper
   execs a non-existent `run_nvshmem_cumem_checkpoint.sh`, and the host
   script's `CUDA_CKPT` default dereferences the unset `GPU_CR_ROOT` under
   `set -u`. Treat them as templates; export `CUDA_CKPT` and fix paths
   before use.

9. **Release builds exclude `multi_cr_client` and all adapters.** If a
   deployment needs the multi-GPU orchestrator or the NCCL adapter, it must
   build from source — the GitHub release only ships `cr_client` +
   `vGPU-NVIDIA.so`.

10. **Two different CUDA base images are in play**: releases build against
    CUDA 12.2.2, the Cloud Build image against CUDA 13.0.0. `ipc_hooks.cpp`
    goes to great lengths to hook both CUDA-11-style and CUDA-12-style
    resolution symbols, but when debugging version-specific interception
    issues, know which toolkit your `.so` was compiled against.

11. **Don't apply `checkpoint_api.cc` to GCR's full internal NCCL tree** —
    it already defines `ncclCommCheckpointPrepare` in `mem_manager.cc`;
    you'll get duplicate-symbol link errors. The `nccl_patches/` files are
    for *stock upstream* NCCL only.

12. **Everything depends on default symbol visibility.** Both preload
    libraries and the plugin set `CXX_VISIBILITY_PRESET default` /
    `VISIBILITY_INLINES_HIDDEN OFF`. If a well-meaning cleanup adds
    `-fvisibility=hidden`, `dlsym(RTLD_DEFAULT, "gcr_checkpoint_*")` and the
    LD_PRELOAD interposition of `dlsym`/`cudaGetDriverEntryPoint` all break
    at runtime with no compile-time signal.
