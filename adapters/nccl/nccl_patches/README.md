# NCCL Upstream Patch

Six pieces in this directory teach NCCL core a public checkpoint API plus a
loader for a third-party checkpoint plugin. They are intentionally minimal —
all IPC/VMM/data work lives in the GPU-CR runtime, not in NCCL.

| File | Apply to (inside NCCL source tree) | Purpose |
|------|------------------------------------|---------|
| `nccl.h.in.checkpoint-additions` | **paste into** `src/nccl.h.in` | `NCCL_CKPT_*` flag macros + `ncclCommCheckpointPrepare/Restore` declarations |
| `checkpoint_api.cc` | `src/checkpoint_api.cc` (**+ add to Makefile**) | The two public API entry points; dispatch into the plugin loader |
| `checkpoint.h` | `src/include/checkpoint.h` | Internal API used to dispatch into the plugin |
| `nccl_checkpoint.h` | `src/include/plugin/nccl_checkpoint.h` | Plugin ABI struct (`ncclCheckpointPlugin_v1_t`) and the load symbol name |
| `checkpoint.cc` | `src/plugin/checkpoint.cc` | Loader; resolves the plugin from `$NCCL_CHECKPOINT_PLUGIN`, calls `init/prepare/restore/finalize` |
| `plugin_loader.patch` | `git apply` from NCCL source root | Registers the "checkpoint" plugin type with NCCL's generic loader (`plugin.h` + `plugin_open.cc`) |

## Apply

```bash
NCCL_SRC=/path/to/nccl          # stock upstream tree, v2.28+ (validated on v2.30.x)
GPU_CR=/path/to/GPU-CR
P=${GPU_CR}/adapters/nccl/nccl_patches

# 1. Four copies
cp ${P}/checkpoint.h       ${NCCL_SRC}/src/include/checkpoint.h
cp ${P}/nccl_checkpoint.h  ${NCCL_SRC}/src/include/plugin/nccl_checkpoint.h
cp ${P}/checkpoint.cc      ${NCCL_SRC}/src/plugin/checkpoint.cc
cp ${P}/checkpoint_api.cc  ${NCCL_SRC}/src/checkpoint_api.cc

# 2. Register the checkpoint plugin type with the generic loader:
cd ${NCCL_SRC} && git apply ${P}/plugin_loader.patch

# 3. Paste the contents of nccl.h.in.checkpoint-additions into
#    ${NCCL_SRC}/src/nccl.h.in — anywhere in the function-declaration area
#    (suggested: right after the ncclCommSuspend/pncclCommSuspend block).

# 4. Add checkpoint_api.cc to the library source list in
#    ${NCCL_SRC}/src/Makefile (the LIBSRCFILES list that already contains
#    init.cc, proxy.cc, ..., mem_manager.cc).

# 5. Rebuild NCCL — CUDARTLIB=cudart is REQUIRED (see below):
cd ${NCCL_SRC} && make -j src.build CUDARTLIB=cudart
```

> **Why `CUDARTLIB=cudart` is mandatory:** NCCL's default build links the
> CUDA runtime **statically** (`cudart_static`). NCCL resolves every CUDA
> driver function (including `cuMemCreate/cuMemExportToShareableHandle/...`)
> through `cudaGetDriverEntryPoint`, and with a static cudart that call is
> internal to `libnccl.so` — it never goes through the PLT, so GPU-CR's
> `LD_PRELOAD` interception cannot see it. Symptom: GCR logs
> `imports=0 exports=0` at prepare (nothing tracked) even with
> `NCCL_CUMEM_ENABLE=1`, and `cuda-checkpoint --action restore` later fails
> with "operation not supported". Linking cudart dynamically
> (`CUDARTLIB=cudart`) routes the resolution through the PLT where the
> preload library interposes it.

> Note: GCR's full internal NCCL tree already contains all of this (the API
> entry points live inside `mem_manager.cc` there, with an additional NATIVE
> suspend/resume backend). The files here are the standalone version for
> stock upstream NCCL. Don't apply `checkpoint_api.cc` to a tree that already
> defines `ncclCommCheckpointPrepare` — you'd get duplicate symbols.

## NCCL_CKPT_* canonical flag values

Declared in `nccl.h.in.checkpoint-additions`; the GPU-CR plugin and examples
are compiled against these exact values, so do not change them:

| Macro | Value | Meaning |
|-------|-------|---------|
| `NCCL_CKPT_MODE_NATIVE` | `0x0001` | NCCL native suspend/resume (needs `ncclCommSuspend`, upstream NCCL ≥ 2.30; older NCCL returns `ncclInvalidUsage`) |
| `NCCL_CKPT_MODE_GCR_GLOBAL` | `0x0004` | Dispatch to the external checkpoint plugin |
| `NCCL_CKPT_OPT_SAVE_DATA` | `0x0100` | Save GPU data before teardown |
| `NCCL_CKPT_OPT_NO_CUDA_CKPT` | `0x0200` | Do not invoke process-level checkpoint |
| `NCCL_CKPT_OPT_VALIDATE` | `0x0400` | Validate mappings after restore |
| `NCCL_CKPT_RESTORE_EXPORT` | `0x1000` | Restore phase 1: rebuild local exports, publish new handles |
| `NCCL_CKPT_RESTORE_IMPORT` | `0x2000` | Restore phase 2: re-import peer handles after fd exchange |

Flag semantics: `prepare` is a single action — the flags select the mode and
are passed through to the plugin/runtime for logging; the GCR runtime does
not branch on them at prepare time. The `RESTORE_EXPORT`/`RESTORE_IMPORT`
bits are consumed by the **plugin** (`gcrPluginRestore`), which dispatches to
the runtime's two separate entry points (`gcr_checkpoint_restore_export` /
`gcr_checkpoint_restore_import`). The `OPT_*` bits are reserved for future
use and currently validated-but-ignored.

## Verify

Once built, sanity-check the public API and the plugin loader made it into
`libnccl.so`:

```bash
nm -D ${NCCL_SRC}/build/lib/libnccl.so | grep -E "ncclCommCheckpoint|ncclCheckpointPlugin"
# expect: ncclCommCheckpointPrepare, ncclCommCheckpointRestore,
#         ncclCheckpointPluginPrepare, ncclCheckpointPluginRestore
```

At runtime, point NCCL at the GCR plugin:

```bash
export NCCL_CHECKPOINT_PLUGIN=/path/to/GPU-CR/build/adapters/nccl/libnccl-checkpoint-gcr.so
```

NCCL will dlopen it the first time a checkpoint API is called in `GCR_GLOBAL`
mode and log `Successfully loaded external checkpoint plugin gcr` via
`NCCL_DEBUG=INFO`.

**Important:** also export the cuMem environment before starting your
application, otherwise NCCL shares its buffers through legacy CUDA IPC,
which GPU-CR cannot tear down and `cuda-checkpoint` cannot restore
(typical symptom: GCR logs `imports=0 exports=0` and
`cuda-checkpoint --action restore` fails with "operation not supported"):

```bash
export NCCL_CUMEM_ENABLE=1        # route NCCL IPC through cuMem* (GPU-CR intercepts these)
```

vLLM-style deployments additionally use `NCCL_CUMEM_HOST_ENABLE=1` and
`NCCL_P2P_DISABLE=1` (see `apps/vllm/launch_multi_gpu.sh`); do **not**
disable P2P for the bundled `two_proc_nccl_test` — its cuMem
exports/imports come from the P2P transport.

## Version Compatibility

These files target the NCCL plugin-loader conventions in upstream `master`
(post-2.28); validated against v2.30.x. The interface
(`ncclCheckpointPlugin_v1_t`) is the same v1 ABI the plugin in `../plugin/`
consumes. If the upstream loader API changes, both sides need to be revised
together.
