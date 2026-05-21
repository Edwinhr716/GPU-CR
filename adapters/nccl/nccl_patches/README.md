# NCCL Upstream Patch

Three files in this directory teach NCCL core to load a third-party checkpoint
plugin. They are intentionally minimal — all IPC/VMM/data work lives in the
GPU-CR runtime, not in NCCL.

| File | Copy to (inside NCCL source tree) | Lines | Purpose |
|------|----------------------------------|------:|---------|
| `checkpoint.h` | `src/include/checkpoint.h` | 16 | Internal API used by `mem_manager.cc` to dispatch into the plugin |
| `nccl_checkpoint.h` | `src/include/plugin/nccl_checkpoint.h` | 27 | Plugin ABI struct (`ncclCheckpointPlugin_v1_t`) and the load symbol name |
| `checkpoint.cc` | `src/plugin/checkpoint.cc` | 119 | Loader; resolves the plugin via `ncclOpenCheckpointPluginLib`, calls `init/prepare/restore/finalize` |

## Apply (manual copy)

```bash
NCCL_SRC=/path/to/nccl
GPU_CR=/path/to/GPU-CR

cp ${GPU_CR}/adapters/nccl/nccl_patches/checkpoint.h        ${NCCL_SRC}/src/include/checkpoint.h
cp ${GPU_CR}/adapters/nccl/nccl_patches/nccl_checkpoint.h   ${NCCL_SRC}/src/include/plugin/nccl_checkpoint.h
cp ${GPU_CR}/adapters/nccl/nccl_patches/checkpoint.cc       ${NCCL_SRC}/src/plugin/checkpoint.cc
```

After copying, rebuild NCCL as usual (`make -j`). The new plugin loader is
called from `mem_manager.cc` when `ncclCommCheckpointPrepare`/`Restore` is
invoked with `NCCL_CKPT_MODE_GCR_GLOBAL`.

## Verify

Once built and installed, sanity-check the linker can find the plugin loader
symbols inside `libnccl.so`:

```bash
nm -D ${NCCL_SRC}/build/lib/libnccl.so | grep ncclCheckpointPlugin
# expect: ncclCheckpointPluginPrepare, ncclCheckpointPluginRestore
```

At runtime, point NCCL at the GCR plugin:

```bash
export NCCL_CHECKPOINT_PLUGIN=/path/to/GPU-CR/build/adapters/nccl/libnccl-checkpoint-gcr.so
```

NCCL will dlopen it the first time a checkpoint API is called in `GCR_GLOBAL`
mode and log `Successfully loaded external checkpoint plugin gcr` via
`NCCL_DEBUG=INFO`.

## Version Compatibility

These files target the NCCL plugin-loader conventions in upstream `master`
(post-2.28). The interface (`ncclCheckpointPlugin_v1_t`) is the same v1 ABI
the plugin in `../plugin/` consumes. If the upstream loader API changes, both
sides need to be revised together.

## NCCL_CKPT_* Flags

The mode and phase flags consumed by these APIs (`NCCL_CKPT_MODE_NATIVE`,
`NCCL_CKPT_MODE_GCR_GLOBAL`, `NCCL_CKPT_RESTORE_EXPORT`,
`NCCL_CKPT_RESTORE_IMPORT`, …) are declared in NCCL's public `nccl.h.in`. No
patch to `nccl.h` is required if you build against a recent NCCL that already
ships them; otherwise add them alongside `ncclCommCheckpointPrepare/Restore`
in `nccl.h.in` and regenerate.
