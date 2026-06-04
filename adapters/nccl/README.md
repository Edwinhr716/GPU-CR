# NCCL Checkpoint Adapter

This adapter lets a NCCL-using application (e.g. multi-GPU vLLM) be
checkpoint/restored by GPU-CR through NCCL's optional checkpoint plugin ABI.

## Architecture

```
  NCCL core (with our 2-file patch applied)
    ├── ncclCommCheckpointPrepare(comm, flags)
    └── ncclCommCheckpointRestore(comm, flags)
              │
              │  if flags & NCCL_CKPT_MODE_GCR_GLOBAL → dlopen plugin
              ▼
  libnccl-checkpoint-gcr.so  (plugin shim, this directory's plugin/)
              │
              │  dlsym(RTLD_DEFAULT, "gcr_checkpoint_*")
              ▼
  libgcr_preload.so          (runtime + IPC tracking, this directory's runtime/)
```

Two shared libraries are built here. The runtime is loaded into the worker
process via `LD_PRELOAD`; the plugin is loaded inside NCCL via
`NCCL_CHECKPOINT_PLUGIN`. The plugin discovers the runtime by `dlsym` on the
default symbol scope — no link-time dependency between them.

## Layout

| Path | Purpose |
|------|---------|
| `runtime/gcr_checkpoint_runtime.cpp` | Implements `gcr_checkpoint_prepare/restore_export/restore_import/validate` |
| `include/gcr_checkpoint.h` | Public C ABI consumed by the plugin |
| `plugin/nccl_checkpoint_gcr_plugin.cpp` | NCCL plugin entry point; bridges to the runtime via `dlsym` |
| `nccl_patches/` | The 5 files you apply to an NCCL source tree: public API + flag macros + plugin loader (see `nccl_patches/README.md`) |
| `examples/two_proc_nccl_test.cpp` | Two-process AllReduce demonstrator with checkpoint/restore |

## Build

The adapter is an opt-in target of the top-level GPU-CR build:

```bash
mkdir build && cd build
cmake -DGPU_VENDOR=NVIDIA \
      -DGPU_CR_BUILD_NCCL_ADAPTER=ON \
      -DNCCL_ROOT=/path/to/nccl \
      ..
make -j
```

`NCCL_ROOT` must point at an NCCL source tree (ideally already built). The
CMake auto-discovers `nccl.h` under `${NCCL_ROOT}/{include,build/include,
nccl_install_stage{1,2}/include}` and `nccl_common.h` under
`${NCCL_ROOT}/src/include`. Override with `-DNCCL_PUBLIC_INCLUDE_DIR=...` or
`-DNCCL_INTERNAL_INCLUDE_DIR=...` if your layout differs.

The example binary is built only if `libnccl.so` is found; otherwise CMake
prints a status line and skips it.

## Use

1. Apply the patches from `nccl_patches/` to your NCCL source tree and rebuild
   NCCL **with `CUDARTLIB=cudart`** (see `nccl_patches/README.md` — NCCL's
   default static cudart bypasses GPU-CR's LD_PRELOAD interception entirely):

   ```bash
   make -j src.build CUDARTLIB=cudart
   ```
2. Run your NCCL application with both shared objects on the right load paths
   **and the cuMem environment enabled**:

```bash
export LD_PRELOAD=/path/to/GPU-CR/build/adapters/nccl/libgcr_preload.so
export NCCL_CHECKPOINT_PLUGIN=/path/to/GPU-CR/build/adapters/nccl/libnccl-checkpoint-gcr.so

# REQUIRED: make NCCL share its buffers through the cuMem* APIs that GPU-CR
# intercepts. Without this, NCCL uses legacy CUDA IPC (cudaIpcGetMemHandle):
# GPU-CR tracks nothing (logs `imports=0 exports=0`) and cuda-checkpoint
# fails to restore ("operation not supported").
export NCCL_CUMEM_ENABLE=1

python my_nccl_app.py
```

For vLLM-style deployments two more knobs are used (see
`apps/vllm/launch_multi_gpu.sh`): `NCCL_CUMEM_HOST_ENABLE=1` routes
host-staged transport buffers through cuMem too, and `NCCL_P2P_DISABLE=1`
avoids direct P2P mappings in that setup. Do **not** set
`NCCL_P2P_DISABLE=1` for the `two_proc_nccl_test` example — the P2P
transport is exactly what produces the cuMem exports/imports the test
exercises (GPU-CR disables peer access itself before the process-level
checkpoint).

3. Inside the application, drive checkpoint/restore through the NCCL API:

```c
ncclCommCheckpointPrepare(comm, NCCL_CKPT_MODE_GCR_GLOBAL);
// ... cuda-checkpoint freeze, swap, unfreeze ...
ncclCommCheckpointRestore(comm, NCCL_CKPT_MODE_GCR_GLOBAL | NCCL_CKPT_RESTORE_EXPORT);
ncclCommCheckpointRestore(comm, NCCL_CKPT_MODE_GCR_GLOBAL | NCCL_CKPT_RESTORE_IMPORT);
```

`examples/two_proc_nccl_test.cpp` shows the full sequence end-to-end.

## Notes

- NVIDIA only — NCCL itself is CUDA-specific. AMD builds skip this adapter.
- The runtime in `libgcr_preload.so` is independent from the `vGPU-NVIDIA.so`
  produced by the top-level build. Pick whichever LD_PRELOAD matches your
  control path: `vGPU-NVIDIA.so` for signal-driven `multi_cr_client`,
  `libgcr_preload.so` for NCCL-plugin-driven checkpoint/restore.
