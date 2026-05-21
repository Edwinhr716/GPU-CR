# NVSHMEM Checkpoint Examples

Unlike the NCCL adapter, **NVSHMEM needs no patch**. GPU-CR's
`src/ipc_hooks.cpp` already intercepts the cuMem driver entry points NVSHMEM
uses (via `dlsym` + `cuGetProcAddress_v2`), so as long as `vGPU-NVIDIA.so` or
`libgcr_preload.so` is preloaded into the application, the IPC/VMM teardown
and rebuild already covers NVSHMEM allocations.

This directory therefore ships only demonstrator binaries and the helper run
scripts.

## Layout

| Path | Purpose |
|------|---------|
| `examples/nvshmem_host_test.cpp` | Host-side allocate/checkpoint/restore loop |
| `examples/nvshmem_device_test.cu` | Same flow, but with a CUDA device kernel that touches the symmetric heap |
| `examples/run_nvshmem_host_test.sh` | Driver wrapping `LD_PRELOAD` + `nvshmrun` |
| `examples/run_nvshmem_device_test.sh` | Driver for the device-kernel example |

## Build

The examples are an opt-in target of the top-level GPU-CR build:

```bash
mkdir build && cd build
cmake -DGPU_VENDOR=NVIDIA \
      -DGPU_CR_BUILD_NVSHMEM_ADAPTER=ON \
      -DNVSHMEM_ROOT=/path/to/nvshmem \
      ..
make -j
```

Outputs (under `build/adapters/nvshmem/`):
- `nvshmem_host_test`
- `nvshmem_device_test`

If `libnvshmem_host` or `libnvshmem_device` is not under
`${NVSHMEM_ROOT}/{lib,build/lib}`, the corresponding example is skipped with
a status line. Override with `-DNVSHMEM_HOST_LIBRARY=...` /
`-DNVSHMEM_DEVICE_LIBRARY=...` if your install puts them elsewhere.

## Run

The two run scripts cover the standard usage: preload either `vGPU-NVIDIA.so`
(signal-driven control via `multi_cr_client`) or `libgcr_preload.so` (NCCL-
plugin-style control via the adapter), then launch the example through
NVSHMEM's `nvshmrun`. Tweak `GCR_STORAGE_BACKEND`, `GCR_STORAGE_DIR`, and
`CUDA_CKPT` env vars in those scripts to match your environment.

## Notes

- NVIDIA only. NVSHMEM has no AMD counterpart in this repo.
- The examples are diagnostic, not a benchmark. Use them to confirm that an
  NVSHMEM application can be successfully checkpointed and restored end-to-
  end on your driver/CUDA version before integrating GPU-CR into a larger
  NVSHMEM workload.
