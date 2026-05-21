#!/bin/bash
# ==============================================================================
# launch_multi_gpu.sh — Launch any multi-GPU application with GPU-CR support
#
# Sets the LD_PRELOAD / NCCL / VLLM env required for GPU-CR's multi-GPU
# checkpoint+restore path, then exec's your command.
#
# Usage:
#   ./launch_multi_gpu.sh <command> [args...]
#
# Examples:
#   ./launch_multi_gpu.sh python -m vllm.entrypoints.openai.api_server \
#       --model /path/to/model --tensor-parallel-size 4 --enforce-eager
#   ./launch_multi_gpu.sh torchrun --nproc_per_node=4 train.py
#
# After launch:
#   PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd,)
#   ./build/multi_cr_client -i -p $PIDS    # one-time after model load
#   ./build/multi_cr_client -c -p $PIDS    # checkpoint
#   ./build/multi_cr_client -r -p $PIDS    # restore
# ==============================================================================

set -e

# --- CUDA toolkit (env-respect; falls back to /usr/local/cuda symlink) ---
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# --- Locate the vGPU shared library (built by the top-level cmake) ---
VGPU_LIB=""
for candidate in "${BUILD_DIR}/vGPU-NVIDIA.so" "${BUILD_DIR}/vGPU-AMD.so"; do
    if [ -f "$candidate" ]; then
        VGPU_LIB="$candidate"
        break
    fi
done
if [ -z "$VGPU_LIB" ]; then
    echo "ERROR: vGPU shared library not found in ${BUILD_DIR}/"
    echo "       Build first:  cd build && cmake -DGPU_VENDOR=NVIDIA .. && make -j"
    exit 1
fi

# --- Hugepage staging dir (only warn; user is expected to mount it) ---
if [ ! -d /mnt/huge-ckpt ]; then
    echo "WARNING: /mnt/huge-ckpt does not exist."
    echo "  Pre-mount with:  sudo mount -t hugetlbfs -o pagesize=2M,size=60G nodev /mnt/huge-ckpt"
fi
# Reset cross-process ID allocator if a stale file exists.
if [ -f /mnt/huge-ckpt/control ]; then
    dd if=/dev/zero of=/mnt/huge-ckpt/control bs=4 count=1 conv=notrunc 2>/dev/null || true
fi

# --- env ---
export GPU_VENDOR=NVIDIA
export NCCL_CUMEM_ENABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_HOST_ENABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# vLLM custom allreduce uses cudaIpcGetMemHandle, which doesn't support our
# VMM-backed pointers. Force the NCCL allreduce fallback.
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_DISABLED_KERNELS=custom_all_reduce

# --- LD_PRELOAD: vGPU.so FIRST, then local NCCL ---
if [ -n "${LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="${VGPU_LIB}:${LD_PRELOAD}"
else
    export LD_PRELOAD="${VGPU_LIB}"
fi

NCCL_LOCAL_LIB="${PROJECT_ROOT}/nccl_install/lib"
NCCL_SO_FILE=""
if [ -d "$NCCL_LOCAL_LIB" ]; then
    NCCL_SO_FILE=$(ls "${NCCL_LOCAL_LIB}"/libnccl.so.2* 2>/dev/null | head -1 || true)
    if [ -n "$NCCL_SO_FILE" ]; then
        export LD_PRELOAD="${LD_PRELOAD}:${NCCL_SO_FILE}"
        export CR_NCCL_LIB="$NCCL_SO_FILE"
        export LD_LIBRARY_PATH="${NCCL_LOCAL_LIB}:${LD_LIBRARY_PATH}"
    fi
fi

# --- Banner ---
echo "========================================"
echo " Multi-GPU CR Launcher"
echo "========================================"
echo " vGPU library : ${VGPU_LIB}"
[ -n "$NCCL_SO_FILE" ] && echo " Local NCCL  : ${NCCL_SO_FILE}" \
                       || echo " Local NCCL  : (not found at ${NCCL_LOCAL_LIB}; PyTorch's bundled NCCL will be used — ncclCommSuspend may be unavailable)"
echo " Command     : $@"
echo "========================================"

exec "$@"
