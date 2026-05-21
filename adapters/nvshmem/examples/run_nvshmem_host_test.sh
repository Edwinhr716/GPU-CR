#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${GCR_ROOT}/build_stage2"
BENCH_NAME="${BENCH_NAME:-nvshmem_cumem_checkpoint_test}"
BENCH="${BUILD_DIR}/${BENCH_NAME}"
PRELOAD="${BUILD_DIR}/libgcr_preload.so"

NVSHMEM_ROOT="${NVSHMEM_ROOT:-/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem}"
NVSHMEM_LIB_DIR="${NVSHMEM_ROOT}/lib"
CUDA_CKPT="${CUDA_CKPT:-${GPU_CR_ROOT}/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint}"
CUDA_CKPT_MODE="${CUDA_CKPT_MODE:-action}"

LOG_ROOT="${LOG_ROOT:-./log}"
OUT_DIR="${OUT_DIR:-${LOG_ROOT}/nvshmem_cumem_checkpoint_$(date +%Y%m%d_%H%M%S)}"
BYTES="${BYTES:-67108864}"
GCR_STORAGE_BACKEND="${GCR_STORAGE_BACKEND:-hugepage}"
GCR_STORAGE_DIR="${GCR_STORAGE_DIR:-/mnt/huge-ckpt/nvshmem-gcr}"
ALLOW_EXISTING_GPU_PROCS="${ALLOW_EXISTING_GPU_PROCS:-0}"

mkdir -p "${OUT_DIR}"

for f in "${BENCH}" "${PRELOAD}" "${NVSHMEM_LIB_DIR}/libnvshmem_host.so.3" "${CUDA_CKPT}"; do
  if [[ ! -e "${f}" ]]; then
    echo "ERROR: missing required file: ${f}" >&2
    exit 1
  fi
done

if [[ "${GCR_STORAGE_BACKEND}" == "hugepage" ]] && ! mount | grep -q ' /mnt/huge-ckpt '; then
  echo "ERROR: /mnt/huge-ckpt is not mounted" >&2
  exit 1
fi

existing_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -u | tr '\n' ',' | sed 's/,$//')"
if [[ -n "${existing_gpu_pids}" && "${ALLOW_EXISTING_GPU_PROCS}" != "1" ]]; then
  echo "ERROR: existing GPU compute processes detected: ${existing_gpu_pids}" >&2
  echo "Set ALLOW_EXISTING_GPU_PROCS=1 only if these processes are safe to ignore." >&2
  exit 1
fi

CONTROL_DIR="${OUT_DIR}/control"
mkdir -p "${CONTROL_DIR}"
UNIQUE_ID_FILE="${CONTROL_DIR}/nvshmem_unique_id.bin"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

echo "bytes,status,rank0_pid,rank1_pid,gpu_before_mb,gpu_after_prepare_mb,gpu_after_ckpt_mb,gpu_after_restore_mb,gpu_after_gcr_restore_mb,ckpt_ms,restore_ms,total_cuda_ckpt_ms,out_dir" > "${SUMMARY_CSV}"

RANK_PIDS_FOR_CLEANUP=""
cleanup_case() {
  set +e
  for pid in ${RANK_PIDS_FOR_CLEANUP:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null
    fi
  done
  sleep 2
  for pid in ${RANK_PIDS_FOR_CLEANUP:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null
    fi
  done
  RANK_PIDS_FOR_CLEANUP=""
  set -e
}
trap cleanup_case EXIT

gpu_used_sum_mb() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {printf "%.3f", sum + 0}'
}

wait_file_or_fail() {
  local app_pid="$1"
  local file="$2"
  while [[ ! -f "${file}" ]]; do
    if ! kill -0 "${app_pid}" 2>/dev/null; then
      return 1
    fi
    sleep 0.2
  done
}

cuda_checkpoint_all() {
  local phase="$1"
  shift
  local pid
  for pid in "$@"; do
    echo "[CUDA-CKPT] ${phase} pid=${pid} mode=${CUDA_CKPT_MODE}"
    if [[ "${CUDA_CKPT_MODE}" == "action" ]]; then
      if [[ "${phase}" == "checkpoint" ]]; then
        "${CUDA_CKPT}" --action lock --pid "${pid}"
        "${CUDA_CKPT}" --action checkpoint --pid "${pid}"
      elif [[ "${phase}" == "restore" ]]; then
        "${CUDA_CKPT}" --action restore --pid "${pid}"
        "${CUDA_CKPT}" --action unlock --pid "${pid}"
      else
        echo "ERROR: unsupported cuda-checkpoint phase ${phase}" >&2
        return 2
      fi
    else
      "${CUDA_CKPT}" --toggle --pid "${pid}"
    fi
  done
}

export LD_LIBRARY_PATH="${NVSHMEM_LIB_DIR}:${BUILD_DIR}:${LD_LIBRARY_PATH:-}"
export NVSHMEM_BOOTSTRAP="${NVSHMEM_BOOTSTRAP:-UID}"
export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-0}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
export NVSHMEM_DISABLE_NCCL="${NVSHMEM_DISABLE_NCCL:-1}"
export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-WARN}"
export GCR_IPC_SCALING_METRICS_CSV="${OUT_DIR}/gcr_runtime_metrics.csv"
export GCR_STORAGE_BACKEND
export GCR_STORAGE_DIR
export GCR_STOP_FD_SERVER_AFTER_IMPORT="${GCR_STOP_FD_SERVER_AFTER_IMPORT:-0}"

echo "[RUN] NVSHMEM cuMem checkpoint bench=${BENCH_NAME} bytes=${BYTES}"

env \
  LD_PRELOAD="${PRELOAD}" \
  GCR_IPC_SCALING_MODE="nvshmem-cumem-checkpoint" \
  GCR_IPC_SCALING_RANK="0" \
  GCR_IPC_SCALING_CASE="${BYTES}" \
  GCR_IPC_SCALING_BUFFER_GB="0" \
  GCR_EXPORT_SHM_PATH="${CONTROL_DIR}/rank0_exports.shm" \
  GCR_IMPORT_SHM_PATH="${CONTROL_DIR}/rank1_exports.shm" \
  "${BENCH}" --rank 0 --nranks 2 --bytes "${BYTES}" \
    --control-dir "${CONTROL_DIR}" --unique-id-file "${UNIQUE_ID_FILE}" \
    > "${OUT_DIR}/rank0.log" 2>&1 &
RANK0_PID=$!

env \
  LD_PRELOAD="${PRELOAD}" \
  GCR_IPC_SCALING_MODE="nvshmem-cumem-checkpoint" \
  GCR_IPC_SCALING_RANK="1" \
  GCR_IPC_SCALING_CASE="${BYTES}" \
  GCR_IPC_SCALING_BUFFER_GB="0" \
  GCR_EXPORT_SHM_PATH="${CONTROL_DIR}/rank1_exports.shm" \
  GCR_IMPORT_SHM_PATH="${CONTROL_DIR}/rank0_exports.shm" \
  "${BENCH}" --rank 1 --nranks 2 --bytes "${BYTES}" \
    --control-dir "${CONTROL_DIR}" --unique-id-file "${UNIQUE_ID_FILE}" \
    > "${OUT_DIR}/rank1.log" 2>&1 &
RANK1_PID=$!

RANK_PIDS_FOR_CLEANUP="${RANK0_PID} ${RANK1_PID}"
echo "${RANK0_PID},${RANK1_PID}" > "${OUT_DIR}/rank_pids.txt"

if ! wait_file_or_fail "${RANK0_PID}" "${CONTROL_DIR}/rank0.ready"; then
  echo "ERROR: rank0 exited before ready; see ${OUT_DIR}/rank0.log" >&2
  exit 1
fi
if ! wait_file_or_fail "${RANK1_PID}" "${CONTROL_DIR}/rank1.ready"; then
  echo "ERROR: rank1 exited before ready; see ${OUT_DIR}/rank1.log" >&2
  exit 1
fi

GPU_BEFORE="$(gpu_used_sum_mb)"
touch "${CONTROL_DIR}/prepare"

if ! wait_file_or_fail "${RANK0_PID}" "${CONTROL_DIR}/rank0.prepared"; then
  echo "ERROR: rank0 exited before prepared; see ${OUT_DIR}/rank0.log" >&2
  exit 1
fi
if ! wait_file_or_fail "${RANK1_PID}" "${CONTROL_DIR}/rank1.prepared"; then
  echo "ERROR: rank1 exited before prepared; see ${OUT_DIR}/rank1.log" >&2
  exit 1
fi

GPU_AFTER_PREPARE="$(gpu_used_sum_mb)"
nvidia-smi > "${OUT_DIR}/nvidia_smi_after_prepare.txt"

CKPT_START="$(date +%s%N)"
cuda_checkpoint_all "checkpoint" "${RANK0_PID}" "${RANK1_PID}" > "${OUT_DIR}/cuda_checkpoint.log" 2>&1
CKPT_END="$(date +%s%N)"
GPU_AFTER_CKPT="$(gpu_used_sum_mb)"
nvidia-smi > "${OUT_DIR}/nvidia_smi_after_ckpt.txt"

RESTORE_START="$(date +%s%N)"
cuda_checkpoint_all "restore" "${RANK0_PID}" "${RANK1_PID}" > "${OUT_DIR}/cuda_restore.log" 2>&1
RESTORE_END="$(date +%s%N)"
GPU_AFTER_RESTORE="$(gpu_used_sum_mb)"

touch "${CONTROL_DIR}/restore_export"
if ! wait_file_or_fail "${RANK0_PID}" "${CONTROL_DIR}/rank0.restore_export_done"; then
  echo "ERROR: rank0 exited before restore_export_done; see ${OUT_DIR}/rank0.log" >&2
  exit 1
fi
if ! wait_file_or_fail "${RANK1_PID}" "${CONTROL_DIR}/rank1.restore_export_done"; then
  echo "ERROR: rank1 exited before restore_export_done; see ${OUT_DIR}/rank1.log" >&2
  exit 1
fi

touch "${CONTROL_DIR}/restore_import"
if ! wait_file_or_fail "${RANK0_PID}" "${CONTROL_DIR}/rank0.restore_import_done"; then
  echo "ERROR: rank0 exited before restore_import_done; see ${OUT_DIR}/rank0.log" >&2
  exit 1
fi
if ! wait_file_or_fail "${RANK1_PID}" "${CONTROL_DIR}/rank1.restore_import_done"; then
  echo "ERROR: rank1 exited before restore_import_done; see ${OUT_DIR}/rank1.log" >&2
  exit 1
fi

GPU_AFTER_GCR_RESTORE="$(gpu_used_sum_mb)"

STATUS=0
wait "${RANK0_PID}" || STATUS=$?
wait "${RANK1_PID}" || STATUS=$?
RANK_PIDS_FOR_CLEANUP=""

CKPT_MS="$(( (CKPT_END - CKPT_START) / 1000000 ))"
RESTORE_MS="$(( (RESTORE_END - RESTORE_START) / 1000000 ))"
TOTAL_MS="$(( CKPT_MS + RESTORE_MS ))"
echo "${BYTES},${STATUS},${RANK0_PID},${RANK1_PID},${GPU_BEFORE},${GPU_AFTER_PREPARE},${GPU_AFTER_CKPT},${GPU_AFTER_RESTORE},${GPU_AFTER_GCR_RESTORE},${CKPT_MS},${RESTORE_MS},${TOTAL_MS},${OUT_DIR}" >> "${SUMMARY_CSV}"

if [[ "${STATUS}" != "0" ]]; then
  echo "ERROR: benchmark failed; see ${OUT_DIR}/rank*.log" >&2
  exit "${STATUS}"
fi

echo "[DONE] results: ${OUT_DIR}"
