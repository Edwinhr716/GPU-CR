#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_NAME="${BENCH_NAME:-nvshmem_cumem_checkpoint_device_test}"

exec "${SCRIPT_DIR}/run_nvshmem_cumem_checkpoint.sh" "$@"
