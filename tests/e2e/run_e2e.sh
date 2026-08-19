#!/usr/bin/env bash
# End-to-end test for the consolidated GPU-CR stack on a GPU node.
#
# Drives a real CUDA workload under LD_PRELOAD through the full
# checkpoint/restore surface, gating on byte-identical GPU memory after
# every restore:
#   G1  baseline pattern verify
#   G2  destination-path selective checkpoint (-o) succeeds
#   G3  destination-path selective restore succeeds
#   G4  pattern verify after dest-path restore (byte-identical)
#   G5  buffer-path selective checkpoint/restore + verify
#   G6  full checkpoint/restore data plane + verify (stubbed toggle unless
#       E2E_FULL_TOGGLE=1 and a real cuda-checkpoint is on PATH)
#
# Required env: CR_CLIENT, WORKLOAD, VGPU_SO, STORE.
# Optional env: GPU_CR_CTL_PATH, E2E_NUM_BUFFERS, E2E_BUFFER_MB,
#               GPU_CR_SHM_MB (defaults to a size fitting the buffers),
#               CR_TIMEOUT (seconds per cr_client call, default 120).
set -u
HERE=$(dirname "$(readlink -f "$0")")
. "$HERE/e2e_lib.sh"

: "${CR_CLIENT:?}" "${WORKLOAD:?}" "${VGPU_SO:?}" "${STORE:?}"
NUM_BUFFERS=${E2E_NUM_BUFFERS:-4}
BUFFER_MB=${E2E_BUFFER_MB:-64}
# Dump buffer: extents + 2MiB header + slack.
SHM_MB=${GPU_CR_SHM_MB:-$((NUM_BUFFERS * BUFFER_MB + 128))}

PASS=0; FAIL=0
gate() {
    local name=$1; shift
    if "$@"; then echo "PASS: $name"; PASS=$((PASS+1));
    else echo "FAIL: $name"; FAIL=$((FAIL+1)); fi
}
cr() { env EXPORT_FILE_PATH="$STORE" \
          ${GPU_CR_CTL_PATH:+GPU_CR_CTL_PATH="$GPU_CR_CTL_PATH"} \
          timeout "${CR_TIMEOUT:-120}" "$CR_CLIENT" "$@"; }

trap 'stop_workload' EXIT
if [ "${E2E_FULL_TOGGLE:-0}" != "1" ]; then stub_cuda_checkpoint; fi

start_workload "$VGPU_SO" \
    E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
    GPU_CR_SHM_MB="$SHM_MB" || exit 1
echo "workload up: pid=$WL_PID regions=$WL_REGIONS (run dir $RUN)"

cr -i -p "$WL_PID" || { echo "FATAL: init failed ($?)"; exit 1; }

gate "G1 baseline verify" wl_cmd verify

DUMP="$STORE/e2e-dump.bin"
rm -f "$DUMP"
gate "G2 dest-path selective ckpt" cr -c -p "$WL_PID" -s "$WL_REGIONS" -o "$DUMP"
gate "G3 dest-path selective restore" cr -r -p "$WL_PID" -s "$WL_REGIONS" -o "$DUMP"
gate "G4 verify after dest-path restore" wl_cmd verify

gate "G5a buffer-path selective ckpt" cr -c -p "$WL_PID" -s "$WL_REGIONS"
gate "G5b buffer-path selective restore" cr -r -p "$WL_PID" -s "$WL_REGIONS"
gate "G5c verify after buffer-path restore" wl_cmd verify

gate "G6a full ckpt" cr -c -p "$WL_PID"
gate "G6b full restore" cr -r -p "$WL_PID"
gate "G6c verify after full restore" wl_cmd verify

stop_workload
echo
echo "=== e2e summary: $PASS passed, $FAIL failed ==="
echo "workload stderr: $RUN/workload.stderr"
[ "$FAIL" -eq 0 ]
