#!/usr/bin/env bash
# Integration tests for cr_client against the GPU-free fake workload.
#
# Exercises the real control-channel protocol end to end on Linux without
# a GPU: advertisement gating, destination-path checkpoint/restore, dump
# validation, op_status propagation (including the cuda-checkpoint toggle
# gate), timeouts, and version-skew refusals. Asserts the documented exit
# codes: 0 OK, 1 usage, 2 op failed, 3 refused, 4 timeout.
#
# Usage: cr_client_integration_test.sh <cr_client> <fake_workload>
set -u

CR_CLIENT=$(readlink -f "$1")
FAKE=$(readlink -f "$2")

WORK=$(mktemp -d /tmp/gpu-cr-it-data.XXXXXX)
CTL=$(mktemp -d /dev/shm/gpu-cr-it-ctl.XXXXXX)
STUB_DIR=$(mktemp -d /tmp/gpu-cr-it-stub.XXXXXX)
TOGGLE_MARKER="$STUB_DIR/toggled"
cat > "$STUB_DIR/cuda-checkpoint" <<EOF
#!/bin/sh
touch "$TOGGLE_MARKER"
exit 0
EOF
chmod +x "$STUB_DIR/cuda-checkpoint"
export PATH="$STUB_DIR:$PATH"
export GPU_CR_CUDA_CHECKPOINT="$STUB_DIR/cuda-checkpoint"
export GPU_CR_OP_TIMEOUT_SEC=10

FAKE_PID=""
PASS=0
FAIL=0

cleanup() {
    stop_fake
    rm -rf "$WORK" "$CTL" "$STUB_DIR"
}
trap cleanup EXIT

# start_fake [ENV=VAL ...] — starts fake_workload with the given extra env
# and waits for its control channel to come up.
start_fake() {
    stop_fake
    local ready="$WORK/ready.$$"
    rm -f "$ready"
    env "$@" FAKE_READY_FILE="$ready" FAKE_LOG="$WORK/fake.log" \
        "$FAKE" 2>> "$WORK/fake.stderr" &
    FAKE_PID=$!
    for _ in $(seq 1 100); do
        [ -f "$ready" ] && return 0
        sleep 0.1
    done
    echo "FATAL: fake workload did not come up" >&2
    exit 1
}

stop_fake() {
    if [ -n "$FAKE_PID" ]; then
        kill "$FAKE_PID" 2>/dev/null
        wait "$FAKE_PID" 2>/dev/null
        FAKE_PID=""
    fi
    rm -f "$CTL"/* "$WORK"/control* "$WORK"/dump* "$WORK"/fake.log \
          "$TOGGLE_MARKER" 2>/dev/null
    return 0
}

# check <name> <expected_exit> <cmd...>
check() {
    local name=$1 expected=$2
    shift 2
    "$@" > "$WORK/out.log" 2>&1
    local got=$?
    if [ "$got" -eq "$expected" ]; then
        echo "ok:   $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name (exit $got, expected $expected)"
        sed 's/^/      | /' "$WORK/out.log" | tail -15
        FAIL=$((FAIL + 1))
    fi
}

expect_file() {
    if [ -e "$2" ]; then echo "ok:   $1"; PASS=$((PASS + 1));
    else echo "FAIL: $1 (missing $2)"; FAIL=$((FAIL + 1)); fi
}

expect_no_file() {
    if [ ! -e "$2" ]; then echo "ok:   $1"; PASS=$((PASS + 1));
    else echo "FAIL: $1 (unexpected $2)"; FAIL=$((FAIL + 1)); fi
}

CTL_ENV="EXPORT_FILE_PATH=$WORK GPU_CR_CTL_PATH=$CTL"
REGIONS="0x7f0000000000:4096,0x7f0000100000:8192"

# --- ctl mode: happy paths -------------------------------------------------
start_fake $CTL_ENV
check "ctl: init"                    0 env $CTL_ENV "$CR_CLIENT" -i -p "$FAKE_PID"
check "ctl: dest-path selective ckpt" 0 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/dump.bin"
expect_file "ctl: dump created" "$WORK/dump.bin"
check "ctl: dest-path selective restore" 0 env $CTL_ENV "$CR_CLIENT" -r -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/dump.bin"
check "ctl: buffer selective ckpt"   0 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"
check "ctl: full ckpt"               0 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID"
check "ctl: full restore (stub toggle)" 0 env $CTL_ENV "$CR_CLIENT" -r -p "$FAKE_PID"
expect_file "ctl: cuda-checkpoint toggled on success" "$TOGGLE_MARKER"

# --- torn dump is refused post-checkpoint ----------------------------------
start_fake $CTL_ENV FAKE_SKIP_COMMIT=1
check "torn dump: ckpt fails validation" 2 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/dump.bin"

# --- restore refuses a torn dump (fake mirrors .so via ValidateDumpFd) -----
start_fake $CTL_ENV
head -c 100 /dev/zero > "$WORK/torn.bin"
check "torn dump: restore refused"   2 env $CTL_ENV "$CR_CLIENT" -r -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/torn.bin"

# --- op_status propagation --------------------------------------------------
start_fake $CTL_ENV FAKE_OP_STATUS=28    # ENOSPC
check "op_status: selective ckpt surfaces failure" 2 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/dump.bin"
check "op_status: full ckpt fails cleanly" 2 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID"
expect_no_file "op_status: NOT frozen after a failed ckpt" "$TOGGLE_MARKER"
# Restore thaws before the op outcome is known (upstream ordering: the
# in-process handler needs a live CUDA context), so a failed restore
# still exits 2 but the toggle has already run.
check "op_status: full restore fails cleanly" 2 env $CTL_ENV "$CR_CLIENT" -r -p "$FAKE_PID"
expect_file "op_status: restore thawed before the failed op" "$TOGGLE_MARKER"

# --- timeout ----------------------------------------------------------------
start_fake $CTL_ENV FAKE_NO_FINISH=1
check "timeout: wedged workload fails the op" 4 \
    env $CTL_ENV GPU_CR_OP_TIMEOUT_SEC=2 "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"

# --- advertisement gating ----------------------------------------------------
start_fake $CTL_ENV FAKE_V1=1            # v1 .so: no advertisement written
check "gate: no advertisement -> refused" 3 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"

start_fake $CTL_ENV
ADV="$CTL/ctl-ready-$FAKE_PID"
sed -i 's/starttime=[0-9]*/starttime=1/' "$ADV"
check "gate: starttime mismatch (PID reuse) -> refused" 3 env $CTL_ENV "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"

# --- broken ctl path is refused loudly ---------------------------------------
start_fake "EXPORT_FILE_PATH=$WORK"      # fake in legacy mode
check "gate: non-tmpfs GPU_CR_CTL_PATH -> refused" 3 \
    env EXPORT_FILE_PATH="$WORK" GPU_CR_CTL_PATH="$WORK" "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"
check "gate: missing GPU_CR_CTL_PATH dir -> refused" 3 \
    env EXPORT_FILE_PATH="$WORK" GPU_CR_CTL_PATH=/nonexistent-ctl "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"

# --- legacy mode -------------------------------------------------------------
start_fake "EXPORT_FILE_PATH=$WORK"
check "legacy: init"                  0 env EXPORT_FILE_PATH="$WORK" "$CR_CLIENT" -i -p "$FAKE_PID"
check "legacy: buffer selective ckpt" 0 env EXPORT_FILE_PATH="$WORK" "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"
check "legacy: dest-path ckpt with capability" 0 \
    env EXPORT_FILE_PATH="$WORK" "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/dump.bin"

# --- version skew: v1 .so ----------------------------------------------------
start_fake "EXPORT_FILE_PATH=$WORK" FAKE_V1=1
check "skew: v1 .so + dest-path -> refused pre-signal" 3 \
    env EXPORT_FILE_PATH="$WORK" "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/dump.bin"
check "skew: v1 .so + buffer op keeps historical behavior" 0 \
    env EXPORT_FILE_PATH="$WORK" "$CR_CLIENT" -c -p "$FAKE_PID" -s "$REGIONS"

# --- usage errors ------------------------------------------------------------
check "usage: -o without -s"          1 env $CTL_ENV "$CR_CLIENT" -c -o "$WORK/dump.bin" -p 1
start_fake $CTL_ENV
check "usage: restore from missing file" 2 \
    env $CTL_ENV "$CR_CLIENT" -r -p "$FAKE_PID" -s "$REGIONS" -o "$WORK/nonexistent.bin"

echo
echo "cr_client integration: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
