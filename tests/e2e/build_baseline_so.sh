#!/usr/bin/env bash
# Builds the PERF BASELINE artifacts from upstream GPU-CR (e9bbb52) — the
# reference point the perf-regression suite measures against. Applies the
# same deployment patches the consolidated build carries (signal remap;
# 0777 was still a build-time patch at e9bbb52) so both variants run in the
# identical harness and only GPU-CR code differs.
#
# Usage: build_baseline_so.sh [git-ref] [image-tag] [shm-size-gb]
# Produces asia-southeast1-docker.pkg.dev/$PROJECT/time-slicing/gpucr-so:<tag>
#
# shm-size-gb defaults to 5: the baseline has no KEP-0002 env sizing, so its
# compile-time dump buffer must fit the perf pod's hugepage request while
# still covering the perf workload (PERF_NUM_BUFFERS x PERF_BUFFER_MB).
set -eu
REF=${1:-e9bbb52e1f52986587fc631217c0f2b50b46245a}
TAG=${2:-baseline-e9bbb52}
SHM_SIZE_GB=${3:-5}

SRC=$(mktemp -d /tmp/gpu-cr-baseline.XXXXXX)
trap 'rm -rf "$SRC"' EXIT
git archive "$REF" | tar -x -C "$SRC"

cat > "$SRC/Dockerfile.baseline" <<'EOF'
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder
ARG SHM_SIZE_GB=25
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential && rm -rf /var/lib/apt/lists/*
COPY . /tmp/GPU-CR
RUN cd /tmp/GPU-CR && \
    sed -i 's/#define CR_CKPT_SIGNAL     SIGUSR1/#define CR_CKPT_SIGNAL     (SIGRTMAX - 8)/' src/common.h && \
    sed -i 's/#define CR_RESTORE_SIGNAL  SIGUSR2/#define CR_RESTORE_SIGNAL  (SIGRTMAX - 7)/' src/common.h && \
    sed -i 's/int fd_control = open(control_name, O_CREAT | O_RDWR, 0755);/int fd_control = open(control_name, O_CREAT | O_RDWR, 0777); fchmod(fd_control, 0777);/' src/comm/share_mem.cpp && \
    grep -q "SIGRTMAX - 8" src/common.h && \
    grep -q "0777" src/comm/share_mem.cpp && \
    mkdir build && cd build && \
    cmake -DGPU_VENDOR=NVIDIA -DSHM_SIZE_GB=${SHM_SIZE_GB} .. && \
    make -j$(nproc)
FROM busybox:stable
COPY --from=builder /tmp/GPU-CR/build/vGPU-NVIDIA.so /vGPU-NVIDIA.so
COPY --from=builder /tmp/GPU-CR/build/cr_client /cr_client
EOF

cat > "$SRC/cloudbuild-baseline.yaml" <<EOF
steps:
- name: gcr.io/cloud-builders/docker
  args: [build, -f, Dockerfile.baseline,
         --build-arg, SHM_SIZE_GB=$SHM_SIZE_GB,
         -t, asia-southeast1-docker.pkg.dev/\$PROJECT_ID/time-slicing/gpucr-so:$TAG, .]
images: [asia-southeast1-docker.pkg.dev/\$PROJECT_ID/time-slicing/gpucr-so:$TAG]
options: {machineType: E2_HIGHCPU_32, diskSizeGb: 200}
timeout: 3600s
EOF

gcloud builds submit --config "$SRC/cloudbuild-baseline.yaml" "$SRC"
echo "baseline image: gpucr-so:$TAG (from $REF)"
