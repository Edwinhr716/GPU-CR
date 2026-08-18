# GPU-CR test suites

Four tiers, from GPU-free unit tests to on-node performance regression.
The first two run automatically inside every `Dockerfile.build` image build
(`RUN_TESTS=1`, the default) — a red suite fails the build.

## 1. Unit tests — `tests/unit/` (GoogleTest, no GPU, Linux)

Function-level coverage of the code added on top of upstream v0.2.1:
buffer-size config parsing (KEP-0002), control-path resolution and
advertisement round-trip (GEP-0006), dump-format validation (GEP-0001),
granule clamping (GEP-0005), consume-once FINISH bookkeeping, region-spec
parsing, and wire-layout guards.

```sh
cmake -DGPU_VENDOR=NVIDIA -DGPU_CR_BUILD_TESTS=ON .. && make && ctest -R unit
```

## 2. Integration tests — `tests/integration/` (no GPU, Linux)

`cr_client_integration_test.sh` drives the **real `cr_client` binary**
against `fake_workload`, a GPU-free stand-in for the vGPU.so side that
reuses the production control-channel code (ShareMemComm, advertisement
writer, FINISH bookkeeping, dump validator). Covers ctl and legacy modes,
destination-path checkpoint/restore, torn-dump refusal, op_status
propagation (including the "never toggle cuda-checkpoint on failure" gate),
timeouts, PID-reuse refusal, version skew, and the documented exit codes
(0 OK / 1 usage / 2 op failed / 3 refused / 4 timeout).

```sh
ctest -R cr_client_integration --output-on-failure
```

Both GPU-free tiers in one shot via Cloud Build (no image published):

```sh
gcloud builds submit --config cloudbuild-test.yaml .
```

## 3. End-to-end — `tests/e2e/run_e2e.sh` (GPU node)

A real CUDA workload (`pattern_workload`) under `LD_PRELOAD=vGPU-NVIDIA.so`
goes through destination-path selective C/R, buffer-path selective C/R, and
full C/R, gating on **byte-identical GPU memory after every restore**.

## 4. Performance regression — `tests/e2e/perf_regression.sh` (GPU node)

Verifies the consolidated build has not regressed the full checkpoint/
restore data plane that upstream v0.2.1 (`e9bbb52`) delivers: same workload,
same node, baseline .so vs candidate .so, median-of-N compared against a
threshold (default 15%). Candidate selective-path timings are recorded as
informational (no v0.2.1 baseline exists for them).

Build the baseline once with `tests/e2e/build_baseline_so.sh`, then run
both GPU tiers as a one-shot pod: `tests/e2e/e2e-pod.yaml` (exits 0 only if
every e2e gate and the perf gate pass).
