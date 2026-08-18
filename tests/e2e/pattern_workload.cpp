// CUDA pattern-verify workload for the GPU-CR e2e and perf-regression
// suites. Run under LD_PRELOAD=vGPU-<vendor>.so; the runner script drives
// checkpoint/restore via cr_client and asks this process to verify that
// GPU memory is byte-identical after restore.
//
// Protocol (file-based, single-threaded):
//   E2E_SPEC_FILE  written once at startup: "<pid>\n<ptr>:<size>,...\n" —
//                  the exact -s spec for cr_client.
//   E2E_CMD_FILE   runner writes "<seq> verify" or "<seq> exit", then this
//                  process consumes (unlinks) it.
//   E2E_RESP_FILE  responses appended: "<seq> ok" or "<seq> fail <detail>".
//
// Buffers: E2E_NUM_BUFFERS (default 4) x E2E_BUFFER_MB (default 64) MiB,
// each filled with a deterministic per-buffer pattern.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace {

constexpr int kDefaultBuffers = 4;
constexpr size_t kDefaultBufferMb = 64;

uint8_t PatternByte(int buffer_idx, size_t offset) {
  return static_cast<uint8_t>((buffer_idx * 131 + offset * 7 + 17) & 0xFF);
}

size_t EnvSize(const char* name, size_t def) {
  const char* v = getenv(name);
  return (v && atoll(v) > 0) ? static_cast<size_t>(atoll(v)) : def;
}

const char* RequireEnv(const char* name) {
  const char* v = getenv(name);
  if (!v || !v[0]) {
    fprintf(stderr, "[pattern-workload] missing env %s\n", name);
    exit(1);
  }
  return v;
}

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      fprintf(stderr, "[pattern-workload] %s failed: %s\n", #call,        \
              cudaGetErrorString(err_));                                  \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

}  // namespace

int main() {
  const char* spec_file = RequireEnv("E2E_SPEC_FILE");
  const char* cmd_file = RequireEnv("E2E_CMD_FILE");
  const char* resp_file = RequireEnv("E2E_RESP_FILE");
  int num_buffers = static_cast<int>(EnvSize("E2E_NUM_BUFFERS", kDefaultBuffers));
  size_t bytes = EnvSize("E2E_BUFFER_MB", kDefaultBufferMb) << 20;

  std::vector<void*> bufs(num_buffers, nullptr);
  std::vector<uint8_t> host(bytes);

  for (int i = 0; i < num_buffers; i++) {
    CUDA_CHECK(cudaMalloc(&bufs[i], bytes));
    for (size_t off = 0; off < bytes; off++) host[off] = PatternByte(i, off);
    CUDA_CHECK(cudaMemcpy(bufs[i], host.data(), bytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Publish the selective-region spec.
  {
    std::string spec;
    char one[64];
    for (int i = 0; i < num_buffers; i++) {
      snprintf(one, sizeof(one), "%s%p:%zu", i ? "," : "", bufs[i], bytes);
      spec += one;
    }
    FILE* f = fopen(spec_file, "w");
    if (!f) { perror("spec file"); return 1; }
    fprintf(f, "%d\n%s\n", getpid(), spec.c_str());
    fclose(f);
    fprintf(stderr, "[pattern-workload] pid=%d spec=%s\n", getpid(),
            spec.c_str());
  }

  auto respond = [&](const char* seq, const char* result) {
    FILE* f = fopen(resp_file, "a");
    if (!f) return;
    fprintf(f, "%s %s\n", seq, result);
    fclose(f);
  };

  for (;;) {
    FILE* f = fopen(cmd_file, "r");
    if (!f) {
      usleep(100 * 1000);
      continue;
    }
    char seq[64] = "", cmd[64] = "";
    int n = fscanf(f, "%63s %63s", seq, cmd);
    fclose(f);
    unlink(cmd_file);
    if (n != 2) continue;

    if (strcmp(cmd, "exit") == 0) {
      respond(seq, "ok");
      return 0;
    }
    if (strcmp(cmd, "verify") == 0) {
      char detail[128] = "ok";
      bool all_ok = true;
      for (int i = 0; i < num_buffers && all_ok; i++) {
        cudaError_t err =
            cudaMemcpy(host.data(), bufs[i], bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
          snprintf(detail, sizeof(detail), "fail memcpy buf%d %s", i,
                   cudaGetErrorName(err));
          all_ok = false;
          break;
        }
        for (size_t off = 0; off < bytes; off++) {
          if (host[off] != PatternByte(i, off)) {
            snprintf(detail, sizeof(detail),
                     "fail mismatch buf%d off%zu got%02x want%02x", i, off,
                     host[off], PatternByte(i, off));
            all_ok = false;
            break;
          }
        }
      }
      respond(seq, detail);
      fprintf(stderr, "[pattern-workload] verify -> %s\n", detail);
      continue;
    }
    respond(seq, "fail unknown-command");
  }
}
