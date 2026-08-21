#include "gpu_cr_config.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common.h"

namespace gpu_cr {
namespace internal {

long long ParseNonNegative(const char* value, bool* ok) {
  char* end = nullptr;
  errno = 0;
  long long n = strtoll(value, &end, 10);
  *ok = (end && *end == '\0' && end != value && errno != ERANGE && n >= 0);
  return n;
}

namespace {

// Largest value that survives `n << shift` and the subsequent 2MiB
// round-up without wrapping size_t.
long long MaxShiftable(int shift) {
  return static_cast<long long>((SIZE_MAX - (HUGE_PAGE_SIZE - 1)) >> shift);
}

}  // namespace

BufConfig Load(size_t shm_default, size_t staging_default) {
  BufConfig cfg;
  cfg.shm_size = shm_default;
  cfg.staging_size = staging_default;
  cfg.shm_deferred = false;

  // Legacy names were shipped in manifests for years but never read by
  // any code; honoring them retroactively would break working
  // deployments, so they warn instead.
  if (getenv("GPUCR_SHM_GB") || getenv("GPUCR_STAGING_MB")) {
    fprintf(stderr,
            "[gpu-cr-config] WARNING: GPUCR_SHM_GB/GPUCR_STAGING_MB were "
            "never read by any GPU-CR version and remain ignored; use "
            "GPU_CR_SHM_GB / GPU_CR_SHM_MB / GPU_CR_STAGING_MB\n");
  }

  const char* src = "build default";
  const char* mb = getenv("GPU_CR_SHM_MB");
  const char* gb = getenv("GPU_CR_SHM_GB");
  const bool use_mb = (mb && mb[0]);
  const char* val = use_mb ? mb : ((gb && gb[0]) ? gb : nullptr);
  if (val) {
    const int shift = use_mb ? 20 : 30;
    bool ok = false;
    long long n = ParseNonNegative(val, &ok);
    if (!ok || n > MaxShiftable(shift)) {
      fprintf(stderr,
              "[gpu-cr-config] WARNING: unparsable or out-of-range "
              "GPU_CR_SHM_%s=%s; using build default\n",
              use_mb ? "MB" : "GB", val);
    } else if (n == 0) {
      cfg.shm_deferred = true;
      // Creation size if a buffer-path op forces materialization.
      cfg.shm_size = kShmFloorBytes;
      src = "env (deferred)";
    } else if ((static_cast<size_t>(n) << shift) < kShmFloorBytes) {
      fprintf(stderr,
              "[gpu-cr-config] WARNING: GPU_CR_SHM below the 64MiB floor "
              "(%s); using build default\n",
              val);
    } else {
      cfg.shm_size = ROUND_UP_2MB(static_cast<size_t>(n) << shift);
      src = use_mb ? "env GPU_CR_SHM_MB" : "env GPU_CR_SHM_GB";
    }
  }

  const char* smb = getenv("GPU_CR_STAGING_MB");
  if (smb && smb[0]) {
    bool ok = false;
    long long n = ParseNonNegative(smb, &ok);
    if (!ok || n > MaxShiftable(20) ||
        (static_cast<size_t>(n) << 20) < kStagingFloorBytes) {
      fprintf(stderr,
              "[gpu-cr-config] WARNING: GPU_CR_STAGING_MB=%s unparsable, "
              "out-of-range or below the 128MiB floor; using default\n",
              smb);
    } else {
      cfg.staging_size = ROUND_UP_2MB(static_cast<size_t>(n) << 20);
    }
  }

  fprintf(stderr,
          "[gpu-cr-config] dump buffer %zu MiB (%s)%s, staging 2 x %zu MiB\n",
          cfg.shm_size >> 20, src,
          cfg.shm_deferred ? " [deferred until first buffer-path op]" : "",
          cfg.staging_size >> 20);
  return cfg;
}

}  // namespace internal

// Single definition of the buffer config singleton. Function-local
// static: initialized on first call (init() calls it before registering
// any signal handler, so handlers only ever see cached values),
// thread-safe per C++11.
const BufConfig& Config() {
  static BufConfig cfg =
      internal::Load(kShmDefaultBytes, kStagingDefaultBytes);
  return cfg;
}

}  // namespace gpu_cr
