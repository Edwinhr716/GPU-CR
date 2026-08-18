#ifndef GPU_CR_SRC_GPU_CR_CONFIG_H_
#define GPU_CR_SRC_GPU_CR_CONFIG_H_

// KEP-0002: runtime-sizable checkpoint buffers.
//
// Buffer sizes come from the environment, read once and cached; the
// compile-time macros (SHM_SIZE_GB build arg, 1GiB staging) are the
// DEFAULTS, not literals — an env-unset SHM8-built image behaves exactly
// like today's SHM8 image (the fleet-default rule: rolling this code out
// must never change a deployment's footprint by itself).
//
//   GPU_CR_SHM_MB / GPU_CR_SHM_GB   dump buffer (MB wins if both set)
//       0  = deferred: no dump-buffer mapping until a buffer-path op
//            first needs it; it then materializes at the 64MiB floor
//            (covers the shared_mem_fs header + IPC scratch blocks).
//            For -o-only deployments; pool rule: 2×staging + 64MiB +
//            Σ destination files.
//   GPU_CR_STAGING_MB               each of the two DMA staging buffers
//
// No upper clamp: the hugepage pool is the real bound and mmap reports
// ENOMEM honestly. Floors: 64MiB dump (2MiB-aligned), 128MiB staging.
// Unparsable or below-floor values warn and fall back to the default —
// never a silent clamp.
//
// This is a function-local-static singleton, NOT a second ELF
// constructor: cross-TU constructor order vs the GEP-0006 init() is
// unspecified, so init() invokes gpu_cr::Config() as its first statement
// and the signal handlers only ever consume cached values.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace gpu_cr {

inline constexpr size_t kShmFloorBytes = 64UL << 20;
inline constexpr size_t kStagingFloorBytes = 128UL << 20;

constexpr size_t AlignUp2MB(size_t x) {
  return (x + (2UL << 20) - 1) & ~((2UL << 20) - 1);
}

struct BufConfig {
  size_t shm_size;      // dump-buffer size when (or if) materialized
  size_t staging_size;  // per staging buffer (two are allocated)
  bool shm_deferred;    // GPU_CR_SHM_*=0: create only on first use
};

namespace internal {

inline long long ParseNonNegative(const char* value, bool* ok) {
  char* end = nullptr;
  long long n = strtoll(value, &end, 10);
  *ok = (end && *end == '\0' && end != value && n >= 0);
  return n;
}

inline BufConfig Load(size_t shm_default, size_t staging_default) {
  BufConfig cfg;
  cfg.shm_size = shm_default;
  cfg.staging_size = staging_default;
  cfg.shm_deferred = false;

  // Legacy names were shipped in manifests for years but never read by
  // any code; honoring them retroactively would break working
  // deployments (KEP-0002 Drawback 3), so they warn instead.
  if (getenv("GPUCR_SHM_GB") || getenv("GPUCR_STAGING_MB")) {
    fprintf(stderr,
            "[gpu-cr-config] WARNING: GPUCR_SHM_GB/GPUCR_STAGING_MB were "
            "never read by any GPU-CR version and remain ignored; use "
            "GPU_CR_SHM_GB / GPU_CR_SHM_MB / GPU_CR_STAGING_MB\n");
  }

  const char* src = "build default";
  const char* mb = getenv("GPU_CR_SHM_MB");
  const char* gb = getenv("GPU_CR_SHM_GB");
  const char* val = (mb && mb[0]) ? mb : ((gb && gb[0]) ? gb : nullptr);
  if (val) {
    bool ok = false;
    long long n = ParseNonNegative(val, &ok);
    size_t bytes = ok ? static_cast<size_t>(n) << ((mb && mb[0]) ? 20 : 30) : 0;
    if (!ok) {
      fprintf(stderr,
              "[gpu-cr-config] WARNING: unparsable GPU_CR_SHM_%s=%s; using "
              "build default\n",
              (mb && mb[0]) ? "MB" : "GB", val);
    } else if (n == 0) {
      cfg.shm_deferred = true;
      // Creation size if a buffer-path op forces materialization.
      cfg.shm_size = kShmFloorBytes;
      src = "env (deferred)";
    } else if (bytes < kShmFloorBytes) {
      fprintf(stderr,
              "[gpu-cr-config] WARNING: GPU_CR_SHM below the 64MiB floor "
              "(%s); using build default\n",
              val);
    } else {
      cfg.shm_size = AlignUp2MB(bytes);
      src = (mb && mb[0]) ? "env GPU_CR_SHM_MB" : "env GPU_CR_SHM_GB";
    }
  }

  const char* smb = getenv("GPU_CR_STAGING_MB");
  if (smb && smb[0]) {
    bool ok = false;
    long long n = ParseNonNegative(smb, &ok);
    size_t bytes = ok ? static_cast<size_t>(n) << 20 : 0;
    if (!ok || bytes < kStagingFloorBytes) {
      fprintf(stderr,
              "[gpu-cr-config] WARNING: GPU_CR_STAGING_MB=%s unparsable or "
              "below the 128MiB floor; using default\n",
              smb);
    } else {
      cfg.staging_size = AlignUp2MB(bytes);
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

// Cached singleton; defaults come from the including translation unit's
// build configuration (common.h) so this header stays macro-order
// independent.
const BufConfig& Config();

}  // namespace gpu_cr

#endif  // GPU_CR_SRC_GPU_CR_CONFIG_H_
