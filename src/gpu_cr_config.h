#ifndef GPU_CR_CONFIG_H
#define GPU_CR_CONFIG_H

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
// unspecified, so init() invokes gpu_cr_config() as its first statement
// and the signal handlers only ever consume cached values.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CR_SHM_FLOOR      (64UL << 20)
#define GPU_CR_STAGING_FLOOR  (128UL << 20)
#define GPU_CR_ALIGN_2MB(x)   (((x) + (2UL << 20) - 1) & ~((2UL << 20) - 1))

struct GpuCrBufConfig {
    size_t shm_size;      // dump-buffer size when (or if) materialized
    size_t staging_size;  // per staging buffer (two are allocated)
    bool   shm_deferred;  // GPU_CR_SHM_*=0: create only on first use
};

namespace gpu_cr_config_detail {

inline long long parse_ll(const char* v, bool* ok) {
    char* end = nullptr;
    long long n = strtoll(v, &end, 10);
    *ok = (end && *end == '\0' && end != v && n >= 0);
    return n;
}

inline GpuCrBufConfig load(size_t shm_default, size_t staging_default) {
    GpuCrBufConfig cfg;
    cfg.shm_size = shm_default;
    cfg.staging_size = staging_default;
    cfg.shm_deferred = false;

    // Legacy names were shipped in manifests for years but never read by
    // any code; honoring them retroactively would break working
    // deployments (KEP-0002 Drawback 3), so they warn instead.
    if (getenv("GPUCR_SHM_GB") || getenv("GPUCR_STAGING_MB"))
        fprintf(stderr, "[gpu-cr-config] WARNING: GPUCR_SHM_GB/GPUCR_STAGING_MB were never "
                        "read by any GPU-CR version and remain ignored; use GPU_CR_SHM_GB / "
                        "GPU_CR_SHM_MB / GPU_CR_STAGING_MB\n");

    const char* src = "build default";
    const char* mb = getenv("GPU_CR_SHM_MB");
    const char* gb = getenv("GPU_CR_SHM_GB");
    const char* val = (mb && mb[0]) ? mb : ((gb && gb[0]) ? gb : nullptr);
    if (val) {
        bool ok = false;
        long long n = parse_ll(val, &ok);
        size_t bytes = ok ? (size_t)n << ((mb && mb[0]) ? 20 : 30) : 0;
        if (!ok) {
            fprintf(stderr, "[gpu-cr-config] WARNING: unparsable GPU_CR_SHM_%s=%s; using build default\n",
                    (mb && mb[0]) ? "MB" : "GB", val);
        } else if (n == 0) {
            cfg.shm_deferred = true;
            cfg.shm_size = GPU_CR_SHM_FLOOR; // creation size if a buffer-path op forces it
            src = "env (deferred)";
        } else if (bytes < GPU_CR_SHM_FLOOR) {
            fprintf(stderr, "[gpu-cr-config] WARNING: GPU_CR_SHM below the 64MiB floor (%s); using build default\n", val);
        } else {
            cfg.shm_size = GPU_CR_ALIGN_2MB(bytes);
            src = (mb && mb[0]) ? "env GPU_CR_SHM_MB" : "env GPU_CR_SHM_GB";
        }
    }

    const char* smb = getenv("GPU_CR_STAGING_MB");
    if (smb && smb[0]) {
        bool ok = false;
        long long n = parse_ll(smb, &ok);
        size_t bytes = ok ? (size_t)n << 20 : 0;
        if (!ok || bytes < GPU_CR_STAGING_FLOOR) {
            fprintf(stderr, "[gpu-cr-config] WARNING: GPU_CR_STAGING_MB=%s unparsable or below the "
                            "128MiB floor; using default\n", smb);
        } else {
            cfg.staging_size = GPU_CR_ALIGN_2MB(bytes);
        }
    }

    fprintf(stderr, "[gpu-cr-config] dump buffer %zu MiB (%s)%s, staging 2 x %zu MiB\n",
            cfg.shm_size >> 20, src, cfg.shm_deferred ? " [deferred until first buffer-path op]" : "",
            cfg.staging_size >> 20);
    return cfg;
}

} // namespace gpu_cr_config_detail

// Defined by the including translation unit's defaults (common.h) so this
// header stays macro-order independent.
const GpuCrBufConfig& gpu_cr_config();

#endif
