#include "common.h"

// Single definition of the KEP-0002 config singleton. Function-local
// static: initialized on first call (init() calls it first thing, so
// signal handlers only ever see cached values), thread-safe per C++11.
const GpuCrBufConfig& gpu_cr_config() {
    static GpuCrBufConfig cfg =
        gpu_cr_config_detail::load(GPU_CR_SHM_DEFAULT, GPU_CR_STAGING_DEFAULT);
    return cfg;
}
