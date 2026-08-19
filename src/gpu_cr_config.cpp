#include "common.h"

namespace gpu_cr {

// Single definition of the buffer config singleton. Function-local
// static: initialized on first call (init() calls it first thing, so
// signal handlers only ever see cached values), thread-safe per C++11.
const BufConfig& Config() {
  static BufConfig cfg =
      internal::Load(kShmDefaultBytes, kStagingDefaultBytes);
  return cfg;
}

}  // namespace gpu_cr
