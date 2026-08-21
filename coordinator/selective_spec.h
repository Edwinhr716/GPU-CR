#ifndef GPU_CR_COORDINATOR_SELECTIVE_SPEC_H_
#define GPU_CR_COORDINATOR_SELECTIVE_SPEC_H_

// Parser for the cr_client "-s ptr:size,ptr:size,..." region spec.
// Header-only and GPU-free so unit tests can exercise it directly.

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

namespace gpu_cr {

// Parses one unsigned number occupying the WHOLE string: any strtoull
// base-0 form (0x..., decimal), no sign, no leading space, no trailing
// junk. The digit-lead check rejects "", "-1" (which strtoull would
// silently negate to ULLONG_MAX), "+1" and " 1".
inline bool ParseFullUint64(const char* s, uint64_t* out) {
  if (!isdigit(static_cast<unsigned char>(s[0]))) return false;
  char* end = nullptr;
  *out = strtoull(s, &end, 0);
  return end != s && *end == '\0';
}

// Fills req->regions/num_regions from a comma-separated "ptr:size" list.
// Pointers and sizes accept any strtoull base-0 form (0x..., decimal).
// Syntax only: duplicate or overlapping regions pass through — their
// semantics belong to the .so and the agent above it. Returns false —
// with a message on stderr — for an empty spec, an empty or malformed
// region (missing colon, sign, junk before or after either number), a
// zero size, or more than kMaxSelectiveRegions regions. req->num_regions
// is only meaningful on success.
inline bool ParseSelectiveRegions(const char* spec, SelectiveCrRequest* req) {
  req->num_regions = 0;
  char* buf = strdup(spec);
  if (!buf) return false;
  // strsep, not strtok_r: empty tokens must surface as malformed regions
  // ("a:1,,b:2", trailing comma), not silently collapse.
  char* rest = buf;
  for (char* token = strsep(&rest, ","); token; token = strsep(&rest, ",")) {
    if (req->num_regions >= kMaxSelectiveRegions) {
      fprintf(stderr, "Error: too many selective regions (max %u)\n",
              kMaxSelectiveRegions);
      free(buf);
      return false;
    }
    char* colon = strchr(token, ':');
    if (!colon) {
      fprintf(stderr, "Error: invalid region format '%s' (expected ptr:size)\n",
              token);
      free(buf);
      return false;
    }
    *colon = '\0';
    uint64_t ptr = 0;
    uint64_t size = 0;
    if (!ParseFullUint64(token, &ptr) || !ParseFullUint64(colon + 1, &size)) {
      fprintf(stderr,
              "Error: invalid region '%s:%s' (expected unsigned ptr:size)\n",
              token, colon + 1);
      free(buf);
      return false;
    }
    if (size == 0) {
      fprintf(stderr, "Error: region size is 0 for ptr %s\n", token);
      free(buf);
      return false;
    }
    req->regions[req->num_regions].ptr = reinterpret_cast<void*>(ptr);
    req->regions[req->num_regions].size = size;
    req->num_regions++;
  }
  free(buf);
  return req->num_regions > 0;
}

}  // namespace gpu_cr

#endif  // GPU_CR_COORDINATOR_SELECTIVE_SPEC_H_
