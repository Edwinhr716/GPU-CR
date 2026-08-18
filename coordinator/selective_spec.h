#ifndef GPU_CR_COORDINATOR_SELECTIVE_SPEC_H_
#define GPU_CR_COORDINATOR_SELECTIVE_SPEC_H_

// Parser for the cr_client "-s ptr:size,ptr:size,..." region spec.
// Header-only and GPU-free so unit tests can exercise it directly.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

namespace gpu_cr {

// Fills req->regions/num_regions from a comma-separated "ptr:size" list.
// Pointers and sizes accept any strtoull base-0 form (0x..., decimal).
// Returns false — with a message on stderr — for an empty spec, a
// malformed region, a zero size, or more than kMaxSelectiveRegions
// regions. req->num_regions is only meaningful on success.
inline bool ParseSelectiveRegions(const char* spec, SelectiveCrRequest* req) {
  req->num_regions = 0;
  char* buf = strdup(spec);
  char* save = nullptr;
  for (char* token = strtok_r(buf, ",", &save); token;
       token = strtok_r(nullptr, ",", &save)) {
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
    void* ptr = reinterpret_cast<void*>(strtoull(token, nullptr, 0));
    uint64_t size = strtoull(colon + 1, nullptr, 0);
    if (size == 0) {
      fprintf(stderr, "Error: region size is 0 for ptr %s\n", token);
      free(buf);
      return false;
    }
    req->regions[req->num_regions].ptr = ptr;
    req->regions[req->num_regions].size = size;
    req->num_regions++;
  }
  free(buf);
  return req->num_regions > 0;
}

}  // namespace gpu_cr

#endif  // GPU_CR_COORDINATOR_SELECTIVE_SPEC_H_
