// Shared-protocol helpers from common.h: granule clamping and the
// consume-once FINISH bookkeeping.

#include "common.h"

#include <errno.h>
#include <stddef.h>
#include <string.h>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

TEST(GranuleClampLenTest, ShortCopyFromAlignedStartUnclamped) {
  EXPECT_EQ(GranuleClampLen(0, 4096), 4096u);
}

TEST(GranuleClampLenTest, FullGranuleFromAlignedStartUnclamped) {
  EXPECT_EQ(GranuleClampLen(0, kVmmGranuleSize), kVmmGranuleSize);
}

TEST(GranuleClampLenTest, LongCopyClampedToGranule) {
  EXPECT_EQ(GranuleClampLen(0, kVmmGranuleSize + 1), kVmmGranuleSize);
}

TEST(GranuleClampLenTest, UnalignedStartClampsToBoundary) {
  // 4 bytes shy of a boundary: at most 4 bytes may be issued.
  uintptr_t addr = kVmmGranuleSize - 4;
  EXPECT_EQ(GranuleClampLen(addr, 4096), 4u);
  EXPECT_EQ(GranuleClampLen(addr, 2), 2u);
}

TEST(GranuleClampLenTest, EveryClampEndsAlignedOrAtRegionEnd) {
  // Walk a simulated 5MiB+3 region from an unaligned base the way the
  // selective copy loops do, asserting the clamp invariant for each
  // issued copy: <=2MB, and granule-aligned at its start or its end
  // (or it is the final copy of the region).
  uintptr_t addr = 12345;
  size_t remaining = 5 * (1UL << 20) + 3;
  int copies = 0;
  while (remaining > 0) {
    size_t len = GranuleClampLen(addr, remaining);
    ASSERT_GT(len, 0u);
    ASSERT_LE(len, kVmmGranuleSize);
    bool start_aligned = (addr % kVmmGranuleSize) == 0;
    bool end_aligned = ((addr + len) % kVmmGranuleSize) == 0;
    bool is_final = (len == remaining);
    ASSERT_TRUE(start_aligned || end_aligned || is_final);
    addr += len;
    remaining -= len;
    ASSERT_LT(++copies, 100);
  }
}

TEST(FinishOpControlsTest, ReportsStatusAndAck) {
  signal_controls c;
  memset(&c, 0, sizeof(c));
  FinishOpControls(&c, ENOSPC);
  EXPECT_EQ(c.op_status, ENOSPC);
  EXPECT_EQ(c.proto_ack, kSelectiveCrProtoV2);
  EXPECT_EQ(c.capability & kCrCapDestPath, kCrCapDestPath);
}

// A stale dest_path must never redirect a later op: the v2 extension is
// consumed at FINISH.
TEST(FinishOpControlsTest, ConsumesRequestExtension) {
  signal_controls c;
  memset(&c, 0, sizeof(c));
  c.selective_req.proto_version = kSelectiveCrProtoV2;
  strncpy(c.selective_req.dest_path, "/store/dump.bin",
          kSelectiveCrMaxPath - 1);
  c.selective_req.num_regions = 2;  // v1 prefix is NOT consumed

  FinishOpControls(&c, 0);
  EXPECT_EQ(c.selective_req.proto_version, 0u);
  for (size_t i = 0; i < kSelectiveCrMaxPath; i++)
    ASSERT_EQ(c.selective_req.dest_path[i], '\0');
  EXPECT_EQ(c.selective_req.num_regions, 2u);
}

TEST(FinishOpControlsTest, CapabilityPersistsAcrossOps) {
  signal_controls c;
  memset(&c, 0, sizeof(c));
  c.capability = kCrCapDestPath;
  FinishOpControls(&c, EIO);
  EXPECT_EQ(c.capability & kCrCapDestPath, kCrCapDestPath);
}

// Layout guards for the shared-memory wire structs: the v2 fields are
// appended so the v1 prefix keeps its exact offsets.
TEST(WireLayoutTest, V1PrefixOffsetsUnchanged) {
  EXPECT_EQ(offsetof(SelectiveCrRequest, num_regions), 0u);
  EXPECT_EQ(offsetof(SelectiveCrRequest, regions), 8u);
  EXPECT_EQ(offsetof(signal_controls, selective_req), 8u);
}

}  // namespace
}  // namespace gpu_cr
