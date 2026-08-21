// cr_client "-s ptr:size,..." region-spec parser.

#include "selective_spec.h"

#include <string>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

TEST(ParseSelectiveRegionsTest, SingleHexRegion) {
  SelectiveCrRequest req;
  ASSERT_TRUE(ParseSelectiveRegions("0x7f0000000000:4096", &req));
  ASSERT_EQ(req.num_regions, 1u);
  EXPECT_EQ(req.regions[0].ptr, reinterpret_cast<void*>(0x7f0000000000UL));
  EXPECT_EQ(req.regions[0].size, 4096u);
}

TEST(ParseSelectiveRegionsTest, MultipleRegionsMixedBases) {
  SelectiveCrRequest req;
  ASSERT_TRUE(ParseSelectiveRegions("0x1000:0x2000,4096:65536", &req));
  ASSERT_EQ(req.num_regions, 2u);
  EXPECT_EQ(req.regions[0].ptr, reinterpret_cast<void*>(0x1000UL));
  EXPECT_EQ(req.regions[0].size, 0x2000u);
  EXPECT_EQ(req.regions[1].ptr, reinterpret_cast<void*>(4096UL));
  EXPECT_EQ(req.regions[1].size, 65536u);
}

TEST(ParseSelectiveRegionsTest, RejectsEmptySpec) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsMissingColon) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("0x1000", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsZeroSize) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("0x1000:0", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsMalformedTrailingRegion) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("0x1000:4096,bogus", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsGarbagePtr) {
  SelectiveCrRequest req;
  // strtoull would quietly parse "bogus" as 0 — the parser must not.
  EXPECT_FALSE(ParseSelectiveRegions("bogus:4096", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsTrailingJunkInSize) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("0x1000:12junk", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsEmptyMiddleToken) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("0x1000:1,,0x2000:1", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsTrailingComma) {
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions("0x1000:1,", &req));
}

TEST(ParseSelectiveRegionsTest, RejectsNegativeSize) {
  SelectiveCrRequest req;
  // strtoull would negate "-1" to ULLONG_MAX — the parser must not.
  EXPECT_FALSE(ParseSelectiveRegions("0x1000:-1", &req));
}

// Syntax-only parser: duplicate regions pass through untouched —
// dedup/overlap semantics belong to the .so and the agent above it.
TEST(ParseSelectiveRegionsTest, AcceptsDuplicateRegions) {
  SelectiveCrRequest req;
  ASSERT_TRUE(ParseSelectiveRegions("0x1000:4096,0x1000:4096", &req));
  ASSERT_EQ(req.num_regions, 2u);
  EXPECT_EQ(req.regions[0].ptr, req.regions[1].ptr);
  EXPECT_EQ(req.regions[0].size, req.regions[1].size);
}

TEST(ParseSelectiveRegionsTest, AcceptsExactlyMaxRegions) {
  std::string spec;
  for (uint32_t i = 0; i < kMaxSelectiveRegions; i++) {
    if (i) spec += ",";
    spec += "0x1000:1";
  }
  SelectiveCrRequest req;
  ASSERT_TRUE(ParseSelectiveRegions(spec.c_str(), &req));
  EXPECT_EQ(req.num_regions, kMaxSelectiveRegions);
}

TEST(ParseSelectiveRegionsTest, RejectsOverMaxRegions) {
  std::string spec;
  for (uint32_t i = 0; i < kMaxSelectiveRegions + 1; i++) {
    if (i) spec += ",";
    spec += "0x1000:1";
  }
  SelectiveCrRequest req;
  EXPECT_FALSE(ParseSelectiveRegions(spec.c_str(), &req));
}

}  // namespace
}  // namespace gpu_cr
