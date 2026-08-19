// Upstream-baseline memcpy_multi: the multi-threaded copy behind the
// staging pipelines must be byte-exact for every size/thread split.

#include <string.h>

#include <vector>

#include "common.h"
#include "gtest/gtest.h"

namespace {

// Fills src with a position-dependent pattern, copies, and checks dest
// byte-for-byte (plus a canary past the end to catch overruns).
void CheckCopy(size_t size) {
  std::vector<unsigned char> src(size + 1, 0);
  std::vector<unsigned char> dest(size + 1, 0xEE);
  for (size_t i = 0; i < size; i++) src[i] = static_cast<unsigned char>(i * 31 + 7);

  memcpy_multi(dest.data(), src.data(), size);

  ASSERT_EQ(memcmp(dest.data(), src.data(), size), 0) << "size=" << size;
  EXPECT_EQ(dest[size], 0xEE) << "overrun at size=" << size;
}

TEST(MemcpyMultiTest, ZeroSizeIsANoOp) {
  unsigned char dest[4] = {0xAA, 0xBB, 0xCC, 0xDD};
  unsigned char src[4] = {1, 2, 3, 4};
  memcpy_multi(dest, src, 0);
  EXPECT_EQ(dest[0], 0xAA);
  EXPECT_EQ(dest[3], 0xDD);
}

TEST(MemcpyMultiTest, SizesSmallerThanThreadCount) {
  // Fewer bytes than NUM_COPY_THREADS: trailing threads must not spawn on
  // out-of-range offsets.
  for (size_t size = 1; size < NUM_COPY_THREADS; size++) CheckCopy(size);
}

TEST(MemcpyMultiTest, ExactChunkMultiples) {
  CheckCopy(NUM_COPY_THREADS);
  CheckCopy(NUM_COPY_THREADS * 4096);
}

TEST(MemcpyMultiTest, OddSizesSplitUnevenly) {
  CheckCopy(NUM_COPY_THREADS + 1);
  CheckCopy(4096 - 1);
  CheckCopy((1UL << 20) + 7);
}

TEST(MemcpyMultiTest, MultiMegabyteCopyIsByteExact) {
  CheckCopy(8UL << 20);
}

}  // namespace
