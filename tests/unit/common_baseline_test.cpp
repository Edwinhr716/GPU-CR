// Upstream-baseline pieces of common.h (present since v0.2.1): the 2MB
// rounding macro, hugepage geometry, the C/R signal numbers, and the v1
// control-word prefix that every later extension must leave undisturbed.

#include "common.h"

#include <stddef.h>

#include <set>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

TEST(RoundUp2MBTest, ZeroStaysZero) {
  EXPECT_EQ(ROUND_UP_2MB(0UL), 0UL);
}

TEST(RoundUp2MBTest, RoundsUpToNextBoundary) {
  EXPECT_EQ(ROUND_UP_2MB(1UL), 2UL << 20);
  EXPECT_EQ(ROUND_UP_2MB((2UL << 20) - 1), 2UL << 20);
  EXPECT_EQ(ROUND_UP_2MB((2UL << 20) + 1), 4UL << 20);
}

TEST(RoundUp2MBTest, ExactMultiplesUnchanged) {
  EXPECT_EQ(ROUND_UP_2MB(2UL << 20), 2UL << 20);
  EXPECT_EQ(ROUND_UP_2MB(64UL << 20), 64UL << 20);
}

TEST(RoundUp2MBTest, Idempotent) {
  size_t once = ROUND_UP_2MB(12345UL);
  EXPECT_EQ(ROUND_UP_2MB(once), once);
}

// The VMM mapping granule and the hugepage size are the same 2MB constant;
// ROUND_UP_2MB and GranuleClampLen must agree on where boundaries are.
TEST(HugePageGeometryTest, GranuleMatchesHugePageSize) {
  EXPECT_EQ(static_cast<size_t>(HUGE_PAGE_SIZE), kVmmGranuleSize);
  EXPECT_EQ(ROUND_UP_2MB(1UL), kVmmGranuleSize);
}

// The v1 control word must stay at offset 0: a v0.2.1 cr_client and a
// current .so share the same zero-initialized mapping.
TEST(SignalControlsLayoutTest, SignalWordIsFirst) {
  EXPECT_EQ(offsetof(signal_controls, signal), 0u);
}

// The dump filesystem header is reserved at ROUND_UP_2MB(sizeof) — it must
// actually fit there, and the first data extent must start 2MB-aligned.
TEST(SharedMemFsTest, HeaderFitsReservedExtent) {
  size_t first_data_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));
  EXPECT_GE(first_data_offset, sizeof(shared_mem_fs));
  EXPECT_EQ(first_data_offset % kVmmGranuleSize, 0u);
}

TEST(SharedMemFsTest, FileTableHoldsMaxFileNumEntries) {
  shared_mem_fs fs;
  EXPECT_EQ(sizeof(fs.files) / sizeof(fs.files[0]),
            static_cast<size_t>(MAX_FILE_NUM));
}

// The six C/R signals must be pairwise distinct or an op would alias
// another; the NCCL names are documented aliases of the IPC ones.
TEST(SignalNumbersTest, DistinctAndAliased) {
  std::set<int> signals = {CR_INIT_SIGNAL,         CR_CKPT_SIGNAL,
                           CR_RESTORE_SIGNAL,      CR_IPC_TEARDOWN_SIGNAL,
                           CR_IPC_REBUILD_SIGNAL,  CR_IPC_VALIDATE_SIGNAL};
  EXPECT_EQ(signals.size(), 6u);
  EXPECT_EQ(CR_NCCL_SUSPEND_SIGNAL, CR_IPC_TEARDOWN_SIGNAL);
  EXPECT_EQ(CR_NCCL_RESUME_SIGNAL, CR_IPC_REBUILD_SIGNAL);
}

}  // namespace
}  // namespace gpu_cr
