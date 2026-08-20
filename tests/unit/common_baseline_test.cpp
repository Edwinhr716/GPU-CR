// Upstream-baseline pieces of common.h (inherited from upstream): the 2MB
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

// The macro's mask is a sign-extended int; a rewrite to an unsigned
// 32-bit mask would zero-extend and truncate every multi-GiB size
// (SHM_SIZE defaults to 25 GiB) while all sub-4GiB cases keep passing.
TEST(RoundUp2MBTest, LargeValuesKeepFullWidth) {
  EXPECT_EQ(ROUND_UP_2MB(25UL << 30), 25UL << 30);
  EXPECT_EQ(ROUND_UP_2MB((25UL << 30) + 1), (25UL << 30) + (2UL << 20));
}

// The VMM mapping granule and the hugepage size are the same 2MB constant;
// ROUND_UP_2MB and GranuleClampLen must agree on where boundaries are.
TEST(HugePageGeometryTest, GranuleMatchesHugePageSize) {
  EXPECT_EQ(static_cast<size_t>(HUGE_PAGE_SIZE), kVmmGranuleSize);
  EXPECT_EQ(ROUND_UP_2MB(1UL), kVmmGranuleSize);
}

// The v1 control word must stay at offset 0: an upstream cr_client and a
// current .so share the same zero-initialized mapping.
TEST(SignalControlsLayoutTest, SignalWordIsFirst) {
  EXPECT_EQ(offsetof(signal_controls, signal), 0u);
}

// The dump header is reserved exactly one 2MB extent at the front of the
// buffer; growing files[] past one hugepage would silently shift every
// data extent. The equality also implies the header fits (round-up never
// shrinks a value), and RoundUp2MBTest pins the macro itself.
TEST(SharedMemFsTest, HeaderFitsWithinOneHugepageExtent) {
  EXPECT_EQ(ROUND_UP_2MB(sizeof(shared_mem_fs)),
            static_cast<size_t>(HUGE_PAGE_SIZE));
}

// shared_mem_file / shared_mem_fs are the persisted dump-header layout,
// shared between cr_client and the .so through the same mapping: pin the
// exact x86-64 offsets and sizes so an added/reordered field (or a
// changed MAX_FILE_NUM) fails here instead of corrupting dumps.
TEST(SharedMemFsTest, FileEntryLayoutIsPinned) {
  EXPECT_EQ(offsetof(shared_mem_file, ptr), 0u);
  EXPECT_EQ(offsetof(shared_mem_file, start_offset), 8u);
  EXPECT_EQ(offsetof(shared_mem_file, size), 16u);
  EXPECT_EQ(sizeof(shared_mem_file), 24u);
}

TEST(SharedMemFsTest, HeaderLayoutIsPinned) {
  EXPECT_EQ(offsetof(shared_mem_fs, file_num), 0u);
  EXPECT_EQ(offsetof(shared_mem_fs, current_offset), 8u);
  EXPECT_EQ(offsetof(shared_mem_fs, files), 16u);
  EXPECT_EQ(static_cast<size_t>(MAX_FILE_NUM), 4096u);
  EXPECT_EQ(sizeof(shared_mem_fs), 16u + 4096u * 24u);  // 98320
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
