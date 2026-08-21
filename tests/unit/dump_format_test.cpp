// Destination-dump layout validation (header plausibility and
// commit marker) against crafted files — no GPU required.

#include "dump_format.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

constexpr uint64_t kHeaderSize = ROUND_UP_2MB(sizeof(shared_mem_fs));

TEST(DumpHeaderPlausibleTest, AcceptsMinimalValidHeader) {
  EXPECT_TRUE(DumpHeaderPlausible(/*file_num=*/1, kHeaderSize,
                                  kHeaderSize + sizeof(DumpCommit)));
}

TEST(DumpHeaderPlausibleTest, RejectsZeroFiles) {
  EXPECT_FALSE(DumpHeaderPlausible(0, kHeaderSize,
                                   kHeaderSize + sizeof(DumpCommit)));
}

TEST(DumpHeaderPlausibleTest, RejectsFileNumAtLimit) {
  EXPECT_FALSE(DumpHeaderPlausible(MAX_FILE_NUM, kHeaderSize,
                                   kHeaderSize + sizeof(DumpCommit)));
}

TEST(DumpHeaderPlausibleTest, RejectsOffsetInsideHeader) {
  EXPECT_FALSE(DumpHeaderPlausible(1, kHeaderSize - 1,
                                   kHeaderSize + sizeof(DumpCommit)));
}

TEST(DumpHeaderPlausibleTest, RejectsCommitMarkerPastEof) {
  EXPECT_FALSE(DumpHeaderPlausible(1, kHeaderSize,
                                   kHeaderSize + sizeof(DumpCommit) - 1));
}

// A header-supplied offset near UINT64_MAX must not wrap
// current_offset + sizeof(DumpCommit) back under total_size.
TEST(DumpHeaderPlausibleTest, RejectsWrappingOffset) {
  EXPECT_FALSE(DumpHeaderPlausible(1, UINT64_MAX - 8,
                                   kHeaderSize + sizeof(DumpCommit)));
}

// Writes a synthetic dump: header (file_num, current_offset), one extent of
// `extent` bytes, and (optionally) the trailing commit marker.
class ValidateDumpFdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    snprintf(path_, sizeof(path_), "/tmp/gpu-cr-dump-test-XXXXXX");
    fd_ = mkstemp(path_);
    ASSERT_GE(fd_, 0);
  }
  void TearDown() override {
    close(fd_);
    unlink(path_);
  }

  void WriteDump(uint64_t file_num, uint64_t extent, bool with_commit,
                 uint64_t magic = kDumpCommitMagic) {
    uint64_t current_offset = kHeaderSize + extent;
    uint64_t hdr[2] = {file_num, current_offset};
    ASSERT_EQ(pwrite(fd_, hdr, sizeof(hdr), 0),
              static_cast<ssize_t>(sizeof(hdr)));
    ASSERT_EQ(ftruncate(fd_, current_offset + sizeof(DumpCommit)), 0);
    if (with_commit) {
      DumpCommit commit = {magic, /*generation=*/1};
      ASSERT_EQ(pwrite(fd_, &commit, sizeof(commit),
                       static_cast<off_t>(current_offset)),
                static_cast<ssize_t>(sizeof(commit)));
    }
  }

  char path_[64];
  int fd_ = -1;
};

TEST_F(ValidateDumpFdTest, AcceptsCommittedDump) {
  WriteDump(/*file_num=*/1, /*extent=*/4096, /*with_commit=*/true);
  EXPECT_TRUE(ValidateDumpFd(fd_));
}

TEST_F(ValidateDumpFdTest, RejectsTornDump) {
  WriteDump(1, 4096, /*with_commit=*/false);
  EXPECT_FALSE(ValidateDumpFd(fd_));
}

TEST_F(ValidateDumpFdTest, RejectsWrongMagic) {
  WriteDump(1, 4096, true, /*magic=*/0xdeadbeef);
  EXPECT_FALSE(ValidateDumpFd(fd_));
}

TEST_F(ValidateDumpFdTest, RejectsEmptyFile) {
  EXPECT_FALSE(ValidateDumpFd(fd_));
}

TEST_F(ValidateDumpFdTest, RejectsZeroFileNum) {
  WriteDump(0, 4096, true);
  EXPECT_FALSE(ValidateDumpFd(fd_));
}

TEST_F(ValidateDumpFdTest, RejectsTruncatedBelowCurrentOffset) {
  WriteDump(1, 4096, true);
  ASSERT_EQ(ftruncate(fd_, kHeaderSize), 0);
  EXPECT_FALSE(ValidateDumpFd(fd_));
}

}  // namespace
}  // namespace gpu_cr
