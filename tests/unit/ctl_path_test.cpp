// GEP-0006 control-plane path resolution, /proc/<pid>/stat parsing, and
// readiness-advertisement round trip. Linux-only (statfs, /proc).

#include "ctl_path.h"

#include <stdlib.h>
#include <unistd.h>

#include <string>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

class CtlPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    unsetenv("GPU_CR_CTL_PATH");
    unsetenv("EXPORT_FILE_PATH");
  }
};

TEST_F(CtlPathTest, RoundUp4K) {
  EXPECT_EQ(RoundUp4K(0), 0u);
  EXPECT_EQ(RoundUp4K(1), 4096u);
  EXPECT_EQ(RoundUp4K(4096), 4096u);
  EXPECT_EQ(RoundUp4K(4097), 8192u);
}

TEST_F(CtlPathTest, DataDirDefaultsToHugeCkpt) {
  EXPECT_STREQ(DataDir(), "/mnt/huge-ckpt");
}

TEST_F(CtlPathTest, DataDirHonorsExportFilePath) {
  setenv("EXPORT_FILE_PATH", "/some/dir", 1);
  EXPECT_STREQ(DataDir(), "/some/dir");
}

TEST_F(CtlPathTest, EmptyExportFilePathTreatedAsUnset) {
  setenv("EXPORT_FILE_PATH", "", 1);
  EXPECT_STREQ(DataDir(), "/mnt/huge-ckpt");
}

TEST_F(CtlPathTest, CtlDirUnsetIsLegacyMode) {
  bool ctl_mode = true;
  EXPECT_STREQ(CtlDir(&ctl_mode), "/mnt/huge-ckpt");
  EXPECT_FALSE(ctl_mode);
}

TEST_F(CtlPathTest, CtlDirTmpfsEnablesCtlMode) {
  if (!DirIsTmpfs("/dev/shm")) GTEST_SKIP() << "/dev/shm is not tmpfs here";
  setenv("GPU_CR_CTL_PATH", "/dev/shm", 1);
  bool ctl_mode = false;
  EXPECT_STREQ(CtlDir(&ctl_mode), "/dev/shm");
  EXPECT_TRUE(ctl_mode);
}

TEST_F(CtlPathTest, CtlDirNonTmpfsFallsBackToLegacy) {
  if (DirIsTmpfs("/")) GTEST_SKIP() << "/ is tmpfs here";
  setenv("GPU_CR_CTL_PATH", "/", 1);
  setenv("EXPORT_FILE_PATH", "/data/dir", 1);
  bool ctl_mode = true;
  EXPECT_STREQ(CtlDir(&ctl_mode), "/data/dir");
  EXPECT_FALSE(ctl_mode);
}

TEST_F(CtlPathTest, CtlDirMissingPathFallsBackToLegacy) {
  setenv("GPU_CR_CTL_PATH", "/nonexistent-gpu-cr-test-dir", 1);
  bool ctl_mode = true;
  EXPECT_STREQ(CtlDir(&ctl_mode), "/mnt/huge-ckpt");
  EXPECT_FALSE(ctl_mode);
}

TEST_F(CtlPathTest, DirIsTmpfsFalseForMissingDir) {
  EXPECT_FALSE(DirIsTmpfs("/nonexistent-gpu-cr-test-dir"));
}

// stat-line fields: pid (comm) state then 18 numbers with the 19th
// post-state field (field 22 overall) being starttime.
TEST_F(CtlPathTest, ParseStarttimeFromStat) {
  const char* line =
      "7 (proc) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 9999 555";
  EXPECT_EQ(ParseStarttimeFromStat(line), 9999);
}

// comm may contain spaces and parentheses; parsing anchors on the LAST ')'.
TEST_F(CtlPathTest, ParseStarttimeHandlesHostileComm) {
  const char* line =
      "7 (we ird) (name) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 "
      "4242 555";
  EXPECT_EQ(ParseStarttimeFromStat(line), 4242);
}

TEST_F(CtlPathTest, ParseStarttimeRejectsGarbage) {
  EXPECT_EQ(ParseStarttimeFromStat("not a stat line"), -1);
  EXPECT_EQ(ParseStarttimeFromStat("7 (proc) S 1 2 3"), -1);  // too few fields
  EXPECT_EQ(ParseStarttimeFromStat(""), -1);
}

TEST_F(CtlPathTest, ProcStarttimeOfSelfIsPositive) {
  EXPECT_GT(ProcStarttime(getpid()), 0);
}

TEST_F(CtlPathTest, ProcStarttimeOfMissingPidIsNegative) {
  EXPECT_EQ(ProcStarttime(999999999), -1);
}

TEST_F(CtlPathTest, ParseAdvertisement) {
  Advertisement adv;
  ASSERT_TRUE(ParseAdvertisement(
      "proto=3 starttime=12345 ctl=/dev/shm shm_mb=8192 staging_mb=1024 "
      "deferred=0\n",
      &adv));
  EXPECT_EQ(adv.proto, 3);
  EXPECT_EQ(adv.starttime, 12345);
  EXPECT_STREQ(adv.ctl, "/dev/shm");
}

TEST_F(CtlPathTest, ParseAdvertisementRejectsEmpty) {
  Advertisement adv;
  EXPECT_FALSE(ParseAdvertisement("", &adv));
  EXPECT_FALSE(ParseAdvertisement(nullptr, &adv));
}

TEST_F(CtlPathTest, ParseAdvertisementDefaultsWhenFieldsMissing) {
  Advertisement adv;
  ASSERT_TRUE(ParseAdvertisement("proto=2\n", &adv));
  EXPECT_EQ(adv.proto, 2);
  EXPECT_EQ(adv.starttime, -1);
  EXPECT_STREQ(adv.ctl, "");
}

TEST_F(CtlPathTest, WriteAdvertisementRoundTrip) {
  char dir_template[] = "/tmp/gpu-cr-ctl-test-XXXXXX";
  ASSERT_NE(mkdtemp(dir_template), nullptr);
  ASSERT_TRUE(WriteAdvertisement(dir_template, getpid(), 8192, 1024, false));

  std::string path =
      std::string(dir_template) + "/ctl-ready-" + std::to_string(getpid());
  FILE* f = fopen(path.c_str(), "r");
  ASSERT_NE(f, nullptr);
  char buf[600] = "";
  ASSERT_NE(fgets(buf, sizeof(buf), f), nullptr);
  fclose(f);

  Advertisement adv;
  ASSERT_TRUE(ParseAdvertisement(buf, &adv));
  EXPECT_EQ(adv.proto, kCtlProto);
  EXPECT_EQ(adv.starttime, ProcStarttime(getpid()));
  EXPECT_STREQ(adv.ctl, dir_template);

  unlink(path.c_str());
  rmdir(dir_template);
}

}  // namespace
}  // namespace gpu_cr
