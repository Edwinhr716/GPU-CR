// Buffer-config unit matrix: env parsing, bounds, defaults, legacy warnings.
// Host-only (no CUDA); exercises gpu_cr::internal::Load() directly so each
// case gets a fresh parse (the production singleton caches, by design).

#include "gpu_cr_config.h"

#include <stdlib.h>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

constexpr size_t kShm8 = 8UL << 30;  // an "SHM8 fleet build" default
constexpr size_t kStg1 = 1UL << 30;

class ConfigLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    unsetenv("GPU_CR_SHM_GB");
    unsetenv("GPU_CR_SHM_MB");
    unsetenv("GPU_CR_STAGING_MB");
    unsetenv("GPUCR_SHM_GB");
    unsetenv("GPUCR_STAGING_MB");
  }
  BufConfig Load() { return internal::Load(kShm8, kStg1); }
};

// Fleet-default rule (F1): env unset => the BUILD's default, exactly.
TEST_F(ConfigLoadTest, UnsetEnvUsesBuildDefaults) {
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_EQ(c.staging_size, kStg1);
  EXPECT_FALSE(c.shm_deferred);
}

TEST_F(ConfigLoadTest, ShmGbHonored) {
  setenv("GPU_CR_SHM_GB", "12", 1);
  EXPECT_EQ(Load().shm_size, 12UL << 30);
}

TEST_F(ConfigLoadTest, ShmMbWinsOverGb) {
  setenv("GPU_CR_SHM_MB", "300", 1);
  setenv("GPU_CR_SHM_GB", "12", 1);
  EXPECT_EQ(Load().shm_size, 300UL << 20);
}

// No upper clamp (F5): values above the old 25GiB literal are honored.
TEST_F(ConfigLoadTest, NoUpperClamp) {
  setenv("GPU_CR_SHM_GB", "40", 1);
  EXPECT_EQ(Load().shm_size, 40UL << 30);
}

// Floor (F5): below 64MiB -> default with warning, never a clamp.
TEST_F(ConfigLoadTest, BelowFloorFallsBackToDefault) {
  setenv("GPU_CR_SHM_MB", "10", 1);
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_FALSE(c.shm_deferred);
}

// Deferred mode (=0) stays special: not subsumed by the floor.
TEST_F(ConfigLoadTest, ZeroMeansDeferredAtFloor) {
  setenv("GPU_CR_SHM_GB", "0", 1);
  BufConfig c = Load();
  EXPECT_TRUE(c.shm_deferred);
  EXPECT_EQ(c.shm_size, kShmFloorBytes);
}

TEST_F(ConfigLoadTest, UnparsableFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "8x", 1);
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_FALSE(c.shm_deferred);
}

TEST_F(ConfigLoadTest, NegativeFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "-3", 1);
  EXPECT_EQ(Load().shm_size, kShm8);
}

TEST_F(ConfigLoadTest, ShmAlignsUpTo2MiB) {
  setenv("GPU_CR_SHM_MB", "129", 1);
  EXPECT_EQ(Load().shm_size, 130UL << 20);
}

TEST_F(ConfigLoadTest, StagingHonored) {
  setenv("GPU_CR_STAGING_MB", "256", 1);
  EXPECT_EQ(Load().staging_size, 256UL << 20);
}

TEST_F(ConfigLoadTest, StagingBelowFloorFallsBackToDefault) {
  setenv("GPU_CR_STAGING_MB", "64", 1);
  EXPECT_EQ(Load().staging_size, kStg1);
}

TEST_F(ConfigLoadTest, StagingNoUpperBound) {
  setenv("GPU_CR_STAGING_MB", "2048", 1);
  EXPECT_EQ(Load().staging_size, 2048UL << 20);
}

// Legacy names: never honored (values ignored), only warned about.
TEST_F(ConfigLoadTest, LegacyNamesIgnored) {
  setenv("GPUCR_SHM_GB", "60", 1);
  EXPECT_EQ(Load().shm_size, kShm8);
}

TEST_F(ConfigLoadTest, ParseNonNegativeRejectsTrailingGarbage) {
  bool ok = true;
  internal::ParseNonNegative("12abc", &ok);
  EXPECT_FALSE(ok);
  EXPECT_EQ(internal::ParseNonNegative("12", &ok), 12);
  EXPECT_TRUE(ok);
  internal::ParseNonNegative("", &ok);
  EXPECT_FALSE(ok);
  internal::ParseNonNegative("-1", &ok);
  EXPECT_FALSE(ok);
}

TEST(AlignUp2MBTest, Boundaries) {
  constexpr size_t k2MB = 2UL << 20;
  EXPECT_EQ(AlignUp2MB(0), 0u);
  EXPECT_EQ(AlignUp2MB(1), k2MB);
  EXPECT_EQ(AlignUp2MB(k2MB), k2MB);
  EXPECT_EQ(AlignUp2MB(k2MB + 1), 2 * k2MB);
}

}  // namespace
}  // namespace gpu_cr
