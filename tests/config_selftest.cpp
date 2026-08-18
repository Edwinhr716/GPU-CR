// KEP-0002 unit matrix: env parsing, bounds, defaults, legacy warnings.
// Host-only (no CUDA); exercises gpu_cr::internal::Load() directly so
// each case gets a fresh parse (the production singleton caches, by design).
// Run by Dockerfile.build after compilation; nonzero exit fails the build.
#include "../src/gpu_cr_config.h"

#include <assert.h>

// Satisfy the declaration in the header; unused here (tests call load()).
namespace gpu_cr {
const BufConfig& Config() {
    static BufConfig cfg = internal::Load(25UL << 30, 1UL << 30);
    return cfg;
}
}  // namespace gpu_cr

static constexpr size_t SHM8 = 8UL << 30;   // an "SHM8 fleet build" default
static constexpr size_t STG1 = 1UL << 30;

static void clear_env() {
    unsetenv("GPU_CR_SHM_GB");
    unsetenv("GPU_CR_SHM_MB");
    unsetenv("GPU_CR_STAGING_MB");
    unsetenv("GPUCR_SHM_GB");
    unsetenv("GPUCR_STAGING_MB");
}

static int failures = 0;
#define CHECK(cond, name) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", name); failures++; } \
    else { fprintf(stderr, "ok:   %s\n", name); } \
} while (0)

int main() {
    // Fleet-default rule (F1): env unset => the BUILD's default, exactly.
    clear_env();
    gpu_cr::BufConfig c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == SHM8 && c.staging_size == STG1 && !c.shm_deferred, "unset -> build defaults");

    // GB and MB paths; MB wins when both set.
    clear_env(); setenv("GPU_CR_SHM_GB", "12", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == (12UL << 30), "GPU_CR_SHM_GB=12");
    clear_env(); setenv("GPU_CR_SHM_MB", "300", 1); setenv("GPU_CR_SHM_GB", "12", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == (300UL << 20), "GPU_CR_SHM_MB=300 wins over _GB");

    // No upper clamp (F5): values above the old 25GiB literal are honored.
    clear_env(); setenv("GPU_CR_SHM_GB", "40", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == (40UL << 30), "no upper clamp: 40GiB honored");

    // Floor (F5): below 64MiB -> default with warning, never a clamp.
    clear_env(); setenv("GPU_CR_SHM_MB", "10", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == SHM8 && !c.shm_deferred, "below-floor SHM -> default");

    // Deferred mode (=0) stays special: not subsumed by the floor.
    clear_env(); setenv("GPU_CR_SHM_GB", "0", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_deferred && c.shm_size == gpu_cr::kShmFloorBytes, "SHM=0 -> deferred at 64MiB floor");

    // Unparsable -> default.
    clear_env(); setenv("GPU_CR_SHM_GB", "8x", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == SHM8 && !c.shm_deferred, "unparsable -> default");
    clear_env(); setenv("GPU_CR_SHM_GB", "-3", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == SHM8, "negative -> default");

    // 2MiB alignment.
    clear_env(); setenv("GPU_CR_SHM_MB", "129", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == (130UL << 20), "129MiB aligns up to 130MiB");

    // Staging: floor 128MiB, no upper bound, unparsable -> default.
    clear_env(); setenv("GPU_CR_STAGING_MB", "256", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.staging_size == (256UL << 20), "staging 256MiB");
    clear_env(); setenv("GPU_CR_STAGING_MB", "64", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.staging_size == STG1, "staging below floor -> default");
    clear_env(); setenv("GPU_CR_STAGING_MB", "2048", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.staging_size == (2048UL << 20), "staging 2GiB honored (no upper bound)");

    // Legacy names: never honored (values ignored), only warned about.
    clear_env(); setenv("GPUCR_SHM_GB", "60", 1);
    c = gpu_cr::internal::Load(SHM8, STG1);
    CHECK(c.shm_size == SHM8, "legacy GPUCR_SHM_GB ignored");

    if (failures) {
        fprintf(stderr, "config_selftest: %d FAILURES\n", failures);
        return 1;
    }
    fprintf(stderr, "config_selftest: all checks passed\n");
    return 0;
}
