#include <cuda_runtime.h>
#include <nccl.h>

#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(10); \
  } \
} while (0)

#define CHECK_NCCL(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    std::fprintf(stderr, "[rank %d] NCCL error %s:%d: %s\n", rank, __FILE__, __LINE__, ncclGetErrorString(r)); \
    std::exit(11); \
  } \
} while (0)

struct SharedState {
  pthread_barrier_t barrier;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  ncclUniqueId id;
  int status[2];
  int preparedCount;
  int proceedRestore;
  char exportPath[2][256];
};

enum class TestMode {
  Native,
  Gcr,
  GcrCheckpointAction,
  GcrCheckpointToggle,
};

static void barrier_wait(SharedState* shared, int rank, const char* label) {
  int ret = pthread_barrier_wait(&shared->barrier);
  if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD) {
    std::fprintf(stderr, "[rank %d] barrier failed at %s: %d\n", rank, label, ret);
    std::exit(12);
  }
}

static void run_allreduce_check(int rank, ncclComm_t comm, const char* label) {
  const int count = 1024;
  float* send = nullptr;
  float* recv = nullptr;
  float host[count];

  CHECK_CUDA(cudaMalloc(&send, count * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&recv, count * sizeof(float)));

  for (int i = 0; i < count; ++i) host[i] = static_cast<float>(rank + 1);
  CHECK_CUDA(cudaMemcpy(send, host, count * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(recv, 0, count * sizeof(float)));

  CHECK_NCCL(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, 0));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host, recv, count * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < count; ++i) {
    if (std::fabs(host[i] - 3.0f) > 1e-5f) {
      std::fprintf(stderr, "[rank %d] %s mismatch at %d: got %.6f expected 3.000000\n",
                   rank, label, i, host[i]);
      std::exit(13);
    }
  }

  CHECK_CUDA(cudaFree(send));
  CHECK_CUDA(cudaFree(recv));
  std::fprintf(stderr, "[rank %d] %s allreduce OK\n", rank, label);
}

static bool mode_uses_gcr(TestMode mode) {
  return mode == TestMode::Gcr ||
         mode == TestMode::GcrCheckpointAction ||
         mode == TestMode::GcrCheckpointToggle;
}

static bool mode_uses_external_checkpoint(TestMode mode) {
  return mode == TestMode::GcrCheckpointAction ||
         mode == TestMode::GcrCheckpointToggle;
}

static const char* mode_name(TestMode mode) {
  switch (mode) {
    case TestMode::Native: return "native";
    case TestMode::Gcr: return "gcr";
    case TestMode::GcrCheckpointAction: return "gcr-ckpt-action";
    case TestMode::GcrCheckpointToggle: return "gcr-ckpt-toggle";
  }
  return "unknown";
}

static void notify_prepared_and_wait_restore(SharedState* shared, int rank) {
  int ret = pthread_mutex_lock(&shared->mutex);
  if (ret != 0) {
    std::fprintf(stderr, "[rank %d] mutex lock failed after prepare: %d\n", rank, ret);
    std::exit(14);
  }

  shared->preparedCount++;
  std::fprintf(stderr, "[rank %d] preparedCount=%d\n", rank, shared->preparedCount);
  pthread_cond_broadcast(&shared->cond);

  while (!shared->proceedRestore) {
    ret = pthread_cond_wait(&shared->cond, &shared->mutex);
    if (ret != 0) {
      std::fprintf(stderr, "[rank %d] cond wait failed after prepare: %d\n", rank, ret);
      std::exit(15);
    }
  }

  pthread_mutex_unlock(&shared->mutex);
  std::fprintf(stderr, "[rank %d] external checkpoint phase complete; starting restore\n", rank);
}

static int run_cuda_checkpoint_action(const char* ckptBin, pid_t pid, const char* action) {
  std::fprintf(stderr, "[parent] cuda-checkpoint --action %s --pid %d\n", action, static_cast<int>(pid));
  pid_t child = fork();
  if (child == 0) {
    unsetenv("LD_PRELOAD");
    unsetenv("NCCL_CHECKPOINT_PLUGIN");
    char pidBuf[32];
    std::snprintf(pidBuf, sizeof(pidBuf), "%d", static_cast<int>(pid));
    execl(ckptBin, ckptBin, "--action", action, "--pid", pidBuf, static_cast<char*>(nullptr));
    std::fprintf(stderr, "[parent] execl failed for cuda-checkpoint action %s pid %d: %s\n",
                 action, static_cast<int>(pid), std::strerror(errno));
    std::_Exit(127);
  }
  if (child < 0) {
    std::fprintf(stderr, "[parent] fork failed for cuda-checkpoint action %s pid %d: %s\n",
                 action, static_cast<int>(pid), std::strerror(errno));
    return 1;
  }

  int status = 0;
  if (waitpid(child, &status, 0) < 0) {
    std::fprintf(stderr, "[parent] waitpid failed for cuda-checkpoint action %s pid %d: %s\n",
                 action, static_cast<int>(pid), std::strerror(errno));
    return 1;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "[parent] cuda-checkpoint action %s pid %d failed, status=0x%x\n",
                 action, static_cast<int>(pid), status);
    return 1;
  }
  return 0;
}

static int run_cuda_checkpoint_toggle(const char* ckptBin, pid_t pid, const char* phase) {
  std::fprintf(stderr, "[parent] cuda-checkpoint --toggle --pid %d (%s)\n", static_cast<int>(pid), phase);
  pid_t child = fork();
  if (child == 0) {
    unsetenv("LD_PRELOAD");
    unsetenv("NCCL_CHECKPOINT_PLUGIN");
    char pidBuf[32];
    std::snprintf(pidBuf, sizeof(pidBuf), "%d", static_cast<int>(pid));
    execl(ckptBin, ckptBin, "--toggle", "--pid", pidBuf, static_cast<char*>(nullptr));
    std::fprintf(stderr, "[parent] execl failed for cuda-checkpoint toggle pid %d: %s\n",
                 static_cast<int>(pid), std::strerror(errno));
    std::_Exit(127);
  }
  if (child < 0) {
    std::fprintf(stderr, "[parent] fork failed for cuda-checkpoint toggle pid %d: %s\n",
                 static_cast<int>(pid), std::strerror(errno));
    return 1;
  }

  int status = 0;
  if (waitpid(child, &status, 0) < 0) {
    std::fprintf(stderr, "[parent] waitpid failed for cuda-checkpoint toggle pid %d: %s\n",
                 static_cast<int>(pid), std::strerror(errno));
    return 1;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "[parent] cuda-checkpoint toggle pid %d failed, status=0x%x\n",
                 static_cast<int>(pid), status);
    return 1;
  }
  return 0;
}

static int run_external_checkpoint(TestMode mode, const char* ckptBin, pid_t* pids) {
  if (mode == TestMode::GcrCheckpointAction) {
    const char* freezeActions[] = {"lock", "checkpoint"};
    const char* restoreActions[] = {"restore", "unlock"};
    for (const char* action : freezeActions) {
      for (int rank = 0; rank < 2; ++rank) {
        if (run_cuda_checkpoint_action(ckptBin, pids[rank], action) != 0) return 1;
      }
    }
    for (const char* action : restoreActions) {
      for (int rank = 0; rank < 2; ++rank) {
        if (run_cuda_checkpoint_action(ckptBin, pids[rank], action) != 0) return 1;
      }
    }
    return 0;
  }

  if (mode == TestMode::GcrCheckpointToggle) {
    for (int rank = 0; rank < 2; ++rank) {
      if (run_cuda_checkpoint_toggle(ckptBin, pids[rank], "freeze") != 0) return 1;
    }
    for (int rank = 0; rank < 2; ++rank) {
      if (run_cuda_checkpoint_toggle(ckptBin, pids[rank], "restore") != 0) return 1;
    }
    return 0;
  }

  return 0;
}

static void parent_wait_for_prepared(SharedState* shared) {
  int ret = pthread_mutex_lock(&shared->mutex);
  if (ret != 0) {
    std::fprintf(stderr, "[parent] mutex lock failed while waiting prepared: %d\n", ret);
    std::exit(16);
  }
  while (shared->preparedCount < 2) {
    ret = pthread_cond_wait(&shared->cond, &shared->mutex);
    if (ret != 0) {
      std::fprintf(stderr, "[parent] cond wait failed while waiting prepared: %d\n", ret);
      std::exit(17);
    }
  }
  pthread_mutex_unlock(&shared->mutex);
  std::fprintf(stderr, "[parent] both ranks prepared; running external cuda-checkpoint phase\n");
}

static void parent_release_restore(SharedState* shared) {
  int ret = pthread_mutex_lock(&shared->mutex);
  if (ret != 0) {
    std::fprintf(stderr, "[parent] mutex lock failed while releasing restore: %d\n", ret);
    std::exit(18);
  }
  shared->proceedRestore = 1;
  pthread_cond_broadcast(&shared->cond);
  pthread_mutex_unlock(&shared->mutex);
  std::fprintf(stderr, "[parent] restore phase released\n");
}

static int child_main(SharedState* shared, int rank, TestMode mode) {
  const bool gcrMode = mode_uses_gcr(mode);
  const bool externalCheckpoint = mode_uses_external_checkpoint(mode);
  CHECK_CUDA(cudaSetDevice(rank));

  ncclComm_t comm = nullptr;
  CHECK_NCCL(ncclCommInitRank(&comm, 2, shared->id, rank));
  std::fprintf(stderr, "[rank %d] comm initialized\n", rank);

  run_allreduce_check(rank, comm, "before checkpoint");
  barrier_wait(shared, rank, "before prepare");

  if (gcrMode) {
    CHECK_NCCL(ncclCommCheckpointPrepare(comm, NCCL_CKPT_MODE_GCR_GLOBAL));
  } else {
    CHECK_NCCL(ncclCommCheckpointPrepare(comm, NCCL_CKPT_MODE_NATIVE));
  }
  std::fprintf(stderr, "[rank %d] prepare OK\n", rank);

  if (externalCheckpoint) {
    notify_prepared_and_wait_restore(shared, rank);
  } else {
    barrier_wait(shared, rank, "after prepare");
  }

  if (gcrMode) {
    setenv("GCR_EXPORT_SHM_PATH", shared->exportPath[rank], 1);
    CHECK_NCCL(ncclCommCheckpointRestore(comm, NCCL_CKPT_MODE_GCR_GLOBAL | NCCL_CKPT_RESTORE_EXPORT));
    std::fprintf(stderr, "[rank %d] restore export OK\n", rank);

    barrier_wait(shared, rank, "after restore export");

    setenv("GCR_IMPORT_SHM_PATH", shared->exportPath[1 - rank], 1);
    CHECK_NCCL(ncclCommCheckpointRestore(comm, NCCL_CKPT_MODE_GCR_GLOBAL | NCCL_CKPT_RESTORE_IMPORT | NCCL_CKPT_OPT_VALIDATE));
    std::fprintf(stderr, "[rank %d] restore import OK\n", rank);
  } else {
    CHECK_NCCL(ncclCommCheckpointRestore(comm, NCCL_CKPT_MODE_NATIVE));
    std::fprintf(stderr, "[rank %d] restore OK\n", rank);
  }

  barrier_wait(shared, rank, "after restore");
  run_allreduce_check(rank, comm, "after checkpoint");

  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_CUDA(cudaDeviceReset());
  shared->status[rank] = 0;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s native|gcr|gcr-ckpt-action|gcr-ckpt-toggle [cuda-checkpoint-bin]\n", argv[0]);
    return 2;
  }

  TestMode mode;
  if (std::strcmp(argv[1], "native") == 0) {
    mode = TestMode::Native;
  } else if (std::strcmp(argv[1], "gcr") == 0) {
    mode = TestMode::Gcr;
  } else if (std::strcmp(argv[1], "gcr-ckpt-action") == 0) {
    mode = TestMode::GcrCheckpointAction;
  } else if (std::strcmp(argv[1], "gcr-ckpt-toggle") == 0) {
    mode = TestMode::GcrCheckpointToggle;
  } else {
    std::fprintf(stderr, "usage: %s native|gcr|gcr-ckpt-action|gcr-ckpt-toggle [cuda-checkpoint-bin]\n", argv[0]);
    return 2;
  }

  const char* ckptBin = nullptr;
  if (mode_uses_external_checkpoint(mode)) {
    if (argc != 3) {
      std::fprintf(stderr, "mode %s requires cuda-checkpoint binary path\n", mode_name(mode));
      return 2;
    }
    ckptBin = argv[2];
    if (access(ckptBin, X_OK) != 0) {
      std::fprintf(stderr, "cuda-checkpoint binary is not executable: %s (%s)\n", ckptBin, std::strerror(errno));
      return 2;
    }
  }

  SharedState* shared = reinterpret_cast<SharedState*>(
      mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (shared == MAP_FAILED) {
    std::perror("mmap");
    return 3;
  }
  std::memset(shared, 0, sizeof(*shared));
  shared->status[0] = shared->status[1] = -1;

  pthread_barrierattr_t attr;
  pthread_barrierattr_init(&attr);
  pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_barrier_init(&shared->barrier, &attr, 2);
  pthread_barrierattr_destroy(&attr);

  pthread_mutexattr_t mutexAttr;
  pthread_mutexattr_init(&mutexAttr);
  pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_init(&shared->mutex, &mutexAttr);
  pthread_mutexattr_destroy(&mutexAttr);

  pthread_condattr_t condAttr;
  pthread_condattr_init(&condAttr);
  pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
  pthread_cond_init(&shared->cond, &condAttr);
  pthread_condattr_destroy(&condAttr);

  std::snprintf(shared->exportPath[0], sizeof(shared->exportPath[0]), "/tmp/gcr_nccl_stage2_%d_rank0.shm", getpid());
  std::snprintf(shared->exportPath[1], sizeof(shared->exportPath[1]), "/tmp/gcr_nccl_stage2_%d_rank1.shm", getpid());

  ncclResult_t idResult = ncclGetUniqueId(&shared->id);
  if (idResult != ncclSuccess) {
    std::fprintf(stderr, "ncclGetUniqueId failed: %s\n", ncclGetErrorString(idResult));
    return 4;
  }

  pid_t pids[2];
  for (int rank = 0; rank < 2; ++rank) {
    pids[rank] = fork();
    if (pids[rank] == 0) {
      return child_main(shared, rank, mode);
    }
    if (pids[rank] < 0) {
      std::perror("fork");
      return 5;
    }
  }

  int externalFailures = 0;
  if (mode_uses_external_checkpoint(mode)) {
    parent_wait_for_prepared(shared);
    int ckptRc = run_external_checkpoint(mode, ckptBin, pids);
    if (ckptRc != 0) {
      std::fprintf(stderr, "[parent] external cuda-checkpoint phase failed\n");
      externalFailures++;
      parent_release_restore(shared);
    } else {
      std::fprintf(stderr, "[parent] external cuda-checkpoint phase OK\n");
      parent_release_restore(shared);
    }
  }

  int failures = externalFailures;
  for (int i = 0; i < 2; ++i) {
    int status = 0;
    waitpid(pids[i], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::fprintf(stderr, "child %d failed, status=0x%x\n", i, status);
      failures++;
    }
  }

  pthread_barrier_destroy(&shared->barrier);
  pthread_mutex_destroy(&shared->mutex);
  pthread_cond_destroy(&shared->cond);
  munmap(shared, sizeof(*shared));

  if (failures) {
    std::fprintf(stderr, "TEST FAILED: %d child process(es) failed\n", failures);
    return 1;
  }

  std::fprintf(stderr, "TEST PASSED: %s mode\n", mode_name(mode));
  return 0;
}
