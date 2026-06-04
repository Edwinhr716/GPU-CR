#include "gcr_checkpoint.h"
#include "ipc_fd_exchange.h"
#include "ipc_hooks.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#define GCR_CKPT_OPT_VALIDATE 0x0400

static std::mutex g_state_mutex;
static bool g_prepared = false;

struct StorageBuffer {
  void* ptr = nullptr;
  size_t size = 0;
  int fd = -1;
  bool pinned = false;
  char path[512] = {};
};

static StorageBuffer g_export_data;
static StorageBuffer g_local_alloc_data;
static int g_storage_file_seq = 0;

static long elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

static const char* storage_backend() {
  const char* backend = std::getenv("GCR_STORAGE_BACKEND");
  if (!backend || !backend[0]) return "anon";
  if (std::strcmp(backend, "anon") == 0 ||
      std::strcmp(backend, "pinned") == 0 ||
      std::strcmp(backend, "hugepage") == 0) {
    return backend;
  }
  std::fprintf(stderr, "[GCR-RUNTIME] WARNING: unknown GCR_STORAGE_BACKEND=%s, using anon\n",
               backend);
  return "anon";
}

static const char* storage_dir() {
  const char* dir = std::getenv("GCR_STORAGE_DIR");
  if (dir && dir[0]) return dir;
  return "/mnt/huge-ckpt/nccl-gcr";
}

static void release_buffer(StorageBuffer* buffer) {
  if (buffer->ptr && buffer->size) {
    if (buffer->pinned) {
      cudaError_t err = cudaFreeHost(buffer->ptr);
      if (err != cudaSuccess) {
        std::fprintf(stderr, "[GCR-RUNTIME] WARNING: cudaFreeHost failed: %s\n",
                     cudaGetErrorString(err));
        cudaGetLastError();
      }
    } else {
      munmap(buffer->ptr, buffer->size);
    }
  }
  if (buffer->fd >= 0) close(buffer->fd);
  if (buffer->path[0]) unlink(buffer->path);
  buffer->ptr = nullptr;
  buffer->size = 0;
  buffer->fd = -1;
  buffer->pinned = false;
  buffer->path[0] = '\0';
}

static int ensure_storage_dir(const char* dir) {
  if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: mkdir(%s) failed: %s\n",
                 dir, std::strerror(errno));
    return -1;
  }
  return 0;
}

static int allocate_buffer(StorageBuffer* buffer, size_t needed, const char* name) {
  release_buffer(buffer);
  if (needed == 0) return 0;

  const char* backend = storage_backend();
  if (std::strcmp(backend, "pinned") == 0) {
    cudaError_t err = cudaHostAlloc(&buffer->ptr, needed, cudaHostAllocPortable);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[GCR-RUNTIME] ERROR: cudaHostAlloc for %s buffer failed: %s\n",
                   name, cudaGetErrorString(err));
      cudaGetLastError();
      return -1;
    }
    buffer->size = needed;
    buffer->pinned = true;
    return 0;
  }

  int fd = -1;
  int mmapFlags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (std::strcmp(backend, "hugepage") == 0) {
    const char* dir = storage_dir();
    if (ensure_storage_dir(dir) != 0) return -1;
    std::snprintf(buffer->path, sizeof(buffer->path), "%s/%s-%d-%d.bin",
                  dir, name, static_cast<int>(getpid()), g_storage_file_seq++);
    fd = open(buffer->path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
      std::fprintf(stderr, "[GCR-RUNTIME] ERROR: open(%s) failed: %s\n",
                   buffer->path, std::strerror(errno));
      buffer->path[0] = '\0';
      return -1;
    }
    if (ftruncate(fd, needed) != 0) {
      std::fprintf(stderr, "[GCR-RUNTIME] ERROR: ftruncate(%s) failed: %s\n",
                   buffer->path, std::strerror(errno));
      close(fd);
      unlink(buffer->path);
      buffer->path[0] = '\0';
      return -1;
    }
    mmapFlags = MAP_SHARED;
  }

  void* ptr = mmap(nullptr, needed, PROT_READ | PROT_WRITE, mmapFlags, fd, 0);
  if (ptr == MAP_FAILED) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: mmap for %s buffer failed: %s\n",
                 name, std::strerror(errno));
    if (fd >= 0) {
      close(fd);
      unlink(buffer->path);
    }
    buffer->path[0] = '\0';
    return -1;
  }

  buffer->ptr = ptr;
  buffer->size = needed;
  buffer->fd = fd;
  return 0;
}

static int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !value[0]) return fallback;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0') return fallback;
  return static_cast<int>(parsed);
}

static double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (!value || !value[0]) return fallback;
  char* end = nullptr;
  double parsed = std::strtod(value, &end);
  if (!end || *end != '\0') return fallback;
  return parsed;
}

static void append_runtime_metric(const char* phase, size_t exportBytes, size_t localBytes,
                                  int imports, int exports, int events, int locals, int peers,
                                  const IpcTimingSnapshot& timing, long totalMs) {
  const char* path = std::getenv("GCR_IPC_SCALING_METRICS_CSV");
  if (!path || !path[0]) return;

  int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0666);
  if (fd < 0) {
    std::fprintf(stderr, "[GCR-RUNTIME] WARNING: open metrics csv %s failed: %s\n",
                 path, std::strerror(errno));
    return;
  }

  if (flock(fd, LOCK_EX) != 0) {
    std::fprintf(stderr, "[GCR-RUNTIME] WARNING: flock metrics csv %s failed: %s\n",
                 path, std::strerror(errno));
    close(fd);
    return;
  }

  struct stat st;
  if (fstat(fd, &st) == 0 && st.st_size == 0) {
    dprintf(fd,
            "mode,phase,pid,rank,case_idx,buffer_gb,export_bytes,local_bytes,"
            "imports,exports,events,locals,peers,"
            "import_teardown_ms,import_teardown_count,import_teardown_bytes,"
            "export_teardown_dtoh_ms,export_teardown_fd_close_ms,"
            "export_teardown_unmap_release_ms,export_teardown_count,export_teardown_bytes,"
            "local_teardown_dtoh_ms,local_teardown_unmap_release_ms,"
            "local_teardown_count,local_teardown_bytes,"
            "export_rebuild_alloc_map_access_ms,export_rebuild_htod_ms,"
            "export_rebuild_reexport_ms,export_rebuild_total_ms,"
            "export_rebuild_count,export_rebuild_bytes,"
            "local_rebuild_alloc_map_access_ms,local_rebuild_htod_ms,"
            "local_rebuild_total_ms,local_rebuild_count,local_rebuild_bytes,"
            "import_rebuild_ms,import_rebuild_count,import_rebuild_bytes,"
            "storage_backend,storage_dir,total_ms\n");
  }

  const char* mode = std::getenv("GCR_IPC_SCALING_MODE");
  if (!mode || !mode[0]) mode = "unknown";
  dprintf(fd,
          "%s,%s,%d,%d,%d,%.6f,%zu,%zu,%d,%d,%d,%d,%d,"
          "%ld,%d,%zu,%ld,%ld,%ld,%d,%zu,%ld,%ld,%d,%zu,"
          "%ld,%ld,%ld,%ld,%d,%zu,%ld,%ld,%ld,%d,%zu,%ld,%d,%zu,%s,%s,%ld\n",
          mode, phase, static_cast<int>(getpid()),
          env_int("GCR_IPC_SCALING_RANK", -1),
          env_int("GCR_IPC_SCALING_CASE", -1),
          env_double("GCR_IPC_SCALING_BUFFER_GB", 0.0),
          exportBytes, localBytes, imports, exports, events, locals, peers,
          timing.import_teardown_ms, timing.import_teardown_count, timing.import_teardown_bytes,
          timing.export_teardown_dtoh_ms, timing.export_teardown_fd_close_ms,
          timing.export_teardown_unmap_release_ms, timing.export_teardown_count,
          timing.export_teardown_bytes,
          timing.local_teardown_dtoh_ms, timing.local_teardown_unmap_release_ms,
          timing.local_teardown_count, timing.local_teardown_bytes,
          timing.export_rebuild_alloc_map_access_ms, timing.export_rebuild_htod_ms,
          timing.export_rebuild_reexport_ms, timing.export_rebuild_total_ms,
          timing.export_rebuild_count, timing.export_rebuild_bytes,
          timing.local_rebuild_alloc_map_access_ms, timing.local_rebuild_htod_ms,
          timing.local_rebuild_total_ms, timing.local_rebuild_count, timing.local_rebuild_bytes,
          timing.import_rebuild_ms, timing.import_rebuild_count, timing.import_rebuild_bytes,
          storage_backend(), storage_dir(),
          totalMs);

  flock(fd, LOCK_UN);
  close(fd);
}

static int map_shm_block(const char* path, bool create, IpcRebuildShmBlock** block, int* fdOut) {
  int flags = create ? (O_RDWR | O_CREAT) : O_RDWR;
  int fd = open(path, flags, 0666);
  if (fd < 0) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: open(%s) failed: %s\n", path, std::strerror(errno));
    return -1;
  }

  if (create && ftruncate(fd, sizeof(IpcRebuildShmBlock)) != 0) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: ftruncate(%s) failed: %s\n", path, std::strerror(errno));
    close(fd);
    return -1;
  }

  void* ptr = mmap(nullptr, sizeof(IpcRebuildShmBlock), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: mmap(%s) failed: %s\n", path, std::strerror(errno));
    close(fd);
    return -1;
  }

  *block = reinterpret_cast<IpcRebuildShmBlock*>(ptr);
  *fdOut = fd;
  return 0;
}

static void unmap_shm_block(IpcRebuildShmBlock* block, int fd) {
  if (block) munmap(block, sizeof(IpcRebuildShmBlock));
  if (fd >= 0) close(fd);
}

extern "C" int gcr_checkpoint_prepare(int flags) {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (g_prepared) {
    std::fprintf(stderr, "[GCR-RUNTIME] prepare already completed for this process\n");
    return 0;
  }

  auto phaseStart = std::chrono::high_resolution_clock::now();
  std::fprintf(stderr, "[GCR-RUNTIME] === checkpoint prepare === flags=0x%x imports=%d exports=%d events=%d\n",
               flags, ipc_get_import_count(), ipc_get_export_count(), ipc_get_event_count());

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(syncErr));
    cudaGetLastError();
    return -1;
  }

  ipc_dump_state();
  ipc_dump_nvidia_fds("BEFORE GCR prepare");

  ipc_reset_timing_snapshot();
  int imports = ipc_teardown_all_imports();
  if (imports < 0) return -1;

  size_t exportNeeded = ipc_get_export_data_size();
  if (allocate_buffer(&g_export_data, exportNeeded, "export") != 0) return -1;

  int exports = 0;
  if (g_export_data.ptr && g_export_data.size) {
    exports = ipc_save_and_teardown_all_exports(g_export_data.ptr, g_export_data.size);
    if (exports < 0) return -1;
  }

  int events = ipc_teardown_all_events();
  if (events < 0) return -1;

  size_t localNeeded = ipc_get_local_alloc_data_size();
  std::fprintf(stderr, "[GCR-RUNTIME] prepare resource sizes: export_bytes=%zu local_bytes=%zu imports=%d exports=%d events=%d\n",
               exportNeeded, localNeeded, imports, exports, events);
  if (allocate_buffer(&g_local_alloc_data, localNeeded, "local_cumem") != 0) return -1;

  int locals = 0;
  if (g_local_alloc_data.ptr && g_local_alloc_data.size) {
    locals = ipc_save_and_teardown_local_allocs(g_local_alloc_data.ptr, g_local_alloc_data.size);
    if (locals < 0) return -1;
  }

  int peers = ipc_disable_all_peer_access();

  ipc_dump_nvidia_fds("AFTER GCR prepare");
  g_prepared = true;

  IpcTimingSnapshot timing;
  ipc_get_timing_snapshot(&timing);
  long totalMs = elapsed_ms(phaseStart);
  append_runtime_metric("prepare", exportNeeded, localNeeded, imports, exports, events,
                        locals, peers, timing, totalMs);
  std::fprintf(stderr,
               "[GCR-RUNTIME] prepare done: imports=%d exports=%d events=%d locals=%d peers=%d total=%ld ms\n",
               imports, exports, events, locals, peers, totalMs);
  if (imports == 0 && exports == 0 && locals == 0) {
    std::fprintf(stderr,
                 "[GCR-RUNTIME] WARNING: nothing was tracked — NCCL likely shared its "
                 "buffers via legacy CUDA IPC instead of cuMem*. A subsequent "
                 "cuda-checkpoint restore will probably fail with \"operation not "
                 "supported\". Export NCCL_CUMEM_ENABLE=1 before starting the "
                 "application (vLLM-style deployments additionally use "
                 "NCCL_CUMEM_HOST_ENABLE=1 NCCL_P2P_DISABLE=1).\n");
  }
  return 0;
}

extern "C" int gcr_checkpoint_restore_export(int flags) {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  auto phaseStart = std::chrono::high_resolution_clock::now();
  std::fprintf(stderr, "[GCR-RUNTIME] === restore export === flags=0x%x\n", flags);

  if (!g_prepared) {
    std::fprintf(stderr, "[GCR-RUNTIME] WARNING: restore_export called before prepare\n");
  }

  ipc_reset_timing_snapshot();
  int exports = 0;
  if (g_export_data.ptr && g_export_data.size) {
    exports = ipc_rebuild_and_restore_all_exports(g_export_data.ptr, g_export_data.size);
    release_buffer(&g_export_data);
    if (exports < 0) return -1;
  }

  int locals = 0;
  if (g_local_alloc_data.ptr && g_local_alloc_data.size) {
    locals = ipc_rebuild_local_allocs(g_local_alloc_data.ptr, g_local_alloc_data.size);
    release_buffer(&g_local_alloc_data);
    if (locals < 0) return -1;
  }

  const char* exportPath = std::getenv("GCR_EXPORT_SHM_PATH");
  if (exportPath && exportPath[0]) {
    IpcRebuildShmBlock* block = nullptr;
    int fd = -1;
    if (map_shm_block(exportPath, true, &block, &fd) != 0) return -1;
    int written = ipc_write_export_info_to_shm(block);
    unmap_shm_block(block, fd);
    if (written < 0) return -1;
    std::fprintf(stderr, "[GCR-RUNTIME] wrote export info to %s entries=%d\n", exportPath, written);
  } else if (ipc_get_export_count() > 0) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: GCR_EXPORT_SHM_PATH is required when exports exist\n");
    return -1;
  }

  if (uds_fd_server_start() != 0) return -1;

  IpcTimingSnapshot timing;
  ipc_get_timing_snapshot(&timing);
  long totalMs = elapsed_ms(phaseStart);
  append_runtime_metric("restore_export", timing.export_rebuild_bytes,
                        timing.local_rebuild_bytes, 0, exports, 0, locals, 0,
                        timing, totalMs);
  std::fprintf(stderr, "[GCR-RUNTIME] restore export done: exports=%d locals=%d total=%ld ms\n",
               exports, locals, totalMs);
  return 0;
}

extern "C" int gcr_checkpoint_restore_import(int flags) {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  auto phaseStart = std::chrono::high_resolution_clock::now();
  std::fprintf(stderr, "[GCR-RUNTIME] === restore import === flags=0x%x\n", flags);

  ipc_reset_timing_snapshot();
  int imported = 0;
  const char* importPath = std::getenv("GCR_IMPORT_SHM_PATH");
  if (importPath && importPath[0]) {
    IpcRebuildShmBlock* block = nullptr;
    int fd = -1;
    if (map_shm_block(importPath, false, &block, &fd) != 0) return -1;
    imported = ipc_import_from_shm_block(block);
    unmap_shm_block(block, fd);
    if (imported < 0) return -1;
  } else if (ipc_get_import_count() > 0) {
    std::fprintf(stderr, "[GCR-RUNTIME] ERROR: GCR_IMPORT_SHM_PATH is required when imports exist\n");
    return -1;
  }

  if (env_int("GCR_STOP_FD_SERVER_AFTER_IMPORT", 0) != 0) {
    uds_fd_server_stop();
  } else {
    std::fprintf(stderr, "[GCR-RUNTIME] keeping FD server alive after import for peer restore overlap\n");
  }

  int validateErrors = 0;
  if (flags & GCR_CKPT_OPT_VALIDATE) {
    validateErrors = ipc_validate_all_mappings("AFTER GCR restore import");
    if (validateErrors != 0) return -1;
  }

  int peers = ipc_reenable_all_peer_access();

  g_prepared = false;
  IpcTimingSnapshot timing;
  ipc_get_timing_snapshot(&timing);
  long totalMs = elapsed_ms(phaseStart);
  append_runtime_metric("restore_import", 0, 0, imported, 0, 0, 0, peers, timing, totalMs);
  std::fprintf(stderr, "[GCR-RUNTIME] restore import done: imported=%d validate_errors=%d peers=%d total=%ld ms\n",
               imported, validateErrors, peers, totalMs);
  return 0;
}

extern "C" int gcr_checkpoint_validate(int flags) {
  (void)flags;
  return ipc_validate_all_mappings("GCR validate");
}
