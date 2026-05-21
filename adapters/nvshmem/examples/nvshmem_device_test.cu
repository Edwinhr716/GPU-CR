#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct Options {
  int rank = -1;
  int nranks = 2;
  size_t bytes = 64ULL * 1024ULL * 1024ULL;
  std::string control_dir;
  std::string unique_id_file;
};

using gcr_fn_t = int (*)(int);

#define CHECK_CUDA(cmd) do { \
  cudaError_t e__ = (cmd); \
  if (e__ != cudaSuccess) { \
    std::fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n", \
                 options.rank, __FILE__, __LINE__, cudaGetErrorString(e__)); \
    std::exit(30); \
  } \
} while (0)

__global__ void device_put_kernel(int* sym, int stage) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int pe = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (pe + 1) % npes;
  nvshmem_int_p(sym, stage * 1000 + pe, peer);
  nvshmem_quiet();
}

__global__ void device_get_kernel(const int* sym, int* out, int stage) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int pe = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (pe + 1) % npes;
  out[0] = nvshmem_int_g(sym, peer);
  out[1] = stage * 1000 + pe;
}

static bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static void write_text_file(const std::string& path, const char* text) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "failed to open %s: %s\n", path.c_str(), std::strerror(errno));
    std::exit(22);
  }
  std::fputs(text, f);
  std::fclose(f);
}

static void write_unique_id_file(const std::string& path, const nvshmemx_uniqueid_t& id) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "failed to open %s: %s\n", path.c_str(), std::strerror(errno));
    std::exit(22);
  }
  size_t written = std::fwrite(&id, 1, sizeof(id), f);
  std::fclose(f);
  if (written != sizeof(id)) {
    std::fprintf(stderr, "failed to write NVSHMEM unique id to %s\n", path.c_str());
    std::exit(22);
  }
}

static void read_unique_id_file(const std::string& path, nvshmemx_uniqueid_t* id) {
  for (int tries = 0; tries < 6000; ++tries) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && st.st_size >= static_cast<off_t>(sizeof(*id))) {
      FILE* f = std::fopen(path.c_str(), "rb");
      if (!f) {
        std::fprintf(stderr, "failed to open %s: %s\n", path.c_str(), std::strerror(errno));
        std::exit(22);
      }
      size_t got = std::fread(id, 1, sizeof(*id), f);
      std::fclose(f);
      if (got == sizeof(*id)) return;
      std::fprintf(stderr, "failed to read NVSHMEM unique id from %s\n", path.c_str());
      std::exit(22);
    }
    usleep(100000);
  }
  std::fprintf(stderr, "timed out waiting for NVSHMEM unique id file %s\n", path.c_str());
  std::exit(23);
}

static void wait_for_file(const std::string& path, const char* label, int rank) {
  for (int tries = 0; tries < 12000; ++tries) {
    if (file_exists(path)) return;
    usleep(100000);
  }
  std::fprintf(stderr, "[rank %d] timed out waiting for %s: %s\n",
               rank, label, path.c_str());
  std::exit(24);
}

static gcr_fn_t resolve_gcr_fn(const char* name, int rank) {
  void* sym = dlsym(RTLD_DEFAULT, name);
  if (!sym) {
    std::fprintf(stderr, "[rank %d] failed to resolve %s; is libgcr_preload.so preloaded?\n",
                 rank, name);
    std::exit(25);
  }
  return reinterpret_cast<gcr_fn_t>(sym);
}

static void call_gcr(const char* name, int rank) {
  gcr_fn_t fn = resolve_gcr_fn(name, rank);
  std::fprintf(stderr, "[rank %d] calling %s\n", rank, name);
  int ret = fn(0);
  if (ret != 0) {
    std::fprintf(stderr, "[rank %d] %s failed ret=%d\n", rank, name, ret);
    std::exit(26);
  }
  std::fprintf(stderr, "[rank %d] %s done\n", rank, name);
}

static void verify_sample(const Options& options, int* sym, size_t int_count,
                          int expected_peer_value, unsigned char pattern,
                          const char* label) {
  int got = 0;
  CHECK_CUDA(cudaMemcpy(&got, sym, sizeof(got), cudaMemcpyDeviceToHost));
  if (got != expected_peer_value) {
    std::fprintf(stderr, "[rank %d] %s peer value mismatch: got %d expected %d\n",
                 options.rank, label, got, expected_peer_value);
    std::exit(33);
  }

  if (int_count > 16) {
    unsigned char sample = 0;
    unsigned char* ptr = reinterpret_cast<unsigned char*>(sym) + options.bytes / 2;
    CHECK_CUDA(cudaMemcpy(&sample, ptr, sizeof(sample), cudaMemcpyDeviceToHost));
    if (sample != pattern) {
      std::fprintf(stderr, "[rank %d] %s middle pattern mismatch: got %u expected %u\n",
                   options.rank, label, static_cast<unsigned>(sample),
                   static_cast<unsigned>(pattern));
      std::exit(34);
    }
  }
}

static void run_nvshmem_device_exchange(const Options& options, int* sym,
                                        size_t int_count, int stage) {
  int peer = (options.rank + 1) % options.nranks;
  unsigned char pattern = static_cast<unsigned char>(0x50 + options.rank + stage);
  int empty = -1;
  int expected_put_from_peer = stage * 1000 + peer;
  int expected_get_from_peer = stage * 1000 + options.rank;

  CHECK_CUDA(cudaMemset(sym, pattern, options.bytes));
  CHECK_CUDA(cudaMemcpy(sym, &empty, sizeof(empty), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());

  nvshmem_barrier_all();
  {
    void* args[] = {&sym, &stage};
    int launch_ret = nvshmemx_collective_launch(
        reinterpret_cast<const void*>(device_put_kernel), dim3(1), dim3(1), args, 0, nullptr);
    if (launch_ret != 0) {
      std::fprintf(stderr, "[rank %d] nvshmemx_collective_launch(device_put) failed ret=%d\n",
                   options.rank, launch_ret);
      std::exit(36);
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  nvshmem_barrier_all();

  verify_sample(options, sym, int_count, expected_put_from_peer, pattern,
                "nvshmem_device_put");

  int* d_get = nullptr;
  int h_get[2] = {-1, -1};
  CHECK_CUDA(cudaMalloc(&d_get, sizeof(h_get)));
  CHECK_CUDA(cudaMemset(d_get, 0xff, sizeof(h_get)));
  {
    void* args[] = {&sym, &d_get, &stage};
    int launch_ret = nvshmemx_collective_launch(
        reinterpret_cast<const void*>(device_get_kernel), dim3(1), dim3(1), args, 0, nullptr);
    if (launch_ret != 0) {
      std::fprintf(stderr, "[rank %d] nvshmemx_collective_launch(device_get) failed ret=%d\n",
                   options.rank, launch_ret);
      std::exit(37);
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_get, d_get, sizeof(h_get), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_get));

  if (h_get[0] != expected_get_from_peer || h_get[1] != expected_get_from_peer) {
    std::fprintf(stderr,
                 "[rank %d] nvshmem_device_get mismatch: got remote=%d local_expected=%d expected=%d\n",
                 options.rank, h_get[0], h_get[1], expected_get_from_peer);
    std::exit(35);
  }

  std::fprintf(stderr,
               "[rank %d] NVSHMEM device put/get stage=%d ok put_value=%d get_value=%d\n",
               options.rank, stage, expected_put_from_peer, h_get[0]);
}

static bool parse_options(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
      options->rank = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--nranks") == 0 && i + 1 < argc) {
      options->nranks = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      char* end = nullptr;
      unsigned long long value = std::strtoull(argv[++i], &end, 10);
      if (!end || *end != '\0' || value == 0) return false;
      options->bytes = static_cast<size_t>(value);
    } else if (std::strcmp(argv[i], "--control-dir") == 0 && i + 1 < argc) {
      options->control_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--unique-id-file") == 0 && i + 1 < argc) {
      options->unique_id_file = argv[++i];
    } else {
      return false;
    }
  }
  return options->rank >= 0 &&
         options->rank < options->nranks &&
         options->nranks == 2 &&
         !options->control_dir.empty() &&
         !options->unique_id_file.empty() &&
         options->bytes >= sizeof(int);
}

static int run(const Options& options) {
  std::fprintf(stderr, "[rank %d] start pid=%d bytes=%zu mode=device-kernel\n",
               options.rank, static_cast<int>(getpid()), options.bytes);
  CHECK_CUDA(cudaSetDevice(options.rank));

  nvshmemx_uniqueid_t unique_id;
  if (options.rank == 0) {
    int ret = nvshmemx_get_uniqueid(&unique_id);
    if (ret != 0) {
      std::fprintf(stderr, "[rank %d] nvshmemx_get_uniqueid failed ret=%d\n",
                   options.rank, ret);
      return 40;
    }
    write_unique_id_file(options.unique_id_file, unique_id);
  } else {
    read_unique_id_file(options.unique_id_file, &unique_id);
  }

  nvshmemx_init_attr_t attr;
  std::memset(&attr, 0, sizeof(attr));
  int ret = nvshmemx_set_attr_uniqueid_args(options.rank, options.nranks, &unique_id, &attr);
  if (ret != 0) {
    std::fprintf(stderr, "[rank %d] nvshmemx_set_attr_uniqueid_args failed ret=%d\n",
                 options.rank, ret);
    return 41;
  }
  ret = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  if (ret != 0) {
    std::fprintf(stderr, "[rank %d] nvshmemx_init_attr failed ret=%d\n",
                 options.rank, ret);
    return 42;
  }

  int pe = nvshmem_my_pe();
  int pes = nvshmem_n_pes();
  if (pe != options.rank || pes != options.nranks) {
    std::fprintf(stderr, "[rank %d] NVSHMEM rank mismatch pe=%d pes=%d\n",
                 options.rank, pe, pes);
    return 43;
  }

  int* sym = static_cast<int*>(nvshmem_malloc(options.bytes));
  if (!sym) {
    std::fprintf(stderr, "[rank %d] nvshmem_malloc(%zu) failed\n",
                 options.rank, options.bytes);
    return 44;
  }
  size_t int_count = options.bytes / sizeof(int);
  std::fprintf(stderr, "[rank %d] nvshmem_malloc ptr=%p bytes=%zu\n",
               options.rank, static_cast<void*>(sym), options.bytes);

  run_nvshmem_device_exchange(options, sym, int_count, 1);

  char path[1024];
  std::snprintf(path, sizeof(path), "%s/rank%d.ready", options.control_dir.c_str(), options.rank);
  write_text_file(path, "READY\n");

  std::snprintf(path, sizeof(path), "%s/prepare", options.control_dir.c_str());
  wait_for_file(path, "prepare", options.rank);
  call_gcr("gcr_checkpoint_prepare", options.rank);
  std::snprintf(path, sizeof(path), "%s/rank%d.prepared", options.control_dir.c_str(), options.rank);
  write_text_file(path, "PREPARED\n");

  std::snprintf(path, sizeof(path), "%s/restore_export", options.control_dir.c_str());
  wait_for_file(path, "restore_export", options.rank);
  call_gcr("gcr_checkpoint_restore_export", options.rank);
  std::snprintf(path, sizeof(path), "%s/rank%d.restore_export_done",
                options.control_dir.c_str(), options.rank);
  write_text_file(path, "RESTORE_EXPORT_DONE\n");

  std::snprintf(path, sizeof(path), "%s/restore_import", options.control_dir.c_str());
  wait_for_file(path, "restore_import", options.rank);
  call_gcr("gcr_checkpoint_restore_import", options.rank);
  std::snprintf(path, sizeof(path), "%s/rank%d.restore_import_done",
                options.control_dir.c_str(), options.rank);
  write_text_file(path, "RESTORE_IMPORT_DONE\n");

  run_nvshmem_device_exchange(options, sym, int_count, 2);

  nvshmem_free(sym);
  nvshmem_finalize();
  CHECK_CUDA(cudaDeviceReset());
  std::fprintf(stderr, "[rank %d] TEST PASSED\n", options.rank);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    std::fprintf(stderr,
                 "usage: %s --rank 0|1 --nranks 2 --bytes N "
                 "--control-dir DIR --unique-id-file PATH\n",
                 argv[0]);
    return 2;
  }
  return run(options);
}
