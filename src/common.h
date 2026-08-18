#ifndef COMMON_H
#define COMMON_H

#include <unistd.h>   // for many thing
#include <stdlib.h>   // for standard library
#include <stdio.h>    // for file dump
#include <time.h>     // for timing
#include <dlfcn.h>    // for loading real shared library
#include <stdint.h>   // for uint64_t defn
#include <stdbool.h>  // for true false
#include <elf.h>      // for ELF Header
#include <sys/wait.h> // for waiting subprocess
#include <sys/stat.h> // for directory
#include <sys/mman.h>  // for mmap
#include <pthread.h>  // for mutex lock
#include <signal.h>   // for signal handling
#include <map>
#include <utility>
#include <atomic>
#include <mutex>

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ROUND_UP_2MB(x) (((x) + (2 * 1024 * 1024 - 1)) & ~(2 * 1024 * 1024 - 1))
// CUDA VMM mapping granularity: every hooked allocation is reserved/mapped in
// units of this size (see ROUND_UP_2MB uses in the cudaMalloc hook).
#define VMM_GRANULE_SIZE (2UL * 1024 * 1024)

// SHM_SIZE: Per-GPU checkpoint buffer on hugepages.
// Each GPU process allocates SHM_SIZE + STAGING_BUF_SIZE*STAGING_BUF_NUM.
// For TP=N, total hugepage needed = N * (SHM_SIZE + 2*staging) + overhead.
//
// KEP-0002: these are runtime values now. The compile-time SHM_SIZE_GB
// (cmake -DSHM_SIZE_GB=40) sets the DEFAULT; GPU_CR_SHM_GB / GPU_CR_SHM_MB
// / GPU_CR_STAGING_MB env vars override it at library load. The macros
// below keep every historical use site source-compatible while reading
// the cached runtime config.
#ifndef SHM_SIZE_GB
#define SHM_SIZE_GB 25
#endif
#define GPU_CR_SHM_DEFAULT ((unsigned long)SHM_SIZE_GB << 30)
#define GPU_CR_STAGING_DEFAULT (1UL << 30)

#include "gpu_cr_config.h"

#define SHM_SIZE (gpu_cr_config().shm_size)

#define MAX_FILE_NUM 4096
#define COPY_THRESHOLD (1UL << 29) // 0.5GB, when to copy from host_buf to shm
#define NUM_COPY_THREADS 4
#define CR_INIT_SIGNAL     SIGRTMAX
#define CR_CKPT_SIGNAL     SIGUSR1
#define CR_RESTORE_SIGNAL  SIGUSR2

// Multi-GPU: IPC teardown/rebuild signals (real-time signals)
// These replace the old NCCL suspend/resume signals.
#define CR_IPC_TEARDOWN_SIGNAL  (SIGRTMAX - 1)
#define CR_IPC_REBUILD_SIGNAL   (SIGRTMAX - 2)
#define CR_IPC_VALIDATE_SIGNAL  (SIGRTMAX - 3)

// Legacy aliases (for backward compatibility during transition)
#define CR_NCCL_SUSPEND_SIGNAL  CR_IPC_TEARDOWN_SIGNAL
#define CR_NCCL_RESUME_SIGNAL   CR_IPC_REBUILD_SIGNAL

// Maximum number of processes in multi-GPU checkpoint
#define MAX_MULTI_GPU_PROCS 32

#define STAGING_BUF_SIZE (gpu_cr_config().staging_size) // default 1GB, env-overridable (KEP-0002)
#define STAGING_BUF_NUM 2

typedef void (*sighandler_t)(int);
typedef sighandler_t (*signal_func_t)(int, sighandler_t);

// Global memory tracking map: ptr -> size
extern std::map<void*, size_t> allocated_memory;

// Global memory type tracking: ptr -> type (0=runtime Malloc, 1=VMM)
extern std::map<void*, int> allocated_memory_type;

extern std::mutex gpu_mem_mutex;

// Helper function declarations
void memcpy_multi(void* dest, void* src, size_t size);

struct shared_mem_file {
    void* ptr;
    uint64_t start_offset;
    uint64_t size;
};

struct shared_mem_fs {
    uint64_t file_num;
    uint64_t current_offset;
    struct shared_mem_file files[MAX_FILE_NUM];
};

#define MAX_SELECTIVE_REGIONS 4096

struct selective_cr_region {
    void* ptr;
    uint64_t size;
};

// GEP-0001: destination-path selective checkpoints.
// The v2 fields are APPENDED so the v1 prefix (num_regions + regions[])
// keeps its exact offsets: a v1 .so never reads past regions[], and the
// zero-initialized control mapping makes proto_version==0 (v1) the default.
#define SELECTIVE_CR_PROTO_V2   2
#define SELECTIVE_CR_MAX_PATH 256

struct selective_cr_request {
    uint32_t num_regions;
    struct selective_cr_region regions[MAX_SELECTIVE_REGIONS];
    /* --- v2 extension (GEP-0001) --- */
    uint32_t proto_version;                    /* 0 = v1, 2 = v2 */
    char     dest_path[SELECTIVE_CR_MAX_PATH]; /* empty = per-PID buffer */
};

// Capability bits published by the .so in signal_controls.capability at
// init_CR and re-asserted at every FINISH (consume-once zeroing of the
// request extension deliberately excludes this word).
#define CR_CAP_DEST_PATH (1u << 0)

// Trailing commit marker for destination-file dumps: written at
// fs->current_offset only after the last extent has landed, so a torn
// dump is detectable. Restores from a destination file refuse dumps
// whose marker is absent or stale.
#define DUMP_COMMIT_MAGIC 0x31524347u /* "GCR1" */

struct dump_commit {
    uint64_t magic;
    uint64_t generation;
};

struct signal_controls {
    uint32_t signal;
    struct selective_cr_request selective_req;
    /* --- v2 extension (GEP-0001): appended, invisible to v1 readers --- */
    uint32_t capability; /* CR_CAP_* bits, persistent across ops */
    uint32_t proto_ack;  /* proto level the .so served the last op at */
    int32_t  op_status;  /* 0 = OK, else positive errno-style code */
};

#endif