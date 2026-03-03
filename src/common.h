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

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ROUND_UP_2MB(x) (((x) + (2 * 1024 * 1024 - 1)) & ~(2 * 1024 * 1024 - 1))
#define SHM_SIZE (50UL << 30) // 40GB
#define MAX_FILE_NUM 4096
#define COPY_THRESHOLD (1UL << 29) // 0.5GB, when to copy from host_buf to shm
#define NUM_COPY_THREADS 4
#define CR_INIT_SIGNAL     SIGRTMAX
#define CR_CKPT_SIGNAL     SIGUSR1
#define CR_RESTORE_SIGNAL  SIGUSR2
#define STAGING_BUF_SIZE (1UL << 30) // 1GB staging buffer
#define STAGING_BUF_NUM 2

typedef void (*sighandler_t)(int);
typedef sighandler_t (*signal_func_t)(int, sighandler_t);

// Global memory tracking map: ptr -> size
extern std::map<void*, size_t> allocated_memory;

// Global memory type tracking: ptr -> type (0=cudaMalloc, 1=VMM)
extern std::map<void*, int> allocated_memory_type;

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

struct signal_controls {
    uint32_t signal;
};

#endif