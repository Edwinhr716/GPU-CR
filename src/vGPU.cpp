#include <atomic>
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"
#include "comm/comm.h"
#include "backend/backend.h"
#include "GPUs/GPU.h"

std::mutex fs_mutex;
Comm *comm;
Backend *backend;
GPU *gpu;

void* staging_buf[STAGING_BUF_NUM];

bool CR_initialized = false;

// Helper function: multi-threaded memcpy
void memcpy_multi(void* dest, void* src, size_t size) {
    std::vector<std::thread> threads;
    size_t chunk_size = (size + NUM_COPY_THREADS - 1) / NUM_COPY_THREADS;
    for (int i = 0; i < NUM_COPY_THREADS; i++) {
        size_t offset = i * chunk_size;
        if (offset >= size) break;
        size_t this_chunk_size = std::min(chunk_size, size - offset);
        threads.emplace_back([=]() {
            memcpy((char*)dest + offset, (char*)src + offset, this_chunk_size);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}


double ckpt() {
    fprintf(stderr, "[vGPU-CKPT] ckpt() entered, PID=%d\n", getpid());
    fflush(stderr);
    
    double tot_size = 0;
    
    auto time_start = std::chrono::high_resolution_clock::now();
    long sync_time = 0, cpu_copy_time = 0, release_time = 0;

    void* tmp_buf = backend->get_tmp_buf();
    fprintf(stderr, "[vGPU-CKPT] tmp_buf=%p\n", tmp_buf);
    fflush(stderr);
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
    int current_buf = 0;
    size_t buf_offset = 0;
    size_t des_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));
    
    fs_mutex.lock();
    
    fs->file_num = 0;
    fs->current_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));

    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\\n");
        fs_mutex.unlock();
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\\n");
        fs_mutex.unlock();
        exit(-1);
    }
    gpu->recordEvent(event, stream);
    
    fprintf(stderr, "[vGPU-CKPT] ckpt %ld ptrs\n", allocated_memory.size());
    fflush(stderr);

    int ptr_idx = 0;
    for (const auto& entry : allocated_memory) {
        fprintf(stderr, "[vGPU-CKPT] Processing ptr #%d: %p\n", ++ptr_idx, entry.first);
        fflush(stderr);
        void* d = entry.first;
        uint64_t size = ROUND_UP_2MB(entry.second);
        tot_size += size;

        // Record file info
        fs->files[fs->file_num].ptr = d;
        fs->files[fs->file_num].start_offset = fs->current_offset;
        fs->files[fs->file_num].size = size;
        fs->current_offset += size;
        if (fs->current_offset > SHM_SIZE) {
            fprintf(stderr, "[vGPU-CKPT] Error: Not enough space in shared memory\n");
            fs_mutex.unlock();
            exit(-1);
        }
        fs->file_num++;
        if (fs->file_num >= MAX_FILE_NUM) {
            fprintf(stderr, "[vGPU-CKPT] Error: Too many files in shared memory fs\n");
            fs_mutex.unlock();
            exit(-1);
        }
        
        // Copy data from GPU to staging buffer
        while(size > 0) {
            size_t cur_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            void* start_addr = (char*)staging_buf[current_buf & 1] + buf_offset;
            
            if (gpu->memcpyAsync(start_addr, d, cur_size, GPUMemcpyKind::DeviceToHost, stream) != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\\n");
                fs_mutex.unlock();
                exit(-1);
            }
            
            buf_offset += cur_size;
            d = (char*)d + cur_size;
            size -= cur_size;
            if(buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                if(current_buf > 0) {
                    auto t3 = std::chrono::high_resolution_clock::now();
                    gpu->synchronizeEvent(event);
                    auto t4 = std::chrono::high_resolution_clock::now();
                    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
                    
                    auto t5 = std::chrono::high_resolution_clock::now();
                    memcpy_multi((char*)fs + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
                    auto t6 = std::chrono::high_resolution_clock::now();
                    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
                    
                    des_offset += STAGING_BUF_SIZE;
                }
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }
    if(current_buf > 0) {
        auto t3 = std::chrono::high_resolution_clock::now();
        gpu->synchronizeEvent(event);
        auto t4 = std::chrono::high_resolution_clock::now();
        sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        
        auto t5 = std::chrono::high_resolution_clock::now();
        memcpy_multi((char*)fs + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
        auto t6 = std::chrono::high_resolution_clock::now();
        cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        
        des_offset += STAGING_BUF_SIZE;
    }
    
    auto t7 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto t8 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
    
    auto t9 = std::chrono::high_resolution_clock::now();
    memcpy_multi((char*)fs + des_offset, staging_buf[current_buf & 1], buf_offset);
    auto t10 = std::chrono::high_resolution_clock::now();
    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();
    
    assert(des_offset + buf_offset == fs->current_offset);
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);

    // Release physical GPU memory after checkpoint (but keep virtual addresses)
    fprintf(stderr, "Releasing physical GPU memory for %ld pointers...\n", allocated_memory.size());
    auto t11 = std::chrono::high_resolution_clock::now();
    for (const auto& entry : allocated_memory) {
        void* ptr = entry.first;
        if (gpu->releasePhysicalMemory(ptr) != 0) {
            fprintf(stderr, "Error: Failed to release physical memory for ptr %p\n", ptr);
            fs_mutex.unlock();
            exit(-1);
        }
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    release_time = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();
    fprintf(stderr, "Physical GPU memory released, virtual addresses preserved\n");
    

    fprintf(stderr, "=== Checkpoint Timing Breakdown ===\n");
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n", 
            cpu_copy_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0));
    fprintf(stderr, "  Release memory:   %6ld ms\n", release_time / 1000);
    long data_transfer_time = sync_time + cpu_copy_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0));
    fprintf(stderr, "===================================\n");
    
    fs_mutex.unlock();
    return tot_size;
}

double restore_ptr_and_content() {
    double tot_size = 0;
    
    long remap_time = 0, cpu_copy_time = 0, sync_time = 0;
    
    void* tmp_buf = backend->get_tmp_buf();
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;

    uint64_t file_num = fs->file_num;
    fprintf(stderr, "[vGPU-restore] restore %lu ptrs\n", file_num);
    
    // Remap physical memory for all pointers before copying data
    fprintf(stderr, "[vGPU-restore] Remapping physical GPU memory for %lu pointers...\n", file_num);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < file_num; i++) {
        void* ptr = fs->files[i].ptr;
        uint64_t size = fs->files[i].size;
        if (gpu->remapPhysicalMemory(ptr, size) != 0) {
            fprintf(stderr, "Error: Failed to remap physical memory for ptr %p\n", ptr);
            exit(-1);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    remap_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    fprintf(stderr, "[vGPU-restore] Physical GPU memory remapped\n");
    
    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\\n");
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\\n");
        exit(-1);
    }
    gpu->recordEvent(event, stream);

    int current_buf = 0;
    size_t buf_offset = 0;
    size_t src_offset = 0;
    
    for (uint64_t i = 0; i < file_num; i++) {
        void* requestedAddr = fs->files[i].ptr;
        uint64_t offset = fs->files[i].start_offset;
        uint64_t size = fs->files[i].size;
        tot_size += size;
        
        if(i == 0) {
            src_offset = fs->files[i].start_offset;
            size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
            auto tc1 = std::chrono::high_resolution_clock::now();
            memcpy_multi(staging_buf[current_buf & 1], (char*)fs + src_offset, cpu_copy_size);
            auto tc2 = std::chrono::high_resolution_clock::now();
            cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1).count();
            buf_offset = 0;
        }
        
        while(size > 0) {
            size_t this_copy_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            assert(buf_offset == offset - src_offset);
            
            if (gpu->memcpyAsync(requestedAddr, (char*)staging_buf[current_buf & 1] + (offset - src_offset),
                               this_copy_size, GPUMemcpyKind::HostToDevice, stream) != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\\n");
                exit(-1);
            }
            
            buf_offset += this_copy_size;
            offset += this_copy_size;
            requestedAddr = (char*)requestedAddr + this_copy_size;
            size -= this_copy_size;
            
            if(buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                src_offset += STAGING_BUF_SIZE;
                size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
                
                auto ts1 = std::chrono::high_resolution_clock::now();
                gpu->synchronizeEvent(event);
                auto ts2 = std::chrono::high_resolution_clock::now();
                sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts2 - ts1).count();
                
                auto tc3 = std::chrono::high_resolution_clock::now();
                memcpy_multi(staging_buf[(current_buf + 1) & 1], (char*)fs + src_offset, cpu_copy_size);
                auto tc4 = std::chrono::high_resolution_clock::now();
                cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc4 - tc3).count();
                
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }
    
    auto ts3 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto ts4 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts4 - ts3).count();
    
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);
    
    fprintf(stderr, "=== Restore Timing Breakdown ===\n");
    fprintf(stderr, "  Remap memory:     %6ld ms\n", remap_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n",
            cpu_copy_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0));
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    long data_transfer_time = cpu_copy_time + sync_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0));
    fprintf(stderr, "================================\n");
    
    return tot_size;
}

int get_id() {
    char id_name[40];
    int fd_id = open("/mnt/huge-ckpt/control", O_CREAT | O_RDWR, 0755);
    if (fd_id < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    // Set file size before mmap to avoid Bus error
    if (ftruncate(fd_id, HUGE_PAGE_SIZE) < 0) {
        perror("ftruncate()");
        exit(EXIT_FAILURE);
    }
    std::atomic<int>* id_ptr = (std::atomic<int>*)mmap(NULL, HUGE_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_id, 0);
    if (id_ptr == MAP_FAILED) {
        perror("mmap()");
        exit(EXIT_FAILURE);
    }
    int id = id_ptr->fetch_add(1);
    fprintf(stderr, "Process ID: %d, assigned CR ID: %d\n", getpid(), id);
    return id;
}


void init_CR() {
    if (CR_initialized) {
        fprintf(stderr, "[init_CR] CR already initialized\n");
        return;
    }

    fprintf(stderr, "[init_CR] Starting CR initialization...\n");
    int id = get_id();
    comm = new ShareMemComm(getpid());
    comm->setup();
    backend = new ShareMem(id);
    backend->setup();
    gpu = createGPU();  // createGPU() will detect the GPU vendor and return the appropriate GPU object
    fprintf(stderr, "[init_CR] GPU vendor: %s\n", gpu->getVendorName().c_str());
    fprintf(stderr, "[init_CR] Allocating staging buffer (%zu MB)...\n", 
            (STAGING_BUF_SIZE * STAGING_BUF_NUM) / (1024 * 1024));
    
    void* tmp_buf_host = backend->get_host_buffer();
    if (!tmp_buf_host) {
        fprintf(stderr, "[init_CR] Error: Backend host buffer is null\n");
        exit(EXIT_FAILURE);
    }
    
    // Try to register as pinned memory
    size_t total_size = STAGING_BUF_SIZE * STAGING_BUF_NUM;
    if (gpu->registerHostMemory(tmp_buf_host, total_size) == 0) {
        fprintf(stderr, "[init_CR] Successfully registered as pinned memory\n");
    } else {
        fprintf(stderr, "[init_CR] Note: Could not register as pinned (will use regular memory)\n");
    }
    
    for (int i = 0; i < STAGING_BUF_NUM; i++) {
        staging_buf[i] = (char*)tmp_buf_host + i * STAGING_BUF_SIZE;
    }

    CR_initialized = true;
    fprintf(stderr, "[init_CR] Initialization complete, setting CR_initialized = true\n");
}

void cr_signal_handler(int signum) {
    fprintf(stderr, "[vGPU] Received signal %d from process %d\n", signum, getpid());
    fflush(stderr);
    
    // Only handle our specific signals
    if (signum != CR_INIT_SIGNAL && signum != CR_CKPT_SIGNAL && signum != CR_RESTORE_SIGNAL) {
        fprintf(stderr, "[vGPU] Ignoring unknown signal %d (not a CR signal)\n", signum);
        return;
    }
    
    if(signum == CR_INIT_SIGNAL) {
        if (!CR_initialized) {
            fprintf(stderr, "[vGPU] Starting init_CR()...\n");
            init_CR();
            fprintf(stderr, "[vGPU] CR initialization complete\n");
        } else {
            fprintf(stderr, "[vGPU] CR already initialized, skipping\n");
        }
        comm->send_msg(FINISH_MSG);
        fprintf(stderr, "[vGPU] FINISH_MSG sent, returning from signal handler\n");
        fflush(stderr);
        return;
    }

    if(!CR_initialized) {
        fprintf(stderr, "CR not initialized, initializing now...\n");
        init_CR();
    }

    uint32_t msg = comm->recv_msg();
    if(msg == CKPT_MSG) {
        fprintf(stderr, "waiting for kernels to finish...\n");
        gpu->syncAllKernels();
        fprintf(stderr, "start ckpt...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = ckpt();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        fprintf(stderr, "ckpt size: %f GB, time: %ld ms, bw: %f GB/s\n", 
               tot_size / 1024 / 1024 / 1024, duration.count(), 
               tot_size / duration.count() * 1000 / 1024 / 1024 / 1024);
        
        // Note: External checkpoint (cuda-checkpoint for NVIDIA, CRIU for AMD) 
        // is called from cr_client, not here
    } else if (msg == RESTORE_MSG) {
        fprintf(stderr, "start restore...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = restore_ptr_and_content();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        fprintf(stderr, "restore size: %f GB, time: %ld ms, bw: %f GB/s\n", 
               tot_size / 1024 / 1024 / 1024, duration.count(), 
               tot_size / duration.count() * 1000 / 1024 / 1024 / 1024);
        fprintf(stderr, "finish restore\n");
    }
    comm->send_msg(FINISH_MSG);
}

__attribute__((constructor)) void init() {
    fprintf(stderr, "[vGPU] Library loaded! Registering signal handlers...\n");
    fflush(stderr);
    
    signal(CR_INIT_SIGNAL, cr_signal_handler);
    signal(CR_CKPT_SIGNAL, cr_signal_handler);
    signal(CR_RESTORE_SIGNAL, cr_signal_handler);
}