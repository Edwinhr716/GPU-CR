#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../common.h"
#include "backend.h"

Backend::Backend(int id) {
}

Backend::~Backend() {
}

void Backend::setup() {
}

ShareMem::ShareMem(int id) : Backend(id), id(id) {
}

ShareMem::~ShareMem() {
}

void ShareMem::setup() {
    // tmp_buf
    char shm_name[512];
    const char* export_file_path = std::getenv("EXPORT_FILE_PATH");
    bool use_file_backend = (export_file_path != nullptr);

    if(use_file_backend){
        sprintf(shm_name, "%s/ckpt-%d.data", export_file_path, id);
        fprintf(stderr, "[ShareMem] Using File Backend: %s\n", shm_name);
        
        int fd = open(shm_name, O_CREAT | O_RDWR, 0644);
        if (fd < 0) {
            perror("open() file backend");
            exit(EXIT_FAILURE);
        }

        if (ftruncate(fd, SHM_SIZE) < 0) {
            perror("ftruncate() file backend");
            exit(EXIT_FAILURE);
        }

        tmp_buf = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (tmp_buf == MAP_FAILED) {
            perror("mmap file backend failed");
            exit(EXIT_FAILURE);
        }
    }
    else{
        sprintf(shm_name, "/mnt/huge-ckpt/%d", id);
        int fd = open(shm_name, O_CREAT | O_RDWR, 0755);
        if (fd < 0) {
            perror("open()");
            exit(EXIT_FAILURE);
        }

        int ret = ftruncate(fd, SHM_SIZE);
        if (ret < 0) {
            perror("ftruncate()");
            exit(EXIT_FAILURE);
        }

        tmp_buf = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (tmp_buf == MAP_FAILED) {
            perror("mmap with hugepages failed");
            fprintf(stderr, "Check if hugepages are configured properly\n");
            exit(EXIT_FAILURE);
        }

        fprintf(stderr, "Hugepage shared memory mapped at %p\n", tmp_buf);
    }

    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
    fs_mutex.lock();
    fs->file_num = 0;
    fs->current_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));
    fs_mutex.unlock();


    // host_buf
    sprintf(shm_name, "/mnt/huge-ckpt/%d-host", id);
    fd_host = open(shm_name, O_CREAT | O_RDWR, 0755);
    if (fd_host < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    
    size_t host_buf_total_size = STAGING_BUF_SIZE * STAGING_BUF_NUM;
    if (ftruncate(fd_host, host_buf_total_size) < 0) {
        perror("ftruncate() host mem");
        exit(EXIT_FAILURE);
    }

    host_buf_ptr = mmap(NULL, host_buf_total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_host, 0);
    if (host_buf_ptr == MAP_FAILED) {
        perror("mmap host mem failed");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "[ShareMem] Host memory mapped at %p (size: %zu)\n", host_buf_ptr, host_buf_total_size);

    
}

void* ShareMem::get_tmp_buf() {
    return tmp_buf;
}

void* ShareMem::get_host_buffer() {
        return host_buf_ptr;
    }