#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <assert.h>
#include <chrono>
#include <limits.h>
#include <libgen.h>
#include <string.h>
#include <string>

#ifdef __HIP_PLATFORM_AMD__
// AMD platform 
#else
#include <cuda.h>
#endif

#include "common.h"
#include "comm/comm.h"

std::string get_cuda_checkpoint_path() {
    char exe_path[1024];
    ssize_t count = readlink("/proc/self/exe", exe_path, 1024);
    if (count == -1) {
        perror("readlink");
        return "cuda-checkpoint";
    }
    exe_path[count] = '\0';

    char* dir = dirname(exe_path);
    std::string full_path = std::string(dir) + "/../cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint";

    if (access(full_path.c_str(), X_OK) != 0) {
        fprintf(stderr, "WARNING: helper binary not found at: %s\n", full_path.c_str());
        return "cuda-checkpoint"; 
    }

    return full_path;
}


static bool parse_selective_regions(const char* spec, selective_cr_request* req) {
    req->num_regions = 0;
    char* buf = strdup(spec);
    char* token = strtok(buf, ",");
    while (token) {
        if (req->num_regions >= MAX_SELECTIVE_REGIONS) {
            fprintf(stderr, "Error: too many selective regions (max %d)\n", MAX_SELECTIVE_REGIONS);
            free(buf);
            return false;
        }
        char* colon = strchr(token, ':');
        if (!colon) {
            fprintf(stderr, "Error: invalid region format '%s' (expected ptr:size)\n", token);
            free(buf);
            return false;
        }
        *colon = '\0';
        void* ptr = (void*)strtoull(token, nullptr, 0);
        uint64_t size = strtoull(colon + 1, nullptr, 0);
        if (size == 0) {
            fprintf(stderr, "Error: region size is 0 for ptr %s\n", token);
            free(buf);
            return false;
        }
        req->regions[req->num_regions].ptr = ptr;
        req->regions[req->num_regions].size = size;
        req->num_regions++;
        token = strtok(nullptr, ",");
    }
    free(buf);
    return req->num_regions > 0;
}


int main(int argc, char* argv[]) {
    int opt;
    int init = 0;
    int ckpt = 0;
    int restore = 0;
    int dump = 0;
    int pid = 0;
    int criu_pid = 0;
    int buffer_only = 0;
    const char* selective_spec = nullptr;
    while ((opt = getopt(argc, argv, "icrdbp:m:s:")) != -1) {
        switch (opt) {
            case 'i':
                init = 1;
                break;
            case 'c':
                ckpt = 1;
                break;
            case 'r':
                restore = 1;
                break;
            case 'p':
                pid = atoi(optarg);
                break;
            case 'm':
                criu_pid = atoi(optarg);
                break;
            case 'b':
                buffer_only = 1;
                break;
            case 's':
                selective_spec = optarg;
                break;
            default:
                fprintf(stderr, "Usage: %s [-i|-c|-r] -p pid [-m criu_pid] [-b] [-s ptr:size,...]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    if(!ckpt && !restore && !dump && !init) {
        fprintf(stderr, "Either -i, -c, or -r must be specified\n");
        exit(EXIT_FAILURE);
    }
    if(ckpt + restore + dump + init > 1) {
        fprintf(stderr, "Only one of -i, -c, or -r  can be specified\n");
        exit(EXIT_FAILURE);
    }
    assert(pid != 0);
    if (criu_pid == 0) criu_pid = pid;

    ShareMemComm *comm = new ShareMemComm(pid);
    comm->setup();

    int ret;

    if(init) {
        comm->send_msg(INIT_MSG);
        kill(pid, CR_INIT_SIGNAL);
        while(!comm->is_finished()) {
            usleep(1000);
        }
    }  else if(ckpt && selective_spec) {
        selective_cr_request req;
        if (!parse_selective_regions(selective_spec, &req)) {
            fprintf(stderr, "Error: failed to parse selective regions\n");
            exit(EXIT_FAILURE);
        }
        printf("Selective checkpoint: %u regions\n", req.num_regions);
        for (uint32_t i = 0; i < req.num_regions; i++) {
            printf("  region %u: ptr=%p size=%lu\n", i, req.regions[i].ptr, req.regions[i].size);
        }
        comm->control->selective_req = req;
        comm->send_msg(SELECTIVE_CKPT_MSG);
        kill(pid, CR_CKPT_SIGNAL);
        printf("Selective dump signal sent.\n");
        while(!comm->is_finished()) {
            usleep(1000);
        }
        printf("Selective checkpointing done\n");
    } else if(restore && selective_spec) {
        selective_cr_request req;
        if (!parse_selective_regions(selective_spec, &req)) {
            fprintf(stderr, "Error: failed to parse selective regions\n");
            exit(EXIT_FAILURE);
        }
        printf("Selective restore: %u regions\n", req.num_regions);
        comm->control->selective_req = req;
        comm->send_msg(SELECTIVE_RESTORE_MSG);
        kill(pid, CR_RESTORE_SIGNAL);
        printf("Selective restore signal sent.\n");
        while(!comm->is_finished()) {
            usleep(1000);
        }
        printf("Selective restore done\n");
    } else if(ckpt) {
        comm->send_msg(CKPT_MSG);
        kill(pid, CR_CKPT_SIGNAL);
        printf("Dump signal sent.\n");
        while(!comm->is_finished()) {
            usleep(1000);
        }
        printf("Dumping done.\n");
#ifdef __HIP_PLATFORM_AMD__
        // For AMD: call CRIU to dump the process
        const char* ckpt_dir = getenv("AMD_CKPT_DIR");
        if (!ckpt_dir) {
            fprintf(stderr, "ERROR: AMD_CKPT_DIR environment variable not set!\n");
            fprintf(stderr, "Please set: export AMD_CKPT_DIR=/path/to/checkpoint/dir\n");
            exit(EXIT_FAILURE);
        }
        
        printf("AMD: Calling CRIU to checkpoint process %d\n", criu_pid);
        printf("Checkpoint directory: %s\n", ckpt_dir);
        
        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
            "sudo env LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu "
            "criu dump --link-remap --tcp-established -t %d -D %s -j -v4 -o %s/dump.log --ghost-limit 50M --ext-unix-sk -L /usr/local/lib/criu",
            criu_pid, ckpt_dir, ckpt_dir);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only)
            ret = system(cmd);
        if (ret < 0) {
            perror("system()");
            exit(EXIT_FAILURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("CRIU checkpoint time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
#else
        std::string bin_path = get_cuda_checkpoint_path();
        std::string cmd = bin_path + " --toggle --pid " + std::to_string(pid);
        // std::string cmd = "cuda-checkpoint --toggle --pid " + std::to_string(pid);
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only)
            ret = system(cmd.c_str());
        if (ret < 0) {
            perror("system()");
            exit(EXIT_FAILURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("cuda-checkpoint checkpoint time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
#endif
        printf("Checkpointing done\n");
    } else if(restore) {
#ifdef __HIP_PLATFORM_AMD__
        const char* ckpt_dir = getenv("AMD_CKPT_DIR");
        if (!ckpt_dir) {
            fprintf(stderr, "ERROR: AMD_CKPT_DIR environment variable not set!\n");
            fprintf(stderr, "Please set: export AMD_CKPT_DIR=/path/to/checkpoint/dir\n");
            exit(EXIT_FAILURE);
        }
        
        printf("AMD: Calling CRIU to restore process\n");
        printf("Restore directory: %s\n", ckpt_dir);
        
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "sudo env  LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu "
            "criu restore --tcp-established -D %s -j -v4 -o %s/restore.log -L /usr/local/lib/criu --pidfile %s/restored.pid --ghost-limit 50M --ext-unix-sk --restore-detached ",
            ckpt_dir, ckpt_dir, ckpt_dir);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only)
            ret = system(cmd);
        if (ret < 0) {
            perror("system()");
            exit(EXIT_FAILURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("CRIU restore time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
        printf("Calling GPU-CR\n");

        comm->send_msg(RESTORE_MSG);
        kill(pid, CR_RESTORE_SIGNAL);
        
        while(!comm->is_finished()) {
            usleep(1000);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("GPUos restore time: %.3f s\n", std::chrono::duration<double>(t2 - t1).count());

        printf("Process internal restoration finished.\n");
        printf("Restoring done\n");
#else
        comm->send_msg(RESTORE_MSG);
        kill(pid, CR_RESTORE_SIGNAL);
        while(!comm->is_finished()) {
            usleep(1000);
        }
        std::string bin_path = get_cuda_checkpoint_path();
        std::string cmd = bin_path + " --toggle --pid " + std::to_string(pid);
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only)
            ret = system(cmd.c_str());
        if (ret < 0) {
            perror("system()");
            exit(EXIT_FAILURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("cuda-checkpoint restore time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
        printf("Restoring done\n");
#endif
    }
    return 0;
}
