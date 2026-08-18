#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <assert.h>
#include <chrono>
#include <limits.h>
#include <libgen.h>
#include <string.h>
#include <string>

// openat2 hardening for the pre-create (GEP-0001 F8): cr_client runs as
// root in the agent pod against a workload-writable store, so symlink
// swaps must not let it create/truncate arbitrary host files. Definitions
// are local so old kernel headers still build; ENOSYS falls back to
// openat(O_NOFOLLOW).
#ifndef SYS_openat2
#define SYS_openat2 437
#endif
#ifndef RESOLVE_NO_SYMLINKS
#define RESOLVE_NO_SYMLINKS 0x04ULL
#endif
#ifndef RESOLVE_BENEATH
#define RESOLVE_BENEATH 0x08ULL
#endif
struct gpu_cr_open_how {
    uint64_t flags;
    uint64_t mode;
    uint64_t resolve;
};

// Exit codes (consumed by the snapshot agent):
//   0 OK, 1 usage/parse, 2 op failed (op_status or invalid dump),
//   3 gate refused (capability/advertisement), 4 timeout.
#define EXIT_OP_FAILED 2
#define EXIT_REFUSED   3
#define EXIT_TIMEOUT   4

#ifdef __HIP_PLATFORM_AMD__
// AMD platform 
#else
#include <cuda.h>
#endif

#include "common.h"
#include "ctl_path.h"
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


// Pre-create the destination file: existence only (O_CREAT), never sizing —
// the .so alone can compute the dump total (GEP-0001 preloader-authoritative
// sizing), and an inode costs no hugepages/blocks in this cgroup.
static bool secure_precreate(const char* path) {
    if (path[0] != '/') {
        fprintf(stderr, "Error: -o path must be absolute: %s\n", path);
        return false;
    }
    char dir_buf[SELECTIVE_CR_MAX_PATH];
    strncpy(dir_buf, path, sizeof(dir_buf) - 1);
    dir_buf[sizeof(dir_buf) - 1] = '\0';
    char* slash = strrchr(dir_buf, '/');
    const char* base = slash + 1;
    if (*base == '\0') {
        fprintf(stderr, "Error: -o path is a directory: %s\n", path);
        return false;
    }
    *slash = '\0';
    const char* dir = dir_buf[0] ? dir_buf : "/";

    int dfd = open(dir, O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        fprintf(stderr, "Error: cannot open destination dir %s: %s\n", dir, strerror(errno));
        return false;
    }
    struct gpu_cr_open_how how;
    memset(&how, 0, sizeof(how));
    how.flags = O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW;
    how.mode = 0644;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS;
    long fd = syscall(SYS_openat2, dfd, base, &how, sizeof(how));
    if (fd < 0 && errno == ENOSYS)
        fd = openat(dfd, base, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0644);
    close(dfd);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot create destination %s: %s\n", path, strerror(errno));
        return false;
    }
    close((int)fd);
    return true;
}

// Post-checkpoint validation of a destination dump (GEP-0001 F1): header
// plausibility + trailing commit marker, via read(2) — no mmap, so no
// hugetlb reservation or fault lands in this process's cgroup.
static bool validate_dest_dump(const char* path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot reopen destination %s: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    uint64_t hdr[2]; // file_num, current_offset
    dump_commit dc;
    bool ok = fstat(fd, &st) == 0 &&
              pread(fd, hdr, sizeof(hdr), 0) == (ssize_t)sizeof(hdr) &&
              hdr[0] > 0 && hdr[0] < MAX_FILE_NUM &&
              hdr[1] >= ROUND_UP_2MB(sizeof(shared_mem_fs)) &&
              hdr[1] + sizeof(dump_commit) <= (uint64_t)st.st_size &&
              pread(fd, &dc, sizeof(dc), (off_t)hdr[1]) == (ssize_t)sizeof(dc) &&
              dc.magic == DUMP_COMMIT_MAGIC;
    close(fd);
    if (!ok)
        fprintf(stderr, "Error: destination %s failed dump validation (torn or unserved dump)\n", path);
    return ok;
}

// GEP-0006 pre-signal gate: refuse to kill() unless the target's .so wrote
// a readiness advertisement into OUR ctl dir. Presence in this dir proves
// shared backing (the only way the file got here); proto proves the
// library level; starttime kills the PID-reuse case, where the signal
// would terminate an innocent recycled PID. Path equality is a warning
// only (same tmpfs may be mounted at different container paths).
static bool check_advertisement(const char* ctl_dir, int pid) {
    char path[512];
    snprintf(path, sizeof(path), "%s/ctl-ready-%d", ctl_dir, pid);
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no readiness advertisement at %s — vGPU.so not loaded in PID %d, "
                        "GPU_CR_CTL_PATH mismatch, or pre-GEP-0006 library\n", path, pid);
        return false;
    }
    char buf[600] = "";
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        fprintf(stderr, "Error: empty advertisement %s\n", path);
        return false;
    }
    fclose(f);
    int proto = 0;
    long long adv_start = -1;
    char adv_ctl[512] = "";
    sscanf(buf, "proto=%d starttime=%lld ctl=%511s", &proto, &adv_start, adv_ctl);
    if (proto < GPU_CR_CTL_PROTO) {
        fprintf(stderr, "Error: advertisement proto=%d too low (need >=%d)\n", proto, GPU_CR_CTL_PROTO);
        return false;
    }
    long long cur_start = gpu_cr_starttime(pid);
    if (cur_start < 0) {
        fprintf(stderr, "Error: PID %d no longer exists\n", pid);
        return false;
    }
    if (adv_start != cur_start) {
        fprintf(stderr, "Error: starttime mismatch for PID %d (advertised %lld, current %lld) — "
                        "PID reuse, refusing to signal\n", pid, adv_start, cur_start);
        return false;
    }
    if (adv_ctl[0] && strcmp(adv_ctl, ctl_dir) != 0)
        fprintf(stderr, "Warning: advertisement names ctl=%s, ours is %s (same backing store, "
                        "different mount points?)\n", adv_ctl, ctl_dir);
    return true;
}

// Poll for FINISH with a deadline: a dead or wedged workload must fail this
// invocation, not hang it (and the agent behind it) forever.
static bool wait_finished(ShareMemComm* comm) {
    long timeout_sec = 120;
    const char* t = getenv("GPU_CR_OP_TIMEOUT_SEC");
    if (t && atol(t) > 0) timeout_sec = atol(t);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (!comm->is_finished()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fprintf(stderr, "Error: timed out after %lds waiting for FINISH\n", timeout_sec);
            return false;
        }
        usleep(1000);
    }
    return true;
}

// After a FULL ckpt/restore: a v2 .so reports op_status (KEP-0002 clean
// failures: oversized checkpoint, deferred-buffer ENOMEM). On checkpoint
// failure we must NOT proceed to cuda-checkpoint --toggle — freezing a
// process whose state was never saved is unrecoverable; on restore
// failure we must not unfreeze onto stale weights. A v1 .so leaves
// proto_ack at 0 and keeps historical behavior.
static void check_full_result(ShareMemComm* comm, const char* what) {
    if (comm->control->proto_ack >= SELECTIVE_CR_PROTO_V2 && comm->control->op_status != 0) {
        int32_t st = comm->control->op_status;
        fprintf(stderr, "Error: vGPU.so reported %s failure: %s (status=%d); NOT toggling cuda-checkpoint\n",
                what, strerror(st), st);
        exit(EXIT_OP_FAILED);
    }
}

// After a selective op: surface the .so-reported status. A v2 ack with a
// nonzero status is a clean op failure; a missing ack on a dest-path op
// means the .so never saw the destination (v1 skew) — refuse loudly.
static void check_selective_result(ShareMemComm* comm, const char* dest_path) {
    if (comm->control->proto_ack >= SELECTIVE_CR_PROTO_V2) {
        int32_t st = comm->control->op_status;
        if (st != 0) {
            fprintf(stderr, "Error: vGPU.so reported op failure: %s (status=%d)\n", strerror(st), st);
            exit(EXIT_OP_FAILED);
        }
    } else if (dest_path) {
        fprintf(stderr, "Error: no v2 ack from vGPU.so — destination path was ignored (version skew)\n");
        exit(EXIT_REFUSED);
    }
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
    const char* dest_path = nullptr;
    while ((opt = getopt(argc, argv, "icrdbp:m:s:o:")) != -1) {
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
            case 'o':
                dest_path = optarg;
                break;
            default:
                fprintf(stderr, "Usage: %s [-i|-c|-r] -p pid [-m criu_pid] [-b] [-s ptr:size,...] [-o /path/to/dump]\n", argv[0]);
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
    if (dest_path) {
        if (!selective_spec || (!ckpt && !restore)) {
            fprintf(stderr, "Error: -o requires -c or -r together with -s\n");
            exit(EXIT_FAILURE);
        }
        if (strlen(dest_path) >= SELECTIVE_CR_MAX_PATH) {
            fprintf(stderr, "Error: -o path exceeds %d bytes\n", SELECTIVE_CR_MAX_PATH - 1);
            exit(EXIT_FAILURE);
        }
    }
    assert(pid != 0);
    if (criu_pid == 0) criu_pid = pid;

    // GEP-0006: a configured-but-broken ctl path is refused HERE, loudly.
    // (The .so falls back to legacy instead — it must not die — so the
    // coordinator is where the misconfiguration surfaces.)
    int ctl_mode = 0;
    const char* ctl_dir = gpu_cr_ctl_dir(&ctl_mode);
    const char* ctl_env = getenv("GPU_CR_CTL_PATH");
    if (ctl_env && ctl_env[0] && !ctl_mode) {
        fprintf(stderr, "Error: GPU_CR_CTL_PATH=%s is missing or not tmpfs-backed "
                        "(mount-order shadowing?) — refusing to operate\n", ctl_env);
        exit(EXIT_REFUSED);
    }

    // Pre-signal gate (GEP-0006): never kill() a PID whose .so hasn't
    // advertised readiness in our ctl dir — covers not-loaded, env
    // mismatch, pre-GEP libraries and PID reuse in one check.
    if (ctl_mode && !check_advertisement(ctl_dir, pid))
        exit(EXIT_REFUSED);

    ShareMemComm *comm = new ShareMemComm(pid);
    comm->setup();

    // Serialize concurrent cr_clients against the same PID for the whole op
    // (GEP-0001 F2): nothing else orders two writers of selective_req.
    // Fixed order, legacy lock first (GEP-0006 F4): the legacy file is
    // taken with open(O_CREAT)+flock only — an inode lock faults no
    // hugetlb pages, so it stays safe with a zero hugepages request.
    if (ctl_mode) {
        char legacy_lock_path[512];
        snprintf(legacy_lock_path, sizeof(legacy_lock_path), "%s/control-%d", gpu_cr_data_dir(), pid);
        int legacy_lock_fd = open(legacy_lock_path, O_CREAT | O_RDWR | O_CLOEXEC, 0644);
        if (legacy_lock_fd >= 0) {
            if (flock(legacy_lock_fd, LOCK_EX) != 0)
                fprintf(stderr, "Warning: flock(%s) failed: %s\n", legacy_lock_path, strerror(errno));
            // held (with the fd) until process exit
        }
    }
    if (flock(comm->fd_control, LOCK_EX) != 0)
        fprintf(stderr, "Warning: flock(control-%d) failed: %s\n", pid, strerror(errno));

    // Dest-path gate, BEFORE any signal: a .so that never advertised the
    // capability would silently serve the op from the per-PID buffer — for
    // a restore that replays stale bytes into GPU memory, which no post-op
    // check can undo. In ctl mode the advertisement (proto>=3 implies
    // dest-path support) already proved this without needing a prior -i.
    if (dest_path && !ctl_mode && !(comm->control->capability & CR_CAP_DEST_PATH)) {
        fprintf(stderr, "Error: vGPU.so has not advertised dest-path support "
                        "(run -i first, or upgrade the preloaded library)\n");
        exit(EXIT_REFUSED);
    }

    int ret;

    if(init) {
        comm->send_msg(INIT_MSG);
        kill(pid, CR_INIT_SIGNAL);
        if (!wait_finished(comm)) exit(EXIT_TIMEOUT);
    }  else if(ckpt && selective_spec) {
        selective_cr_request req;
        memset(&req, 0, sizeof(req));
        if (!parse_selective_regions(selective_spec, &req)) {
            fprintf(stderr, "Error: failed to parse selective regions\n");
            exit(EXIT_FAILURE);
        }
        printf("Selective checkpoint: %u regions\n", req.num_regions);
        for (uint32_t i = 0; i < req.num_regions; i++) {
            printf("  region %u: ptr=%p size=%lu\n", i, req.regions[i].ptr, req.regions[i].size);
        }
        // Always write the full v2 extension (zeroed dest when -o absent) so
        // a stale dest_path from an earlier op can never survive this write.
        req.proto_version = SELECTIVE_CR_PROTO_V2;
        if (dest_path) {
            if (!secure_precreate(dest_path)) exit(EXIT_OP_FAILED);
            strncpy(req.dest_path, dest_path, SELECTIVE_CR_MAX_PATH - 1);
        }
        comm->control->selective_req = req;
        comm->send_msg(SELECTIVE_CKPT_MSG);
        kill(pid, CR_CKPT_SIGNAL);
        printf("Selective dump signal sent.\n");
        if (!wait_finished(comm)) exit(EXIT_TIMEOUT);
        check_selective_result(comm, dest_path);
        if (dest_path && !validate_dest_dump(dest_path)) exit(EXIT_OP_FAILED);
        printf("Selective checkpointing done\n");
    } else if(restore && selective_spec) {
        selective_cr_request req;
        memset(&req, 0, sizeof(req));
        if (!parse_selective_regions(selective_spec, &req)) {
            fprintf(stderr, "Error: failed to parse selective regions\n");
            exit(EXIT_FAILURE);
        }
        printf("Selective restore: %u regions\n", req.num_regions);
        req.proto_version = SELECTIVE_CR_PROTO_V2;
        if (dest_path) {
            struct stat st;
            if (stat(dest_path, &st) != 0) {
                fprintf(stderr, "Error: restore source %s: %s\n", dest_path, strerror(errno));
                exit(EXIT_OP_FAILED);
            }
            strncpy(req.dest_path, dest_path, SELECTIVE_CR_MAX_PATH - 1);
        }
        comm->control->selective_req = req;
        comm->send_msg(SELECTIVE_RESTORE_MSG);
        kill(pid, CR_RESTORE_SIGNAL);
        printf("Selective restore signal sent.\n");
        if (!wait_finished(comm)) exit(EXIT_TIMEOUT);
        check_selective_result(comm, dest_path);
        printf("Selective restore done\n");
    } else if(ckpt) {
        comm->send_msg(CKPT_MSG);
        kill(pid, CR_CKPT_SIGNAL);
        printf("Dump signal sent.\n");
        if (!wait_finished(comm)) exit(EXIT_TIMEOUT);
        check_full_result(comm, "checkpoint");
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

        if (!wait_finished(comm)) exit(EXIT_TIMEOUT);
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("GPUos restore time: %.3f s\n", std::chrono::duration<double>(t2 - t1).count());

        printf("Process internal restoration finished.\n");
        printf("Restoring done\n");
#else
        comm->send_msg(RESTORE_MSG);
        kill(pid, CR_RESTORE_SIGNAL);
        if (!wait_finished(comm)) exit(EXIT_TIMEOUT);
        check_full_result(comm, "restore");
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
