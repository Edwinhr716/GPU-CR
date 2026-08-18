#ifndef GPU_CR_CTL_PATH_H
#define GPU_CR_CTL_PATH_H

// GEP-0006: the control plane (control-<pid>, ctl-ready-<pid>, pid_map_*,
// the id counter) moves off hugetlbfs onto a caller-provided tmpfs, so no
// hugetlb page is ever faulted from the coordinating process's cgroup.
// Dump/staging DATA files stay in EXPORT_FILE_PATH untouched.
//
// GPU_CR_CTL_PATH unset            -> legacy: ctl dir == data dir.
// GPU_CR_CTL_PATH set, tmpfs       -> ctl mode.
// GPU_CR_CTL_PATH set, NOT tmpfs   -> legacy fallback (the .so must never
//     write an advertisement into a disk-backed hostPath dir that a later
//     tmpfs mount would shadow); cr_client refuses instead, so the
//     misconfiguration is loud on the coordinator side.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/vfs.h>
#include <unistd.h>

#define GPU_CR_TMPFS_MAGIC 0x01021994UL
#define GPU_CR_CTL_PROTO 3
#define ROUND_UP_4K(x) (((x) + 4095UL) & ~4095UL)

static inline const char* gpu_cr_data_dir(void) {
    const char* d = getenv("EXPORT_FILE_PATH");
    return (d && d[0]) ? d : "/mnt/huge-ckpt";
}

static inline int gpu_cr_dir_is_tmpfs(const char* dir) {
    struct statfs sfs;
    if (statfs(dir, &sfs) != 0) return 0;
    return (unsigned long)sfs.f_type == GPU_CR_TMPFS_MAGIC;
}

static inline const char* gpu_cr_ctl_dir(int* ctl_mode) {
    const char* p = getenv("GPU_CR_CTL_PATH");
    if (p && p[0] && gpu_cr_dir_is_tmpfs(p)) {
        if (ctl_mode) *ctl_mode = 1;
        return p;
    }
    if (p && p[0])
        fprintf(stderr, "[gpu-cr] GPU_CR_CTL_PATH=%s missing or not tmpfs; using legacy control dir %s\n",
                p, gpu_cr_data_dir());
    if (ctl_mode) *ctl_mode = 0;
    return gpu_cr_data_dir();
}

// starttime = field 22 of /proc/<pid>/stat: the PID-reuse guard for the
// readiness advertisement (a recycled PID gets a new starttime, and
// signaling an innocent recycled PID is termination by default).
static inline long long gpu_cr_starttime(pid_t pid) {
    char path[64], buf[4096];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return -1;
    buf[n] = '\0';
    char* p = strrchr(buf, ')'); // comm may contain spaces/parens
    if (!p || p[1] == '\0') return -1;
    p += 2;
    long long val = -1;
    int field = 3;
    char* save = NULL;
    for (char* tok = strtok_r(p, " ", &save); tok; tok = strtok_r(NULL, " ", &save), field++) {
        if (field == 22) {
            val = atoll(tok);
            break;
        }
    }
    return val;
}

#endif
