#ifndef GPU_CR_SRC_CTL_PATH_H_
#define GPU_CR_SRC_CTL_PATH_H_

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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <unistd.h>

namespace gpu_cr {

inline constexpr unsigned long kTmpfsMagic = 0x01021994UL;

// Control-channel protocol level advertised by the .so and required by
// cr_client. Proto >= 3 implies GEP-0006 advertisements and GEP-0001
// destination-path support.
inline constexpr int kCtlProto = 3;

constexpr size_t RoundUp4K(size_t x) { return (x + 4095UL) & ~4095UL; }

// Directory holding dump/staging DATA files (unchanged by GEP-0006).
inline const char* DataDir() {
  const char* dir = getenv("EXPORT_FILE_PATH");
  return (dir && dir[0]) ? dir : "/mnt/huge-ckpt";
}

inline bool DirIsTmpfs(const char* dir) {
  struct statfs sfs;
  if (statfs(dir, &sfs) != 0) return false;
  return static_cast<unsigned long>(sfs.f_type) == kTmpfsMagic;
}

// Resolves the control-plane directory. Sets *ctl_mode to true when a
// valid tmpfs-backed GPU_CR_CTL_PATH is in effect, false for legacy mode.
inline const char* CtlDir(bool* ctl_mode) {
  const char* path = getenv("GPU_CR_CTL_PATH");
  if (path && path[0] && DirIsTmpfs(path)) {
    if (ctl_mode) *ctl_mode = true;
    return path;
  }
  if (path && path[0]) {
    fprintf(stderr,
            "[gpu-cr] GPU_CR_CTL_PATH=%s missing or not tmpfs; using legacy "
            "control dir %s\n",
            path, DataDir());
  }
  if (ctl_mode) *ctl_mode = false;
  return DataDir();
}

// Extracts starttime (field 22 of /proc/<pid>/stat) from a stat line.
// Parses backwards from the last ')' because the comm field may contain
// spaces and parentheses. Returns -1 if the buffer is not a stat line.
inline long long ParseStarttimeFromStat(const char* stat_line) {
  const char* p = strrchr(stat_line, ')');
  if (!p || p[1] == '\0') return -1;
  p += 2;
  char buf[4096];
  strncpy(buf, p, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  long long val = -1;
  int field = 3;
  char* save = nullptr;
  for (char* tok = strtok_r(buf, " ", &save); tok;
       tok = strtok_r(nullptr, " ", &save), field++) {
    if (field == 22) {
      val = atoll(tok);
      break;
    }
  }
  return val;
}

// starttime of a live PID: the PID-reuse guard for the readiness
// advertisement (a recycled PID gets a new starttime, and signaling an
// innocent recycled PID is termination by default). Returns -1 when the
// PID does not exist.
inline long long ProcStarttime(pid_t pid) {
  char path[64];
  char buf[4096];
  snprintf(path, sizeof(path), "/proc/%d/stat", pid);
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  if (n == 0) return -1;
  buf[n] = '\0';
  return ParseStarttimeFromStat(buf);
}

// Readiness advertisement written by the .so at load (GEP-0006) and read
// back by cr_client before it signals. KEP-0002 appended the buffer-sizing
// keys so the agent can observe workload sizing.
struct Advertisement {
  int proto = 0;
  long long starttime = -1;
  char ctl[512] = "";
};

// Parses "proto=<n> starttime=<n> ctl=<path> ..." (unknown trailing keys
// ignored). Returns false only for an empty/absent proto line; individual
// fields keep their defaults when missing.
inline bool ParseAdvertisement(const char* line, Advertisement* out) {
  if (!line || !line[0]) return false;
  return sscanf(line, "proto=%d starttime=%lld ctl=%511s", &out->proto,
                &out->starttime, out->ctl) >= 1;
}

// Writes the ctl-ready-<pid> advertisement (GEP-0006), including the
// KEP-0002 buffer-sizing keys so the agent can observe workload sizing.
// Called from the .so's constructor — failures are reported, never fatal:
// this must not take down the workload. Returns false on write failure.
inline bool WriteAdvertisement(const char* ctl_dir, pid_t pid, size_t shm_mb,
                               size_t staging_mb, bool deferred) {
  char ready_path[512];
  char content[600];
  snprintf(ready_path, sizeof(ready_path), "%s/ctl-ready-%d", ctl_dir,
           static_cast<int>(pid));
  int content_len = snprintf(
      content, sizeof(content),
      "proto=%d starttime=%lld ctl=%s shm_mb=%zu staging_mb=%zu deferred=%d\n",
      kCtlProto, ProcStarttime(pid), ctl_dir, shm_mb, staging_mb,
      deferred ? 1 : 0);
  int fd = open(ready_path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
  if (fd < 0) {
    fprintf(stderr, "[vGPU] WARNING: cannot write %s (%s) — cr_client will "
                    "refuse ops\n",
            ready_path, strerror(errno));
    return false;
  }
  bool ok = write(fd, content, content_len) == content_len;
  if (!ok)
    fprintf(stderr, "[vGPU] WARNING: short write to %s (%s)\n", ready_path,
            strerror(errno));
  close(fd);
  return ok;
}

}  // namespace gpu_cr

#endif  // GPU_CR_SRC_CTL_PATH_H_
