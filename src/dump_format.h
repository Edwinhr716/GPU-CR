#ifndef GPU_CR_SRC_DUMP_FORMAT_H_
#define GPU_CR_SRC_DUMP_FORMAT_H_

// Destination-file dump layout helpers.
//
// A dump is a shared_mem_fs header (2MiB-rounded), the extents, and a
// trailing DumpCommit marker at current_offset. The marker is written
// last, so its absence identifies a torn dump. These checks are shared
// by cr_client (post-checkpoint validation, via read(2)) and the .so
// (pre-restore validation, via mmap), and are pure so unit tests can
// exercise them without a GPU.

#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common.h"

namespace gpu_cr {

// Bounds-checks a dump header against the containing file/mapping size.
// Does NOT check the commit marker (callers read it from its offset).
inline bool DumpHeaderPlausible(uint64_t file_num, uint64_t current_offset,
                                uint64_t total_size) {
  const uint64_t header_size = ROUND_UP_2MB(sizeof(shared_mem_fs));
  // The subtraction form (not current_offset + sizeof(DumpCommit)) keeps a
  // header-supplied offset near UINT64_MAX from wrapping past the bound.
  return file_num > 0 && file_num < MAX_FILE_NUM &&
         total_size >= sizeof(DumpCommit) &&
         current_offset >= header_size &&
         current_offset <= total_size - sizeof(DumpCommit);
}

// Validates a whole dump file through an open fd using pread(2) — no
// mmap, so no hugetlb reservation or fault lands in the caller's cgroup.
inline bool ValidateDumpFd(int fd) {
  struct stat st;
  uint64_t hdr[2];  // file_num, current_offset
  DumpCommit commit;
  return fstat(fd, &st) == 0 &&
         pread(fd, hdr, sizeof(hdr), 0) ==
             static_cast<ssize_t>(sizeof(hdr)) &&
         DumpHeaderPlausible(hdr[0], hdr[1],
                             static_cast<uint64_t>(st.st_size)) &&
         pread(fd, &commit, sizeof(commit), static_cast<off_t>(hdr[1])) ==
             static_cast<ssize_t>(sizeof(commit)) &&
         commit.magic == kDumpCommitMagic;
}

}  // namespace gpu_cr

#endif  // GPU_CR_SRC_DUMP_FORMAT_H_
