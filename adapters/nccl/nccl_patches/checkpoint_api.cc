/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * GPU-CR public checkpoint API entry points (for stock upstream NCCL).
 *
 * Copy to ${NCCL_SRC}/src/checkpoint_api.cc and add `checkpoint_api.cc` to
 * the LIBSRCFILES list in ${NCCL_SRC}/src/Makefile. Requires the macro and
 * declaration block from nccl.h.in.checkpoint-additions to be pasted into
 * ${NCCL_SRC}/src/nccl.h.in first (see README.md in this directory).
 *
 * NOTE: GCR's full NCCL tree already contains these entry points (inside
 * mem_manager.cc) — this file is only for patching stock upstream NCCL.
 * Do not add it to a tree that already defines ncclCommCheckpointPrepare.
 *
 * NCCL_CKPT_MODE_NATIVE forwards to ncclCommSuspend/Resume (present in
 * upstream NCCL since ~2.30 and in GCR's full tree; detected via the
 * NCCL_SUSPEND_MEM macro). On older NCCL without suspend support, NATIVE
 * mode returns ncclInvalidUsage — use NCCL_CKPT_MODE_GCR_GLOBAL, which
 * dispatches to the external checkpoint plugin (loaded from
 * $NCCL_CHECKPOINT_PLUGIN by src/plugin/checkpoint.cc).
 */

#include "comm.h"
#include "debug.h"
#include "checkpoint.h"

NCCL_API(ncclResult_t, ncclCommCheckpointPrepare, ncclComm_t comm, int flags);
ncclResult_t ncclCommCheckpointPrepare(ncclComm_t comm, int flags) {
  int mode = flags & NCCL_CKPT_MODE_MASK;
  int knownFlags = NCCL_CKPT_MODE_MASK | NCCL_CKPT_OPT_SAVE_DATA | NCCL_CKPT_OPT_NO_CUDA_CKPT | NCCL_CKPT_OPT_VALIDATE;
  if (mode == NCCL_CKPT_MODE_NATIVE && (flags & ~NCCL_CKPT_MODE_NATIVE) == 0) {
#ifdef NCCL_SUSPEND_MEM
    INFO(NCCL_INIT, "ncclCommCheckpointPrepare: using native suspend backend");
    return ncclCommSuspend(comm, NCCL_SUSPEND_MEM);
#else
    WARN("ncclCommCheckpointPrepare: NATIVE mode requires ncclCommSuspend "
         "(upstream NCCL >= 2.30 or GCR's full tree); use NCCL_CKPT_MODE_GCR_GLOBAL");
    return ncclInvalidUsage;
#endif
  }

  if (mode == NCCL_CKPT_MODE_GCR_GLOBAL) {
    if (flags & ~knownFlags) {
      WARN("ncclCommCheckpointPrepare: unsupported flags 0x%x", flags);
      return ncclInvalidUsage;
    }
    INFO(NCCL_INIT, "ncclCommCheckpointPrepare: dispatching to checkpoint plugin");
    return ncclCheckpointPluginPrepare(comm, flags);
  } else {
    WARN("ncclCommCheckpointPrepare: unsupported flags 0x%x", flags);
  }
  return ncclInvalidUsage;
}

NCCL_API(ncclResult_t, ncclCommCheckpointRestore, ncclComm_t comm, int flags);
ncclResult_t ncclCommCheckpointRestore(ncclComm_t comm, int flags) {
  int mode = flags & NCCL_CKPT_MODE_MASK;
  int restorePhase = flags & (NCCL_CKPT_RESTORE_EXPORT | NCCL_CKPT_RESTORE_IMPORT);
  int knownFlags = NCCL_CKPT_MODE_MASK | NCCL_CKPT_OPT_SAVE_DATA | NCCL_CKPT_OPT_NO_CUDA_CKPT |
      NCCL_CKPT_OPT_VALIDATE | NCCL_CKPT_RESTORE_EXPORT | NCCL_CKPT_RESTORE_IMPORT;
  if (mode == NCCL_CKPT_MODE_NATIVE && (flags & ~NCCL_CKPT_MODE_NATIVE) == 0) {
#ifdef NCCL_SUSPEND_MEM
    INFO(NCCL_INIT, "ncclCommCheckpointRestore: using native resume backend");
    return ncclCommResume(comm);
#else
    WARN("ncclCommCheckpointRestore: NATIVE mode requires ncclCommResume "
         "(upstream NCCL >= 2.30 or GCR's full tree); use NCCL_CKPT_MODE_GCR_GLOBAL");
    return ncclInvalidUsage;
#endif
  }

  if (mode == NCCL_CKPT_MODE_GCR_GLOBAL) {
    if ((flags & ~knownFlags) || restorePhase == 0 ||
        restorePhase == (NCCL_CKPT_RESTORE_EXPORT | NCCL_CKPT_RESTORE_IMPORT)) {
      WARN("ncclCommCheckpointRestore: unsupported flags 0x%x", flags);
      return ncclInvalidUsage;
    }
    INFO(NCCL_INIT, "ncclCommCheckpointRestore: dispatching to checkpoint plugin");
    return ncclCheckpointPluginRestore(comm, flags);
  } else {
    WARN("ncclCommCheckpointRestore: unsupported flags 0x%x", flags);
  }
  return ncclInvalidUsage;
}
