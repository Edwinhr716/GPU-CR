/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHECKPOINT_H_
#define NCCL_CHECKPOINT_H_

#include "nccl.h"
#include "nccl_common.h"

#define NCCL_CHECKPOINT_PLUGIN_SYMBOL "ncclCheckpointPlugin_v1"

typedef struct ncclCheckpointPlugin_v1 {
  const char* name;
  ncclResult_t (*init)(void** context, ncclDebugLogger_t logFunction);
  ncclResult_t (*prepare)(void* context, ncclComm_t comm, int flags);
  ncclResult_t (*restore)(void* context, ncclComm_t comm, int flags);
  ncclResult_t (*finalize)(void* context);
} ncclCheckpointPlugin_v1_t;

typedef ncclCheckpointPlugin_v1_t ncclCheckpointPlugin_t;

#endif // end include guard
