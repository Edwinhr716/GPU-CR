/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_CHECKPOINT_H_
#define NCCL_INT_CHECKPOINT_H_

#include "comm.h"

ncclResult_t ncclCheckpointPluginPrepare(ncclComm_t comm, int flags);
ncclResult_t ncclCheckpointPluginRestore(ncclComm_t comm, int flags);

#endif
