/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <mutex>
#include <stdlib.h>
#include <string.h>

#include "checks.h"
#include "checkpoint.h"
#include "debug.h"
#include "env.h"
#include "os.h"
#include "plugin.h"
#include "plugin/nccl_checkpoint.h"

enum {
  checkpointPluginLoadFailed  = -1,
  checkpointPluginLoadReady   =  0,
  checkpointPluginLoadSuccess =  1,
};

static std::mutex checkpointPluginMutex;
static int checkpointPluginStatus = checkpointPluginLoadReady;
static void* checkpointPluginLib = nullptr;
static ncclCheckpointPlugin_t* checkpointPlugin = nullptr;
static void* checkpointPluginContext = nullptr;
static bool checkpointPluginFinalizeRegistered = false;

static ncclCheckpointPlugin_t* getNcclCheckpoint_v1(void* lib) {
  return (ncclCheckpointPlugin_t*)ncclOsDlsym(lib, NCCL_CHECKPOINT_PLUGIN_SYMBOL);
}

static void ncclCheckpointPluginFinalize(void) {
  std::lock_guard<std::mutex> lock(checkpointPluginMutex);
  if (checkpointPlugin && checkpointPlugin->finalize) {
    INFO(NCCL_DESTROY, "CHECKPOINT/Plugin: Closing checkpoint plugin %s", checkpointPlugin->name);
    checkpointPlugin->finalize(checkpointPluginContext);
  }
  checkpointPluginContext = nullptr;
  checkpointPlugin = nullptr;
  if (checkpointPluginLib) {
    ncclClosePluginLib(checkpointPluginLib, ncclPluginTypeCheckpoint);
    checkpointPluginLib = nullptr;
  }
  checkpointPluginStatus = checkpointPluginLoadReady;
}

static ncclResult_t ncclCheckpointPluginLoad(void) {
  const char* checkpointName = nullptr;

  if (checkpointPluginLoadFailed == checkpointPluginStatus) return ncclInvalidUsage;
  if (checkpointPluginLoadSuccess == checkpointPluginStatus) return ncclSuccess;

  std::lock_guard<std::mutex> lock(checkpointPluginMutex);
  if (checkpointPluginLoadSuccess == checkpointPluginStatus) return ncclSuccess;
  if (checkpointPluginLoadFailed == checkpointPluginStatus) return ncclInvalidUsage;

  if ((checkpointName = ncclGetEnv("NCCL_CHECKPOINT_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV, "NCCL_CHECKPOINT_PLUGIN set by environment to %s", checkpointName);
    if (strcasecmp(checkpointName, "none") == 0) goto fail;
  }

  checkpointPluginLib = ncclOpenCheckpointPluginLib(checkpointName);
  if (checkpointPluginLib == nullptr) goto fail;
  if (ncclPluginLibPaths[ncclPluginTypeCheckpoint]) checkpointName = ncclPluginLibPaths[ncclPluginTypeCheckpoint];

  checkpointPlugin = getNcclCheckpoint_v1(checkpointPluginLib);
  if (checkpointPlugin == nullptr) {
    if (checkpointName) INFO(NCCL_INIT, "External checkpoint plugin %s is unsupported", checkpointName);
    goto fail;
  }
  if (checkpointPlugin->init == nullptr || checkpointPlugin->prepare == nullptr || checkpointPlugin->restore == nullptr) {
    INFO(NCCL_INIT, "External checkpoint plugin %s has missing required callbacks",
         checkpointName ? checkpointName : checkpointPlugin->name);
    goto fail;
  }

  if (checkpointPlugin->init(&checkpointPluginContext, ncclDebugLog) != ncclSuccess) {
    INFO(NCCL_INIT, "External checkpoint plugin %s init failed",
         checkpointName ? checkpointName : checkpointPlugin->name);
    goto fail;
  }
  checkpointPluginStatus = checkpointPluginLoadSuccess;
  if (!checkpointPluginFinalizeRegistered) {
    atexit(ncclCheckpointPluginFinalize);
    checkpointPluginFinalizeRegistered = true;
  }

  INFO(NCCL_INIT, "Successfully loaded external checkpoint plugin %s",
       checkpointName ? checkpointName : checkpointPlugin->name);
  return ncclSuccess;

fail:
  if (checkpointPluginLib) {
    ncclClosePluginLib(checkpointPluginLib, ncclPluginTypeCheckpoint);
  }
  checkpointPluginLib = nullptr;
  checkpointPlugin = nullptr;
  checkpointPluginContext = nullptr;
  checkpointPluginStatus = checkpointPluginLoadFailed;
  return ncclInvalidUsage;
}

ncclResult_t ncclCheckpointPluginPrepare(ncclComm_t comm, int flags) {
  NCCLCHECK(ncclCheckpointPluginLoad());
  if (checkpointPlugin == nullptr) return ncclInvalidUsage;
  return checkpointPlugin->prepare(checkpointPluginContext, comm, flags);
}

ncclResult_t ncclCheckpointPluginRestore(ncclComm_t comm, int flags) {
  NCCLCHECK(ncclCheckpointPluginLoad());
  if (checkpointPlugin == nullptr) return ncclInvalidUsage;
  return checkpointPlugin->restore(checkpointPluginContext, comm, flags);
}
