// The header lives at nccl_patches/nccl_checkpoint.h in this repo. After the
// patch is applied to NCCL upstream it sits at src/include/plugin/nccl_checkpoint.h
// — adjust the include if you reuse this file inside an NCCL source tree.
#include "nccl_checkpoint.h"

#include <dlfcn.h>
#include <cstdio>

typedef int (*gcr_checkpoint_prepare_fn)(int flags);
typedef int (*gcr_checkpoint_restore_export_fn)(int flags);
typedef int (*gcr_checkpoint_restore_import_fn)(int flags);
typedef int (*gcr_checkpoint_validate_fn)(int flags);

struct GcrCheckpointContext {
  ncclDebugLogger_t log;
  gcr_checkpoint_prepare_fn prepare;
  gcr_checkpoint_restore_export_fn restoreExport;
  gcr_checkpoint_restore_import_fn restoreImport;
  gcr_checkpoint_validate_fn validate;
};

static void* resolve_required_symbol(const char* name) {
  void* sym = dlsym(RTLD_DEFAULT, name);
  if (sym == nullptr) {
    std::fprintf(stderr, "[NCCL-GCR-PLUGIN] missing runtime symbol %s: %s\n", name, dlerror());
  }
  return sym;
}

static ncclResult_t gcrPluginInit(void** context, ncclDebugLogger_t logFunction) {
  GcrCheckpointContext* ctx = new GcrCheckpointContext();
  ctx->log = logFunction;
  ctx->prepare = reinterpret_cast<gcr_checkpoint_prepare_fn>(resolve_required_symbol("gcr_checkpoint_prepare"));
  ctx->restoreExport = reinterpret_cast<gcr_checkpoint_restore_export_fn>(resolve_required_symbol("gcr_checkpoint_restore_export"));
  ctx->restoreImport = reinterpret_cast<gcr_checkpoint_restore_import_fn>(resolve_required_symbol("gcr_checkpoint_restore_import"));
  ctx->validate = reinterpret_cast<gcr_checkpoint_validate_fn>(dlsym(RTLD_DEFAULT, "gcr_checkpoint_validate"));

  if (ctx->prepare == nullptr || ctx->restoreExport == nullptr || ctx->restoreImport == nullptr) {
    delete ctx;
    return ncclSystemError;
  }

  *context = ctx;
  return ncclSuccess;
}

static ncclResult_t gcrPluginPrepare(void* context, ncclComm_t comm, int flags) {
  (void)comm;
  GcrCheckpointContext* ctx = reinterpret_cast<GcrCheckpointContext*>(context);
  return ctx->prepare(flags) == 0 ? ncclSuccess : ncclSystemError;
}

static ncclResult_t gcrPluginRestore(void* context, ncclComm_t comm, int flags) {
  (void)comm;
  GcrCheckpointContext* ctx = reinterpret_cast<GcrCheckpointContext*>(context);
  if (flags & NCCL_CKPT_RESTORE_EXPORT) {
    return ctx->restoreExport(flags) == 0 ? ncclSuccess : ncclSystemError;
  }
  if (flags & NCCL_CKPT_RESTORE_IMPORT) {
    return ctx->restoreImport(flags) == 0 ? ncclSuccess : ncclSystemError;
  }
  return ncclInvalidUsage;
}

static ncclResult_t gcrPluginFinalize(void* context) {
  delete reinterpret_cast<GcrCheckpointContext*>(context);
  return ncclSuccess;
}

extern "C" {
ncclCheckpointPlugin_v1_t ncclCheckpointPlugin_v1 = {
  "gcr",
  gcrPluginInit,
  gcrPluginPrepare,
  gcrPluginRestore,
  gcrPluginFinalize
};
}
