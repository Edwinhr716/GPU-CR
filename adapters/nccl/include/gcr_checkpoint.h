/*
 * GCR checkpoint runtime C ABI.
 *
 * This ABI is intentionally independent from NCCL. The NCCL checkpoint plugin
 * locates these symbols with dlsym(RTLD_DEFAULT), so the runtime can live in an
 * LD_PRELOAD library and keep all IPC/VMM tracking outside NCCL core.
 */

#ifndef GCR_CHECKPOINT_H
#define GCR_CHECKPOINT_H

#ifdef __cplusplus
extern "C" {
#endif

int gcr_checkpoint_prepare(int flags);
int gcr_checkpoint_restore_export(int flags);
int gcr_checkpoint_restore_import(int flags);
int gcr_checkpoint_validate(int flags);

#ifdef __cplusplus
}
#endif

#endif
