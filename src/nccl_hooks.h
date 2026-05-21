/**
 * nccl_hooks.h - NCCL Communicator Interception and Lifecycle Management
 *
 * This module intercepts NCCL communicator creation/destruction via LD_PRELOAD
 * hooks and provides suspend/resume capabilities for multi-GPU checkpoint/restore.
 *
 * Key design:
 *   - ncclCommInitRank / ncclCommInitRankConfig are hooked to automatically track
 *     all NCCL communicators created by the application (e.g., PyTorch, vLLM).
 *   - ncclCommDestroy / ncclCommFinalize are hooked to untrack destroyed communicators.
 *   - nccl_suspend_all_comms() / nccl_resume_all_comms() call the NCCL built-in
 *     ncclCommSuspend(comm, NCCL_SUSPEND_MEM) / ncclCommResume(comm) via dlsym,
 *     which release/restore GPU physical memory while preserving virtual addresses
 *     and bootstrap connections.
 *   - A manual registration API (cr_register_nccl_comm / cr_unregister_nccl_comm)
 *     is provided for applications where LD_PRELOAD hooking doesn't work
 *     (e.g., statically linked NCCL).
 *
 * Requirements:
 *   - NCCL must be loaded dynamically (libnccl.so) for hooks to work.
 *   - NCCL_CUMEM_ENABLE=1 must be set for ncclCommSuspend(NCCL_SUSPEND_MEM) to work.
 *   - ncclCommSuspend is a collective operation: all ranks must call it simultaneously.
 */

#ifndef NCCL_HOOKS_H
#define NCCL_HOOKS_H

#include <cstddef>

// ---------------------------------------------------------------------------
// Forward-declared NCCL types (ABI-compatible, no nccl.h dependency)
// ---------------------------------------------------------------------------
typedef struct ncclComm* ncclComm_t;

// ---------------------------------------------------------------------------
// NCCL hooks public API (called from vGPU.cpp signal handlers)
// ---------------------------------------------------------------------------

/**
 * Suspend all tracked NCCL communicators.
 * Calls ncclCommSuspend(comm, NCCL_SUSPEND_MEM) for each tracked comm.
 * This is a collective operation — all ranks in each communicator must call
 * it simultaneously, coordinated by the multi_cr_client sending signals
 * to ALL processes before waiting.
 *
 * @return  0 on success
 *         -1 on error (ncclCommSuspend failed or not available)
 *          1 if no communicators are tracked (no-op)
 */
int nccl_suspend_all_comms();

/**
 * Resume all tracked NCCL communicators.
 * Calls ncclCommResume(comm) for each tracked comm.
 * Also collective — all ranks must call simultaneously.
 *
 * @return  0 on success, -1 on error, 1 if no comms tracked
 */
int nccl_resume_all_comms();

/**
 * Register a communicator manually (thread-safe, idempotent).
 * Use this when LD_PRELOAD hooks cannot intercept ncclCommInitRank
 * (e.g., NCCL is statically linked or loaded via dlopen(RTLD_LOCAL)).
 */
void nccl_register_comm(ncclComm_t comm);

/**
 * Unregister a communicator manually (thread-safe).
 */
void nccl_unregister_comm(ncclComm_t comm);

/** Get number of currently tracked communicators. */
int nccl_get_comm_count();

/** Check if ncclCommSuspend/Resume are available in the loaded NCCL library. */
bool nccl_suspend_available();

#endif // NCCL_HOOKS_H
