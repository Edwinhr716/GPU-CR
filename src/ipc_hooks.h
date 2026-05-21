/**
 * ipc_hooks.h - cuMem IPC Interception and Management for Checkpoint/Restore
 *
 * This module intercepts CUDA Driver API cuMem IPC functions via hooking
 * cudaGetDriverEntryPoint/cudaGetDriverEntryPointByVersion. When NCCL (or any
 * library) requests cuMem function pointers at runtime, we return our hook
 * functions that record all IPC export/import/map operations.
 *
 * During checkpoint:
 *   ipc_teardown_all_imports()            - unmap all imported IPC mappings
 *   ipc_save_and_teardown_all_exports()   - save GPU data to host buffer,
 *       then cuMemUnmap + cuMemRelease + cuMemAddressFree
 *   → each process's CUDA state is "clean" for cuda-checkpoint
 *
 * During restore:
 *   ipc_rebuild_and_restore_all_exports() - cuMemCreate + cuMemMap +
 *       cuMemSetAccess + restore data + re-export (new fds)
 *   ipc_rebuild_all_imports()             - pidfd_getfd + re-import + re-map
 *   → NCCL's internal pointers become valid again
 *
 * Key detail: Teardown calls cuMemAddressFree to fully remove IPC state
 * (required for cuda-checkpoint). Rebuild calls cuMemAddressReserve to
 * reclaim the same VA before re-mapping, so NCCL pointers remain valid.
 *
 * This module completely replaces ncclCommSuspend/Resume and does NOT require
 * any NCCL source modifications.
 */

#ifndef IPC_HOOKS_H
#define IPC_HOOKS_H

#include <cstddef>
#include <vector>

struct IpcTimingSnapshot {
    long import_teardown_ms;
    int import_teardown_count;
    size_t import_teardown_bytes;

    long export_teardown_dtoh_ms;
    long export_teardown_fd_close_ms;
    long export_teardown_unmap_release_ms;
    int export_teardown_count;
    size_t export_teardown_bytes;

    long local_teardown_dtoh_ms;
    long local_teardown_unmap_release_ms;
    int local_teardown_count;
    size_t local_teardown_bytes;

    long export_rebuild_alloc_map_access_ms;
    long export_rebuild_htod_ms;
    long export_rebuild_reexport_ms;
    long export_rebuild_total_ms;
    int export_rebuild_count;
    size_t export_rebuild_bytes;

    long local_rebuild_alloc_map_access_ms;
    long local_rebuild_htod_ms;
    long local_rebuild_total_ms;
    int local_rebuild_count;
    size_t local_rebuild_bytes;

    long import_rebuild_ms;
    int import_rebuild_count;
    size_t import_rebuild_bytes;
};

void ipc_reset_timing_snapshot();
void ipc_get_timing_snapshot(IpcTimingSnapshot* snapshot);

// ---------------------------------------------------------------------------
// IPC state management - called from vGPU.cpp signal handlers
// ---------------------------------------------------------------------------

/**
 * Teardown all imported IPC mappings before checkpoint.
 * Calls cuMemUnmap + cuMemRelease for each import, but preserves VA space.
 * @return number of imports torn down, or -1 on error
 */
int ipc_teardown_all_imports();

/**
 * Save all export-side GPU data to a host buffer, then fully tear down
 * the export allocations (cuMemUnmap + cuMemRelease + cuMemAddressFree).
 *
 * This removes ALL IPC-related CUDA driver state so cuda-checkpoint can
 * freeze cleanly.  The saved data and metadata are used by
 * ipc_rebuild_and_restore_all_exports() during restore.
 *
 * @param host_buf   Host buffer to save export GPU data into.
 *                   Must have at least ipc_get_export_data_size() bytes.
 * @param buf_size   Size of host_buf in bytes.
 * @return number of exports torn down, or -1 on error.
 */
int ipc_save_and_teardown_all_exports(void* host_buf, size_t buf_size);

/**
 * Query the total host buffer size needed to save all export data.
 * Call before ipc_save_and_teardown_all_exports() to allocate buffer.
 */
size_t ipc_get_export_data_size();

/**
 * After cuda-checkpoint restore: re-create all export allocations at
 * their original virtual addresses, restore GPU data from the host buffer,
 * and re-export to produce new shareable fds.
 *
 * @param host_buf   Host buffer containing data saved by
 *                   ipc_save_and_teardown_all_exports().
 * @param buf_size   Size of host_buf in bytes.
 * @return number of exports rebuilt, or -1 on error.
 */
int ipc_rebuild_and_restore_all_exports(void* host_buf, size_t buf_size);

/**
 * After cuda-checkpoint restore + re-export: re-import peer memory and
 * map back to the original virtual addresses.
 *
 * For each import record, reads the peer's new export info from shared memory
 * and uses pidfd_getfd() to obtain the fd, then cuMemImportFromShareableHandle
 * + cuMemMap to the original VA.
 *
 * @param peer_pids  PIDs of all peer worker processes
 * @return number of imports rebuilt, or -1 on error
 */
int ipc_rebuild_all_imports(const std::vector<int>& peer_pids);

// ---------------------------------------------------------------------------
// State query (for diagnostics)
// ---------------------------------------------------------------------------

/** Number of currently tracked IPC import records */
int ipc_get_import_count();

/** Number of currently tracked IPC export records */
int ipc_get_export_count();

/** Dump all IPC state to stderr for debugging */
void ipc_dump_state();

/** Dump /proc/self/fd to find remaining nvidia/cuda fds (diagnostic) */
void ipc_dump_nvidia_fds(const char* label);

/**
 * Validate all export/import mappings by probing each VA with cuMemcpyDtoH.
 * @return number of errors found (0 = all OK)
 */
int ipc_validate_all_mappings(const char* label);

/**
 * Get total size needed to save all non-exported cuMem allocs.
 */
size_t ipc_get_local_alloc_data_size();

/**
 * Save and teardown all non-exported cuMem allocs (IPC-capable but not
 * exported). These must be torn down before cuda-checkpoint because the
 * driver cannot restore cuMem VMM allocations with IPC handle types.
 */
int ipc_save_and_teardown_local_allocs(void* host_buf, size_t buf_size);

/**
 * Rebuild all non-exported cuMem allocs after cuda-checkpoint restore.
 */
int ipc_rebuild_local_allocs(void* host_buf, size_t buf_size);

/**
 * Teardown all IPC events (cudaIpcGetEventHandle / cudaIpcOpenEventHandle).
 * Destroys imported (opened) events; skips exported (owned) events.
 * @return number of events processed, or -1 on error
 */
int ipc_teardown_all_events();

/** Number of currently tracked IPC event records */
int ipc_get_event_count();

/**
 * Disable all tracked cudaDeviceEnablePeerAccess state before
 * cuda-checkpoint. P2P peer access is driver state that cuda-checkpoint may
 * reject even when NCCL's cuMem IPC mappings have been torn down.
 *
 * @return number of peer entries disabled.
 */
int ipc_disable_all_peer_access();

/**
 * Re-enable peer access entries saved by ipc_disable_all_peer_access().
 *
 * @return number of peer entries re-enabled.
 */
int ipc_reenable_all_peer_access();

// ---------------------------------------------------------------------------
// Shared memory exchange structures for IPC rebuild
// ---------------------------------------------------------------------------

// Maximum IPC exports per process (for shared memory layout)
#define IPC_MAX_EXPORTS_PER_PROC 64

/**
 * Per-export entry written to shared memory during rebuild.
 * The exporting process writes these; the importing process reads them
 * and uses pidfd_getfd to copy the fd.
 */
struct IpcExportShmEntry {
    int      owner_pid;      // PID of the exporting process
    int      fd;             // fd number in the exporting process
    void*    local_ptr;      // exporter's virtual address of the allocation
    size_t   size;           // allocation size
    int      valid;          // 1 if this entry is valid
};

/**
 * Per-process shared memory block for IPC rebuild.
 * Each worker writes its export entries here after re-exporting.
 */
struct IpcRebuildShmBlock {
    int                num_exports;
    IpcExportShmEntry  entries[IPC_MAX_EXPORTS_PER_PROC];
};

/**
 * Write this process's export info to the given shared memory block.
 * Called during IPC rebuild phase after re-exporting.
 */
int ipc_write_export_info_to_shm(IpcRebuildShmBlock* block);

/**
 * Read a peer's export info from shared memory and rebuild imports.
 * Uses pidfd_getfd to obtain fds from the peer process.
 */
int ipc_import_from_shm_block(const IpcRebuildShmBlock* block);

#endif // IPC_HOOKS_H
