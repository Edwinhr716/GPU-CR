/**
 * nccl_hooks.cpp - NCCL Communicator Interception (tracking only)
 *
 * This file implements LD_PRELOAD hooks for ncclCommInitRank,
 * ncclCommInitRankConfig, ncclCommDestroy and ncclCommFinalize to
 * automatically track all NCCL communicator handles.
 *
 * NOTE: ncclCommSuspend/Resume are NO LONGER used. IPC state management
 * is handled by ipc_hooks.cpp via cudaGetDriverEntryPoint hook.
 * This file is kept for communicator tracking which may be useful for
 * diagnostics and future features.
 */

#include "nccl_hooks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>    // PATH_MAX
#include <string>
#include <dlfcn.h>
#include <mutex>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// NCCL ABI type definitions (no nccl.h required)
// These match the NCCL C ABI across all 2.x versions.
// ---------------------------------------------------------------------------
#define NCCL_UNIQUE_ID_BYTES 128

typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

// ncclResult_t is an enum in nccl.h but ABI-compatible with int.
// We use int to avoid cross-version enum definition issues.
#define NCCL_OK       0   // ncclSuccess
#define NCCL_ERR_USAGE 5  // ncclInvalidUsage

// ncclCommSuspend flag
#define NCCL_SUSPEND_MEM_FLAG 0x01

// ---------------------------------------------------------------------------
// Function pointer types for NCCL APIs resolved at runtime
// ---------------------------------------------------------------------------
typedef int (*ncclCommInitRank_fn)(ncclComm_t*, int, ncclUniqueId, int);
typedef int (*ncclCommInitRankConfig_fn)(ncclComm_t*, int, ncclUniqueId, int, void*);
typedef int (*ncclCommInitAll_fn)(ncclComm_t*, int, const int*);
typedef int (*ncclCommDestroy_fn)(ncclComm_t);
typedef int (*ncclCommFinalize_fn)(ncclComm_t);

// ---------------------------------------------------------------------------
// Global state: tracked communicators
// ---------------------------------------------------------------------------
static std::vector<ncclComm_t> g_tracked_comms;
static std::mutex              g_comms_mutex;

// ncclCommSuspend/Resume are no longer used (replaced by ipc_hooks.cpp)

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * find_real_nccl_sym - Locate a real NCCL symbol, trying multiple strategies.
 *
 * Strategy order:
 *   1. RTLD_NEXT  — works when libnccl.so is linked directly
 *   2. dlopen("libnccl.so.2", RTLD_NOLOAD) — finds NCCL if already loaded
 *      by PyTorch (even with RTLD_LOCAL)
 *   3. Common PyTorch bundled libnccl paths as last resort
 *
 * This handles the DL ordering issue where PyTorch loads NCCL via dlopen()
 * AFTER our LD_PRELOAD library, making RTLD_NEXT fail.
 */
static void* find_real_nccl_sym(const char* name) {
    void* sym = nullptr;

    // Strategy 1: RTLD_NEXT — skips our LD_PRELOAD library and finds the
    // REAL symbol in libnccl.  This is the ONLY safe first choice because
    // RTLD_DEFAULT would return our own hook function (infinite recursion!).
    sym = dlsym(RTLD_NEXT, name);
    if (sym) return sym;

    // Strategy 2: Find already-loaded libnccl.so.2 via RTLD_NOLOAD.
    // We retry every call because NCCL may be dlopen'd by PyTorch/torch
    // AFTER our initial LD_PRELOAD hooks ran.  dlopen(RTLD_NOLOAD) is cheap.
    static void* nccl_handle = nullptr;
    {
        const char* candidates[] = {
            "libnccl.so.2",
            "libnccl.so",
            nullptr
        };
        for (int i = 0; candidates[i]; i++) {
            void* h = dlopen(candidates[i], RTLD_NOLOAD | RTLD_NOW);
            if (h) {
                if (!nccl_handle) {
                    fprintf(stderr, "[NCCL-HOOKS] Found loaded NCCL via RTLD_NOLOAD: %s\n", candidates[i]);
                }
                nccl_handle = h;
                break;
            }
        }
    }

    if (nccl_handle) {
        sym = dlsym(nccl_handle, name);
        if (sym) return sym;
    }

    // Strategy 3: Explicit path via CR_NCCL_LIB environment variable.
    // This is the PRIMARY mechanism for loading our locally-built NCCL
    // that has ncclCommSuspend/Resume.  Set by launch_multi_gpu.sh.
    static void* nccl_handle_explicit = nullptr;
    static bool explicit_tried = false;
    if (!explicit_tried) {
        explicit_tried = true;
        const char* env_path = getenv("CR_NCCL_LIB");
        if (env_path && env_path[0]) {
            nccl_handle_explicit = dlopen(env_path, RTLD_NOW | RTLD_GLOBAL);
            if (nccl_handle_explicit) {
                fprintf(stderr, "[NCCL-HOOKS] Loaded NCCL from CR_NCCL_LIB=%s\n", env_path);
            } else {
                fprintf(stderr, "[NCCL-HOOKS] WARNING: Failed to load CR_NCCL_LIB=%s: %s\n",
                        env_path, dlerror());
            }
        }
    }
    if (nccl_handle_explicit) {
        sym = dlsym(nccl_handle_explicit, name);
        if (sym) return sym;
    }

    // Strategy 4: Locate NCCL relative to our own .so file.
    // Our vGPU-NVIDIA.so is at <PROJECT>/build/vGPU-NVIDIA.so.
    // Local NCCL is at <PROJECT>/nccl_install/lib/libnccl.so.2.
    static void* nccl_handle_relative = nullptr;
    static bool relative_tried = false;
    if (!relative_tried) {
        relative_tried = true;
        Dl_info dl_info;
        // Use this function's address to find our .so location
        if (dladdr((void*)find_real_nccl_sym, &dl_info) && dl_info.dli_fname) {
            // dl_info.dli_fname = "/path/to/GPU-CR/build/vGPU-NVIDIA.so"
            std::string our_path(dl_info.dli_fname);
            size_t last_slash = our_path.rfind('/');
            if (last_slash != std::string::npos) {
                std::string build_dir = our_path.substr(0, last_slash);
                // Try <build_dir>/../nccl_install/lib/libnccl.so.2
                std::string local_nccl = build_dir + "/../nccl_install/lib/libnccl.so.2";
                // Resolve to canonical path
                char resolved[PATH_MAX];
                if (realpath(local_nccl.c_str(), resolved)) {
                    local_nccl = resolved;
                }
                nccl_handle_relative = dlopen(local_nccl.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (nccl_handle_relative) {
                    fprintf(stderr, "[NCCL-HOOKS] Loaded local NCCL from: %s\n", local_nccl.c_str());
                } else {
                    fprintf(stderr, "[NCCL-HOOKS] Local NCCL not found at: %s\n", local_nccl.c_str());
                }
            }
        }
    }
    if (nccl_handle_relative) {
        sym = dlsym(nccl_handle_relative, name);
        if (sym) return sym;
    }

    // Strategy 5: Generic dlopen from LD_LIBRARY_PATH (last resort).
    // May load system NCCL which likely lacks ncclCommSuspend.
    static void* nccl_handle_fallback = nullptr;
    static bool fallback_tried = false;
    if (!fallback_tried) {
        fallback_tried = true;
        nccl_handle_fallback = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
        if (nccl_handle_fallback) {
            fprintf(stderr, "[NCCL-HOOKS] Loaded NCCL via dlopen fallback (system/LD_LIBRARY_PATH)\n");
        } else {
            fprintf(stderr, "[NCCL-HOOKS] dlopen(libnccl.so.2) failed: %s\n", dlerror());
        }
    }
    if (nccl_handle_fallback) {
        sym = dlsym(nccl_handle_fallback, name);
        if (sym) return sym;
    }

    return nullptr;
}

// resolve_suspend_resume removed — no longer needed

// ---------------------------------------------------------------------------
// Public API: comm tracking
// ---------------------------------------------------------------------------

void nccl_register_comm(ncclComm_t comm) {
    if (!comm) return;
    std::lock_guard<std::mutex> lock(g_comms_mutex);
    // Idempotent — don't add duplicates
    for (auto& c : g_tracked_comms) {
        if (c == comm) return;
    }
    g_tracked_comms.push_back(comm);
    fprintf(stderr, "[NCCL-HOOKS] Registered comm %p (total tracked: %zu)\n",
            (void*)comm, g_tracked_comms.size());
}

void nccl_unregister_comm(ncclComm_t comm) {
    if (!comm) return;
    std::lock_guard<std::mutex> lock(g_comms_mutex);
    auto it = std::find(g_tracked_comms.begin(), g_tracked_comms.end(), comm);
    if (it != g_tracked_comms.end()) {
        g_tracked_comms.erase(it);
        fprintf(stderr, "[NCCL-HOOKS] Unregistered comm %p (remaining: %zu)\n",
                (void*)comm, g_tracked_comms.size());
    }
}

int nccl_get_comm_count() {
    std::lock_guard<std::mutex> lock(g_comms_mutex);
    return (int)g_tracked_comms.size();
}

bool nccl_suspend_available() {
    // ncclCommSuspend/Resume are no longer used.
    // IPC management is handled by ipc_hooks.cpp.
    return false;
}

// nccl_suspend_all_comms / nccl_resume_all_comms removed.
// IPC teardown/rebuild is now handled by ipc_hooks.cpp.
int nccl_suspend_all_comms() {
    fprintf(stderr, "[NCCL-HOOKS] nccl_suspend_all_comms is deprecated — use ipc_teardown_all_imports()\n");
    return 1;  // no-op
}

int nccl_resume_all_comms() {
    fprintf(stderr, "[NCCL-HOOKS] nccl_resume_all_comms is deprecated — use ipc_rebuild_all_imports()\n");
    return 1;  // no-op
}

// ===========================================================================
// LD_PRELOAD hooks — intercept NCCL communicator creation/destruction
//
// When vGPU.so is loaded via LD_PRELOAD these symbols shadow the real ones
// in libnccl.so.  We call through to the real implementation via
// dlsym(RTLD_NEXT, ...) and additionally register/unregister the comm.
// ===========================================================================

extern "C" {

// ------ ncclCommInitRank ------
int ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    static ncclCommInitRank_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (ncclCommInitRank_fn)find_real_nccl_sym("ncclCommInitRank");
        if (!real_fn) {
            fprintf(stderr, "[NCCL-HOOKS] FATAL: real ncclCommInitRank not found\n");
            return 3; // ncclInternalError
        }
    }

    fprintf(stderr, "[NCCL-HOOKS] ncclCommInitRank intercepted: nranks=%d, rank=%d\n",
            nranks, rank);

    int ret = real_fn(comm, nranks, commId, rank);
    if (ret == NCCL_OK && comm && *comm) {
        nccl_register_comm(*comm);
    }
    return ret;
}

// ------ ncclCommInitRankConfig ------
int ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                           int rank, void* config) {
    static ncclCommInitRankConfig_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (ncclCommInitRankConfig_fn)find_real_nccl_sym("ncclCommInitRankConfig");
        if (!real_fn) {
            // Fallback: try ncclCommInitRank instead (older NCCL versions)
            fprintf(stderr, "[NCCL-HOOKS] ncclCommInitRankConfig not found, "
                            "falling back to ncclCommInitRank\n");
            return ncclCommInitRank(comm, nranks, commId, rank);
        }
    }

    fprintf(stderr, "[NCCL-HOOKS] ncclCommInitRankConfig intercepted: nranks=%d, rank=%d\n",
            nranks, rank);

    int ret = real_fn(comm, nranks, commId, rank, config);
    if (ret == NCCL_OK && comm && *comm) {
        nccl_register_comm(*comm);
    }
    return ret;
}

// ------ ncclCommDestroy ------
int ncclCommDestroy(ncclComm_t comm) {
    static ncclCommDestroy_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (ncclCommDestroy_fn)find_real_nccl_sym("ncclCommDestroy");
        if (!real_fn) {
            fprintf(stderr, "[NCCL-HOOKS] FATAL: real ncclCommDestroy not found\n");
            return 3;
        }
    }

    fprintf(stderr, "[NCCL-HOOKS] ncclCommDestroy intercepted: comm=%p\n", (void*)comm);
    nccl_unregister_comm(comm);
    return real_fn(comm);
}

// ------ ncclCommFinalize ------
int ncclCommFinalize(ncclComm_t comm) {
    static ncclCommFinalize_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (ncclCommFinalize_fn)find_real_nccl_sym("ncclCommFinalize");
        if (!real_fn) {
            fprintf(stderr, "[NCCL-HOOKS] WARNING: real ncclCommFinalize not found, skipping\n");
            return NCCL_OK;
        }
    }

    fprintf(stderr, "[NCCL-HOOKS] ncclCommFinalize intercepted: comm=%p\n", (void*)comm);
    // Note: don't unregister here because finalize doesn't destroy the comm.
    // The comm is fully destroyed only in ncclCommDestroy.
    return real_fn(comm);
}

// ------ Manual registration API (for applications using ctypes, etc.) ------

void cr_register_nccl_comm(void* comm) {
    nccl_register_comm((ncclComm_t)comm);
}

void cr_unregister_nccl_comm(void* comm) {
    nccl_unregister_comm((ncclComm_t)comm);
}

} // extern "C"
