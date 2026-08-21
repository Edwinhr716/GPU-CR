#include "nv.h"
#include "../../common.h"
#include <dlfcn.h>
#include <pthread.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <mutex>
#include <set>

// define global maps for memory tracking
std::map<void*, size_t> allocated_memory;
std::map<void*, int> allocated_memory_type;  // 0=cudaMalloc, 1=VMM

// Guards the tracking maps against concurrent hook calls; declared extern
// in common.h.
std::mutex gpu_mem_mutex;

// Global handle map for all VMM allocations (both from hook and nv::allocate)
static std::map<void*, CUmemGenericAllocationHandle> global_handle_map;
// External linkage is intentional: the IPC hooks consume this via extern.
CUcontext g_pytorch_context = nullptr;

namespace {

// Blocks the CR control signals for the guard's lifetime. The CR signal
// handlers take gpu_mem_mutex, so a process-directed CR signal delivered to
// a thread holding the mutex would self-deadlock; masking the holding thread
// routes delivery to a non-holding thread instead. Construct immediately
// before the lock so reverse destruction unlocks before unmasking.
class ScopedBlockCrSignals {
public:
    ScopedBlockCrSignals() {
        sigset_t block;
        sigemptyset(&block);
        sigaddset(&block, CR_INIT_SIGNAL);
        sigaddset(&block, CR_CKPT_SIGNAL);
        sigaddset(&block, CR_RESTORE_SIGNAL);
        sigaddset(&block, CR_IPC_TEARDOWN_SIGNAL);
        sigaddset(&block, CR_IPC_REBUILD_SIGNAL);
        sigaddset(&block, CR_IPC_VALIDATE_SIGNAL);
        pthread_sigmask(SIG_BLOCK, &block, &old_mask_);
    }
    ~ScopedBlockCrSignals() {
        // SIG_SETMASK with the saved set: SIG_UNBLOCK would wrongly unmask
        // signals the caller had already blocked.
        pthread_sigmask(SIG_SETMASK, &old_mask_, nullptr);
    }
    ScopedBlockCrSignals(const ScopedBlockCrSignals&) = delete;
    ScopedBlockCrSignals& operator=(const ScopedBlockCrSignals&) = delete;

private:
    sigset_t old_mask_;
};

}  // namespace

// P2P peer access hooks and helpers live in src/ipc_hooks.cpp (canonical).

typedef cudaError_t (*cudaMalloc_func_t)(void**, size_t);
typedef cudaError_t (*cudaFree_func_t)(void*);

static cudaMalloc_func_t real_cudaMalloc = nullptr;
static cudaFree_func_t real_cudaFree = nullptr;

#define CU_CHECK(call) { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errorStr; \
            cuGetErrorString(result, &errorStr); \
            fprintf(stderr, "%s failed: %s (at %s:%d)\n", #call, errorStr, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

#define CUDA_CHECK_RET(call) { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
           exit(EXIT_FAILURE); \
        } \
    }

nv::nv() : device_(-1), context_(nullptr), cuda_initialized_(false), pushed_context_(nullptr) {
    fprintf(stderr, "[NVIDIA] Initializing NVIDIA GPU backend\n");
}

nv::~nv() {
    fprintf(stderr, "[NVIDIA] Destroying NVIDIA GPU backend\n");
}

void nv::ensureCudaInitialized() {
    if (cuda_initialized_) return;
    
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "[NVIDIA] cuInit failed: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }
    
    // Get or create context
    CU_CHECK(cuCtxGetCurrent(&context_));
    if (context_ == nullptr) {
        CU_CHECK(cuDeviceGet(&device_, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
        fprintf(stderr, "[NVIDIA] Created CUDA context on device %d\n", device_);
    } else {
        CU_CHECK(cuCtxGetDevice(&device_));
        fprintf(stderr, "[NVIDIA] Using existing CUDA context on device %d\n", device_);
    }
    
    cuda_initialized_ = true;
}

// ========== memory management implementation ==========

int nv::allocate(void** ptr, size_t size) {
    throw std::runtime_error("nv::allocate: This subclass does nothing. Use hooked cudaMalloc instead.");
}

int nv::deallocate(void* ptr) {
    throw std::runtime_error("nv::deallocate: This subclass does nothing. Use hooked cudaFree instead.");
}

std::map<void*, size_t>& nv::getMemoryMap() {
    return allocated_memory;
}

// ========== synchronization implementation ==========

int nv::createStream(GPUStream* stream) {
    CUDA_CHECK_RET(cudaStreamCreate((cudaStream_t*)stream));
    return 0;
}

int nv::destroyStream(GPUStream stream) {
    CUDA_CHECK_RET(cudaStreamDestroy((cudaStream_t)stream));
    return 0;
}

int nv::createEvent(GPUEvent* event) {
    CUDA_CHECK_RET(cudaEventCreate((cudaEvent_t*)event));
    return 0;
}

int nv::destroyEvent(GPUEvent event) {
    CUDA_CHECK_RET(cudaEventDestroy((cudaEvent_t)event));
    return 0;
}

int nv::recordEvent(GPUEvent event, GPUStream stream) {
    CUDA_CHECK_RET(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream));
    return 0;
}

int nv::synchronizeEvent(GPUEvent event) {
    CUDA_CHECK_RET(cudaEventSynchronize((cudaEvent_t)event));
    return 0;
}

int nv::memcpyAsync(void* dst, const void* src, size_t size, GPUMemcpyKind kind, GPUStream stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case GPUMemcpyKind::HostToDevice:   cuda_kind = cudaMemcpyHostToDevice; break;
        case GPUMemcpyKind::DeviceToHost:   cuda_kind = cudaMemcpyDeviceToHost; break;
        case GPUMemcpyKind::DeviceToDevice: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: return -1;
    }
    CUDA_CHECK_RET(cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream));
    return 0;
}

int nv::synchronizeStream(GPUStream stream) {
    CUDA_CHECK_RET(cudaStreamSynchronize((cudaStream_t)stream));
    return 0;
}

int nv::syncAllKernels() {
    CUDA_CHECK_RET(cudaDeviceSynchronize());
    return 0;
}

int nv::registerHostMemory(void* ptr, size_t size) {
    ensureCudaInitialized();  // Ensure CUDA is initialized before calling cudaHostRegister
    
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        fprintf(stderr, "[NVIDIA] cudaHostRegister failed: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "[NVIDIA] This is expected for hugepage-backed memory, continuing without pinned memory\n");
        cudaGetLastError(); // Clear the error so it doesn't pollute the runtime state
        return -1;  // Return error but don't exit - non-pinned memory will still work
    }
    return 0;
}

// ========== Checkpoint/Restore memory management implementation ==========

int nv::releasePhysicalMemory(void* ptr) {
    auto it = allocated_memory.find(ptr);
    if (it == allocated_memory.end()) {
        fprintf(stderr, "[NVIDIA] Warning: Pointer %p not found in allocated_memory\n", ptr);
        return -1;
    }

    size_t size = it->second;
    size_t aligned_size = ROUND_UP_2MB(size); 
    CUdeviceptr cuptr = (CUdeviceptr)ptr;

    fprintf(stderr, "[NVIDIA] Releasing physical memory at %p (size=%zu, aligned=%zu)\n", 
            ptr, size, aligned_size);

    // Only release physical memory, keep virtual address space
    CUresult res = cuMemUnmap(cuptr, aligned_size);
    if (res != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "[NVIDIA] cuMemUnmap failed: %s\n", errorStr);
        return -1;
    }

    // Release physical memory handle
    auto handle_it = global_handle_map.find(ptr);
    if (handle_it != global_handle_map.end()) {
        res = cuMemRelease(handle_it->second);
        if (res != CUDA_SUCCESS) {
            const char* errorStr;
            cuGetErrorString(res, &errorStr);
            fprintf(stderr, "[NVIDIA] cuMemRelease failed: %s\n", errorStr);
            return -1;
        }
        global_handle_map.erase(handle_it);
    }

    // Do not call cuMemAddressFree, keep virtual address space
    fprintf(stderr, "[NVIDIA] Physical memory released, virtual address %p preserved\n", ptr);
    return 0;
}


int nv::remapPhysicalMemory(void* ptr, size_t size) {
    // Check if this pointer is in our tracking
    auto it = allocated_memory.find(ptr);
    if (it == allocated_memory.end()) {
        fprintf(stderr, "[NVIDIA] Warning: Trying to remap unknown pointer %p\n", ptr);
        return -1;
    }
    
    // If it is already mapped, release the old physical memory first
    auto handle_it = global_handle_map.find(ptr);
    if (handle_it != global_handle_map.end()) {
        fprintf(stderr, "[NVIDIA] Pointer %p is already mapped, releasing old physical memory first...\n", ptr);
        size_t old_size = it->second;
        size_t old_aligned_size = ROUND_UP_2MB(old_size);
        CUdeviceptr cuptr = reinterpret_cast<CUdeviceptr>(ptr);
        CUresult res = cuMemUnmap(cuptr, old_aligned_size);
        if (res != CUDA_SUCCESS) {
            const char* error_str;
            cuGetErrorString(res, &error_str);
            fprintf(stderr, "[NVIDIA] cuMemUnmap failed during remap: %s\n", error_str);
            return -1;
        }
        res = cuMemRelease(handle_it->second);
        if (res != CUDA_SUCCESS) {
            const char* error_str;
            cuGetErrorString(res, &error_str);
            fprintf(stderr, "[NVIDIA] cuMemRelease failed during remap: %s\n", error_str);
            // The range is already unmapped: drop the stale handle (leaking
            // it) so the tracking stays consistent and a retry can proceed.
            global_handle_map.erase(handle_it);
            return -1;
        }
        global_handle_map.erase(handle_it);
    }

    // The virtual reservation was sized for the tracked allocation, so a
    // differing request cannot be honored safely.
    if (size != it->second) {
        fprintf(stderr, "[NVIDIA] Remap size %zu differs from tracked size %zu for %p, using tracked size\n",
                size, it->second, ptr);
        size = it->second;
    }

    // All allocations now use VMM, so we can remap for all
    size_t aligned_size = ROUND_UP_2MB(size);
    CUdeviceptr cuptr = (CUdeviceptr)ptr;
    
    fprintf(stderr, "[NVIDIA] Remapping physical memory for %p (size=%zu)\n", ptr, aligned_size);
    
    // Allocate new physical memory
    CUmemGenericAllocationHandle memHandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    CU_CHECK(cuMemCreate(&memHandle, aligned_size, &prop, 0));
    
    // Map existing virtual address to new physical memory
    CU_CHECK(cuMemMap(cuptr, aligned_size, 0, memHandle, 0));
    
    // Set access permissions
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device_;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(cuptr, aligned_size, &accessDesc, 1));
    
    // Store new handle in global map
    global_handle_map[ptr] = memHandle;
    
    fprintf(stderr, "[NVIDIA] Physical memory remapped at %p\n", ptr);
    return 0;
}

// ========== external tool interfaces implementation ==========

int nv::externalCheckpoint(int pid) {
    // Prefer using cuda-checkpoint command
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cuda-checkpoint --toggle --pid %d", pid);
    
    fprintf(stderr, "[NVIDIA] Executing: %s\n", cmd);
    int ret = system(cmd);
    
    if (ret != 0) {
        fprintf(stderr, "[NVIDIA] Warning: cuda-checkpoint failed with code %d\n", ret);
        fprintf(stderr, "[NVIDIA] Make sure cuda-checkpoint is in your PATH\n");
        fprintf(stderr, "[NVIDIA] Or set: export PATH=\"/path/to/cuda-checkpoint/bin/x86_64_Linux:$PATH\"\n");
        return -1;
    }
    
    return 0;
}

int nv::externalRestore(int pid) {
    // cuda-checkpoint's restore also uses --toggle
    return externalCheckpoint(pid);
}

int nv::pushContext() {
    ensureCudaInitialized();
    // Snapshot under the allocation lock: the hooks write g_pytorch_context
    // while application threads may still be allocating. Release before the
    // driver call. The signal guard also covers handler context: a handler
    // only auto-blocks its own signum, so a different CR signal nesting here
    // would otherwise deadlock on the same mutex.
    CUcontext captured = nullptr;
    {
        ScopedBlockCrSignals signal_block;
        std::lock_guard<std::mutex> lock(gpu_mem_mutex);
        captured = g_pytorch_context;
    }
    CUcontext target_context = context_;
    if (captured != nullptr) {
        target_context = captured;
        fprintf(stderr, "[NVIDIA] Pushing captured PyTorch context: %p\n", target_context);
    } else {
        fprintf(stderr, "[NVIDIA] Pushing default context: %p\n", target_context);
    }
    CUresult res = cuCtxPushCurrent(target_context);
    if (res != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(res, &error_str);
        fprintf(stderr, "[NVIDIA] cuCtxPushCurrent failed: %s\n", error_str);
        return -1;
    }
    pushed_context_ = target_context;
    return 0;
}

int nv::popContext() {
    // Only pop what pushContext pushed: an unpaired pop (e.g. after a failed
    // push) would detach the application's context from the thread.
    if (pushed_context_ == nullptr) {
        fprintf(stderr, "[NVIDIA] popContext called without an outstanding pushContext, skipping pop\n");
        return -1;
    }
    CUcontext popped;
    CUresult res = cuCtxPopCurrent(&popped);
    if (res != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(res, &error_str);
        fprintf(stderr, "[NVIDIA] cuCtxPopCurrent failed: %s\n", error_str);
        return -1;  // The push is still outstanding; keep pushed_context_ set.
    }
    if (popped != pushed_context_) {
        fprintf(stderr, "[NVIDIA] Popped context %p does not match pushed context %p, restoring it\n",
                popped, pushed_context_);
        cuCtxPushCurrent(popped);
        return -1;
    }
    pushed_context_ = nullptr;
    fprintf(stderr, "[NVIDIA] Popped context: %p\n", popped);
    return 0;
}

// ========== hook functions implementation ==========

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
    ScopedBlockCrSignals signal_block;
    std::lock_guard<std::mutex> lock(gpu_mem_mutex);
    CUcontext curr_ctx = nullptr;
    (void)cuCtxGetCurrent(&curr_ctx);  // Best-effort: curr_ctx stays null on failure.
    fprintf(stderr, "[HOOK] cudaMalloc called! size=%zu, current ctx=%p\n", size, curr_ctx);
    fflush(stderr);

    if (g_pytorch_context == nullptr && curr_ctx != nullptr) {
        g_pytorch_context = curr_ctx;
        fprintf(stderr, "[HOOK] Captured PyTorch CUDA context (fallback): %p\n", g_pytorch_context);
        fflush(stderr);
    }

    if (size == 0) {
        fprintf(stderr, "[HOOK] cudaMalloc(0) -> returning nullptr and cudaSuccess\n");
        *devPtr = nullptr;
        return cudaSuccess;
    }

    nv* gpu_instance = nullptr;

    static nv* hook_gpu = nullptr;
    if (!hook_gpu) {
        hook_gpu = new nv();
    }
    
    size_t aligned_size = ROUND_UP_2MB(size);
    void* ptr = nullptr;
    
    CUdeviceptr virtualAddr = 0;
    
    static bool cuda_inited = false;
    if (!cuda_inited) {
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] cuInit failed\n");
            return cudaErrorInitializationError;
        }
        cuda_inited = true;
    }
    
    // Get device and context
    static CUdevice device = -1;
    static CUcontext context = nullptr;
    if (device == -1) {
        CUresult res = cuCtxGetCurrent(&context);
        if (res == CUDA_SUCCESS && context != nullptr) {
            res = cuCtxGetDevice(&device);
            if (res != CUDA_SUCCESS) {
                fprintf(stderr, "[HOOK] cuCtxGetDevice failed\n");
                return cudaErrorInitializationError;
            }
            fprintf(stderr, "[HOOK] Using existing context, device=%d\n", device);
            g_pytorch_context = context;
        } else {
            res = cuDeviceGet(&device, 0);
            if (res != CUDA_SUCCESS) {
                const char* errorStr;
                cuGetErrorString(res, &errorStr);
                fprintf(stderr, "[HOOK] cuDeviceGet failed: %s\n", errorStr);
                return cudaErrorInitializationError;
            }
            res = cuDevicePrimaryCtxRetain(&context, device);
            if (res != CUDA_SUCCESS) {
                const char* errorStr;
                cuGetErrorString(res, &errorStr);
                fprintf(stderr, "[HOOK] cuCtxCreate failed: %s\n", errorStr);
                return cudaErrorInitializationError;
            }
            fprintf(stderr, "[HOOK] Created new context, device=%d\n", device);
            g_pytorch_context = context;
        }
    }
    
    // VMM allocation
    CUresult res = cuMemAddressReserve(&virtualAddr, aligned_size, 0, 0, 0);
    if (res != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "[HOOK] cuMemAddressReserve failed: %s (code=%d, size=%zu)\n", errorStr, res, aligned_size);
        return cudaErrorMemoryAllocation;
    }
    
    // Allocate physical memory
    CUmemGenericAllocationHandle memHandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    res = cuMemCreate(&memHandle, aligned_size, &prop, 0);
    if (res != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "[HOOK] cuMemCreate failed: %s (code=%d, size=%zu, device=%d)\n", errorStr, res, aligned_size, device);
        cuMemAddressFree(virtualAddr, aligned_size);
        return cudaErrorMemoryAllocation;
    }
    
    // Map virtual to physical
    res = cuMemMap(virtualAddr, aligned_size, 0, memHandle, 0);
    if (res != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "[HOOK] cuMemMap failed: %s (code=%d)\n", errorStr, res);
        cuMemRelease(memHandle);
        cuMemAddressFree(virtualAddr, aligned_size);
        return cudaErrorMemoryAllocation;
    }
    
    // Set access permissions
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    res = cuMemSetAccess(virtualAddr, aligned_size, &accessDesc, 1);
    if (res != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "[HOOK] cuMemSetAccess failed: %s (code=%d)\n", errorStr, res);
        cuMemUnmap(virtualAddr, aligned_size);
        cuMemRelease(memHandle);
        cuMemAddressFree(virtualAddr, aligned_size);
        return cudaErrorMemoryAllocation;
    }
    
    ptr = (void*)virtualAddr;
    *devPtr = ptr;
    
    // Store in global maps
    global_handle_map[ptr] = memHandle;
    allocated_memory[ptr] = size;
    allocated_memory_type[ptr] = 1;  // VMM allocation
    
    fprintf(stderr, "[HOOK] cudaMalloc(%zu) => %p (VMM, aligned to %zu)\n", size, ptr, aligned_size);
    fflush(stderr);

    return cudaSuccess;
}

extern "C" cudaError_t cudaFree(void* ptr) {
    ScopedBlockCrSignals signal_block;
    std::lock_guard<std::mutex> lock(gpu_mem_mutex);
    fprintf(stderr, "[HOOK] cudaFree(%p)\n", ptr);
    fflush(stderr);
    
    auto it = allocated_memory.find(ptr);
    if (it == allocated_memory.end()) {
        fprintf(stderr, "[HOOK] cudaFree: pointer not found in allocated_memory\n");
        // Try calling real cudaFree
        if (!real_cudaFree) {
            real_cudaFree = (cudaFree_func_t)dlsym(RTLD_NEXT, "cudaFree");
        }
        if (real_cudaFree) {
            return real_cudaFree(ptr);
        }
        return cudaSuccess;
    }
    
    size_t size = it->second;
    size_t aligned_size = ROUND_UP_2MB(size);
    
    // Check if it's VMM allocation
    auto type_it = allocated_memory_type.find(ptr);
    if (type_it != allocated_memory_type.end() && type_it->second == 1) {
        // VMM allocation - full cleanup
        auto handle_it = global_handle_map.find(ptr);
        if (handle_it != global_handle_map.end()) {
            CUmemGenericAllocationHandle memHandle = handle_it->second;
            
            // Unmap, release physical memory, and free virtual address
            cuMemUnmap((CUdeviceptr)ptr, aligned_size);
            cuMemRelease(memHandle);
            cuMemAddressFree((CUdeviceptr)ptr, aligned_size);
            
            global_handle_map.erase(handle_it);
            fprintf(stderr, "[HOOK] cudaFree: VMM memory freed\n");
        }
        allocated_memory_type.erase(type_it);
    }
    
    allocated_memory.erase(it);
    fprintf(stderr, "[HOOK] cudaFree completed\n");
    fflush(stderr);

    return cudaSuccess;
}

// P2P peer access hooks (cudaDeviceEnablePeerAccess / cudaDeviceDisablePeerAccess)
// and the disable/reenable helpers used by vGPU.cpp are defined in
// src/ipc_hooks.cpp (canonical), to keep IPC + P2P state in one place.