#include "amd.h"
#include "../../common.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <hip/hip_runtime_api.h>

std::map<void*, size_t> allocated_memory;
static std::map<void*, hipMemGenericAllocationHandle_t> global_handle_map;
std::map<void*, int> allocated_memory_type;

// Function pointer types for hipMalloc/hipFree interception
typedef hipError_t (*hipMalloc_func_t)(void**, size_t);
typedef hipError_t (*hipMemcpy_func_t)(void*, const void*, size_t, hipMemcpyKind);
typedef hipError_t (*hipMemcpyAsync_func_t)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
typedef hipError_t (*hipDeviceSynchronize_func_t)(void);
typedef hipError_t (*hipFree_func_t)(void*);

static hipMalloc_func_t real_hipMalloc = nullptr;
static hipMemcpy_func_t real_hipMemcpy = nullptr;
static hipMemcpyAsync_func_t real_hipMemcpyAsync = nullptr;
static hipDeviceSynchronize_func_t real_hipDeviceSynchronize = nullptr;
static hipFree_func_t real_hipFree = nullptr;

#define HIP_CHECK(call) { \
        hipError_t result = (call); \
        if (result != hipSuccess) { \
            fprintf(stderr, "%s failed: %s (at %s:%d)\n", #call, hipGetErrorString(result), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

amd::amd() : device_(0), hip_initialized_(false) {
    fprintf(stderr, "[AMD] Initializing AMD GPU backend\n");
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] Warning: hipSetDevice failed\n");
    }
}

amd::~amd() {
    fprintf(stderr, "[AMD] Destroying AMD GPU backend\n");
}

void amd::ensureHipInitialized() {
    if (hip_initialized_) return;
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] Warning: hipInit not available in HIP runtime API\n");
    }
    hip_initialized_ = true;
}

int amd::allocate(void** ptr, size_t size) {
    throw std::runtime_error("amd::allocate: This subclass does nothing. Use hooked hipMalloc instead.");
}

int amd::deallocate(void* ptr) {
    throw std::runtime_error("amd::deallocate: This subclass does nothing. Use hooked hipFree instead.");
}

std::map<void*, size_t>& amd::getMemoryMap() {
    return memory_map_;
}

int amd::releasePhysicalMemory(void* ptr) {
    fprintf(stderr, "[AMD][PID:%d] releasePhysicalMemory: ptr=%p\n", getpid(), ptr);
    fflush(stderr);

    auto it = allocated_memory.find(ptr);
    if (it == allocated_memory.end()) {
        fprintf(stderr, "[AMD][PID:%d] releasePhysicalMemory: ptr not tracked, skipping\n", getpid());
        return 0;
    }

    size_t size = it->second;
    auto h_it = global_handle_map.find(ptr);
    if (h_it == global_handle_map.end()) {
        fprintf(stderr, "[AMD][PID:%d] releasePhysicalMemory: handle not found for %p\n", getpid(), ptr);
        return -1;
    }

    hipMemGenericAllocationHandle_t handle = h_it->second;

    hipError_t err = hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(ptr), size);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD][PID:%d] hipMemUnmap failed: %s\n", getpid(), hipGetErrorString(err));
        return -1;
    }

    err = hipMemRelease(handle);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD][PID:%d] hipMemRelease failed: %s\n", getpid(), hipGetErrorString(err));
        return -1;
    }

    fprintf(stderr, "[AMD][PID:%d] Physical memory released, virtual address kept: %p (size=%zu)\n", getpid(), ptr, size);
    fflush(stderr);
    return 0;
}

int amd::remapPhysicalMemory(void* ptr, size_t size) {
    fprintf(stderr, "[AMD][PID:%d] remapPhysicalMemory: ptr=%p, size=%zu\n", getpid(), ptr, size);
    fflush(stderr);

    hipMemGenericAllocationHandle_t handle;

    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device_;

    hipError_t err = hipMemCreate(&handle, size, &prop, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD][PID:%d] hipMemCreate failed: %s\n", getpid(), hipGetErrorString(err));
        return -1;
    }

    err = hipMemMap(reinterpret_cast<hipDeviceptr_t>(ptr), size, 0, handle, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD][PID:%d] hipMemMap failed: %s\n", getpid(), hipGetErrorString(err));
        (void)hipMemRelease(handle);
        return -1;
    }

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device_;
    access.flags = hipMemAccessFlagsProtReadWrite;
    err = hipMemSetAccess(reinterpret_cast<hipDeviceptr_t>(ptr), size, &access, 1);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD][PID:%d] hipMemSetAccess failed: %s\n", getpid(), hipGetErrorString(err));
        (void)hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(ptr), size);
        (void)hipMemRelease(handle);
        return -1;
    }

    global_handle_map[ptr] = handle;
    allocated_memory[ptr] = size;
    allocated_memory_type[ptr] = 1;

    fprintf(stderr, "[AMD][PID:%d] Remapped physical memory to virtual addr %p (size=%zu)\n", getpid(), ptr, size);
    fflush(stderr);
    return 0;
}

int amd::createStream(GPUStream* stream) {
    hipStream_t hip_stream;
    hipError_t err = hipStreamCreate(&hip_stream);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipStreamCreate failed\n");
        return -1;
    }
    *stream = (GPUStream)hip_stream;
    return 0;
}

int amd::destroyStream(GPUStream stream) {
    hipError_t err = hipStreamDestroy((hipStream_t)stream);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipStreamDestroy failed\n");
        return -1;
    }
    return 0;
}

int amd::createEvent(GPUEvent* event) {
    hipEvent_t hip_event;
    hipError_t err = hipEventCreate(&hip_event);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipEventCreate failed\n");
        return -1;
    }
    *event = (GPUEvent)hip_event;
    return 0;
}

int amd::destroyEvent(GPUEvent event) {
    hipError_t err = hipEventDestroy((hipEvent_t)event);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipEventDestroy failed\n");
        return -1;
    }
    return 0;
}

int amd::recordEvent(GPUEvent event, GPUStream stream) {
    hipError_t err = hipEventRecord((hipEvent_t)event, (hipStream_t)stream);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipEventRecord failed\n");
        return -1;
    }
    return 0;
}

int amd::synchronizeEvent(GPUEvent event) {
    hipError_t err = hipEventSynchronize((hipEvent_t)event);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipEventSynchronize failed\n");
        return -1;
    }
    return 0;
}

int amd::memcpyAsync(void* dst, const void* src, size_t size, 
                     GPUMemcpyKind kind, GPUStream stream) {
    hipMemcpyKind hip_kind;
    switch (kind) {
        case GPUMemcpyKind::HostToDevice:
            hip_kind = hipMemcpyHostToDevice;
            break;
        case GPUMemcpyKind::DeviceToHost:
            hip_kind = hipMemcpyDeviceToHost;
            break;
        case GPUMemcpyKind::DeviceToDevice:
            hip_kind = hipMemcpyDeviceToDevice;
            break;
        default:
            hip_kind = hipMemcpyDefault;
            break;
    }
    
    hipError_t err = hipMemcpyAsync(dst, src, size, hip_kind, (hipStream_t)stream);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipMemcpyAsync failed\n");
        return -1;
    }
    return 0;
}

int amd::synchronizeStream(GPUStream stream) {
    hipError_t err = hipStreamSynchronize((hipStream_t)stream);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipStreamSynchronize failed\n");
        return -1;
    }
    return 0;
}

int amd::syncAllKernels() {
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipDeviceSynchronize failed\n");
        return -1;
    }
    return 0;
}

int amd::registerHostMemory(void* ptr, size_t size) {
    hipError_t err = hipHostRegister(ptr, size, hipHostRegisterDefault);
    if (err != hipSuccess) {
        fprintf(stderr, "[AMD] hipHostRegister failed\n");
        return -1;
    }
    return 0;
}

int amd::externalCheckpoint(int pid) {
    const char* ckpt_dir = getenv("AMD_CKPT_DIR");
    if (!ckpt_dir) {
        fprintf(stderr, "[AMD] ERROR: AMD_CKPT_DIR environment variable not set!\n");
        fprintf(stderr, "[AMD] Please set: export AMD_CKPT_DIR=/path/to/checkpoint/dir\n");
        exit(EXIT_FAILURE);
    }
    
    fprintf(stderr, "[AMD] Using CRIU for checkpoint\n");
    fprintf(stderr, "[AMD] Checkpoint directory: %s\n", ckpt_dir);
    
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), 
        "sudo env LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu "
        "criu dump "
        "--link-remap "
        "--tcp-established "
        "-t %d "
        "-D %s "
        "-j -v4 "
        "-o %s/dump.log "
        "-L /usr/local/lib/criu",
        pid, ckpt_dir, ckpt_dir);
    
    fprintf(stderr, "[AMD] Executing: %s\n", cmd);
    return system(cmd);
}

int amd::externalRestore(int pid) {
    const char* ckpt_dir = getenv("AMD_CKPT_DIR");
    if (!ckpt_dir) {
        fprintf(stderr, "[AMD] ERROR: AMD_CKPT_DIR environment variable not set!\n");
        fprintf(stderr, "[AMD] Please set: export AMD_CKPT_DIR=/path/to/checkpoint/dir\n");
        exit(EXIT_FAILURE);
    }
    
    fprintf(stderr, "[AMD] Using CRIU for restore\n");
    fprintf(stderr, "[AMD] Restore directory: %s\n", ckpt_dir);
    
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "sudo env LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu "
        "criu restore "
        "-D %s "
        "-j -v4 "
        "-o %s/restore.log "
        "-L /usr/local/lib/criu",
        ckpt_dir, ckpt_dir);
    
    fprintf(stderr, "[AMD] Executing: %s\n", cmd);
    return system(cmd);
}

// hook functions implementation
extern "C" hipError_t hipMalloc(void **devPtr, size_t size) {
    fprintf(stderr, "[HOOK-HIP][PID:%d] hipMalloc called! size=%zu\n", getpid(), size);
    fflush(stderr);

    if (size == 0) {
        *devPtr = nullptr;
        return hipSuccess;
    }

    int device = 0;
    (void)hipGetDevice(&device);

    size_t aligned_size = ROUND_UP_2MB(size);
    hipDeviceptr_t virtualAddr = 0;

    // reserve virtual address space
    hipError_t err = hipMemAddressReserve(&virtualAddr, aligned_size, 0, 0, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[HOOK-HIP][PID:%d] hipMemAddressReserve failed: %s\n", getpid(), hipGetErrorString(err));
        return err;
    }

    // allocate physical memory
    hipMemGenericAllocationHandle_t memHandle;
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;

    err = hipMemCreate(&memHandle, aligned_size, &prop, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[HOOK-HIP][PID:%d] hipMemCreate failed: %s\n", getpid(), hipGetErrorString(err));
        (void)hipMemAddressFree(virtualAddr, aligned_size);
        return err;
    }

    // Map physical memory to virtual address
    err = hipMemMap(virtualAddr, aligned_size, 0, memHandle, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[HOOK-HIP][PID:%d] hipMemMap failed: %s\n", getpid(), hipGetErrorString(err));
        (void)hipMemRelease(memHandle);
        (void)hipMemAddressFree(virtualAddr, aligned_size);
        return err;
    }

    // Set access permissions
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    err = hipMemSetAccess(virtualAddr, aligned_size, &access, 1);
    if (err != hipSuccess) {
        fprintf(stderr, "[HOOK-HIP][PID:%d] hipMemSetAccess failed: %s\n", getpid(), hipGetErrorString(err));
        (void)hipMemUnmap(virtualAddr, aligned_size);
        (void)hipMemRelease(memHandle);
        (void)hipMemAddressFree(virtualAddr, aligned_size);
        return err;
    }

    *devPtr = reinterpret_cast<void*>(virtualAddr);

    // Record allocation information
    allocated_memory[*devPtr] = aligned_size;
    allocated_memory_type[*devPtr] = 1;
    global_handle_map[*devPtr] = memHandle;

    fprintf(stderr, "[HOOK-HIP][PID:%d] VMM Allocated %zu bytes at %p\n", getpid(), aligned_size, *devPtr);
    fprintf(stderr, "[HOOK-HIP] allocated_memory.size() = %zu\n", allocated_memory.size());
    fflush(stderr);
    return hipSuccess;
}

extern "C" hipError_t hipFree(void *devPtr) {
    fprintf(stderr, "[HOOK-HIP][PID:%d] hipFree called! ptr=%p\n", getpid(), devPtr);
    fflush(stderr);

    if (!devPtr) return hipSuccess;

    auto it = allocated_memory.find(devPtr);
    if (it != allocated_memory.end()) {
        size_t size = it->second;
        auto h_it = global_handle_map.find(devPtr);
        if (h_it != global_handle_map.end()) {
            hipMemGenericAllocationHandle_t handle = h_it->second;
            (void)hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(devPtr), size);
            (void)hipMemRelease(handle);
            (void)hipMemAddressFree(reinterpret_cast<hipDeviceptr_t>(devPtr), size);
            global_handle_map.erase(h_it);
            fprintf(stderr, "[HOOK-HIP][PID:%d] VMM freed %p (size=%zu)\n", getpid(), devPtr, size);
        } else {
            if (!real_hipFree) {
                real_hipFree = (hipFree_func_t)dlsym(RTLD_NEXT, "hipFree");
            }
            if (real_hipFree) (void)real_hipFree(devPtr);
            fprintf(stderr, "[HOOK-HIP][PID:%d] Regular hipFree for %p\n", getpid(), devPtr);
        }

        allocated_memory.erase(it);
        allocated_memory_type.erase(devPtr);
        fprintf(stderr, "[HOOK-HIP] allocated_memory.size() = %zu\n", allocated_memory.size());
        fflush(stderr);
        return hipSuccess;
    }

    if (!real_hipFree) {
        real_hipFree = (hipFree_func_t)dlsym(RTLD_NEXT, "hipFree");
    }
    if (real_hipFree) return real_hipFree(devPtr);
    return hipSuccess;
}
