#ifndef GPU_H
#define GPU_H

#include <map>
#include <string>
#include <cstddef>
#include <cstdint>

typedef void* GPUStream;   // CUDA: cudaStream_t, ROCm: hipStream_t
typedef void* GPUEvent;    // CUDA: cudaEvent_t,  ROCm: hipEvent_t
typedef void* GPUDevice;   // CUDA: CUdevice,     ROCm: hipDevice_t

enum class GPUVendor {
    NVIDIA,
    AMD,
    UNKNOWN
};

enum class GPUMemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

// ==================== GPU ====================
class GPU {
public:
    GPU() = default;
    virtual ~GPU() = default;

    // ========== memory management ==========

    virtual int allocate(void** ptr, size_t size) = 0;

    virtual int deallocate(void* ptr) = 0;
    
    virtual std::map<void*, size_t>& getMemoryMap() = 0;

    virtual int memcpyAsync(void* dst, const void* src, size_t size, 
                           GPUMemcpyKind kind, GPUStream stream) = 0;

    // ========== synchronization ==========
    
    virtual int createStream(GPUStream* stream) = 0;
    
    virtual int destroyStream(GPUStream stream) = 0;

    virtual int createEvent(GPUEvent* event) = 0;

    virtual int destroyEvent(GPUEvent event) = 0;
    
    virtual int recordEvent(GPUEvent event, GPUStream stream) = 0;

    virtual int synchronizeEvent(GPUEvent event) = 0;
    
    virtual int synchronizeStream(GPUStream stream) = 0;
    
    virtual int syncAllKernels() = 0;
    
    virtual int registerHostMemory(void* ptr, size_t size) = 0;

    // ========== checkpoint/restore memory management ==========
    
    // Release physical GPU memory but keep virtual address reserved
    virtual int releasePhysicalMemory(void* ptr) = 0;
    
    // Remap physical GPU memory to previously released virtual address
    virtual int remapPhysicalMemory(void* ptr, size_t size) = 0;

    // ========== external tool interfaces ==========
    
    virtual int externalCheckpoint(int pid) = 0;
    
    virtual int externalRestore(int pid) = 0;
    
    virtual int pushContext() { return 0; }
    virtual int popContext() { return 0; }
    
    virtual GPUVendor getVendor() const = 0;
    
    virtual std::string getVendorName() const = 0;
};

// ==================== factory function ====================
/**
 * @brief Automatically detect GPU vendor and create corresponding GPU object
 * @return GPU* instance (caller is responsible for deleting the object)
 */
GPU* createGPU();

#endif // GPU_H