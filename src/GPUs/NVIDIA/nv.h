#ifndef NV_H
#define NV_H

#include "../GPU.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>

class nv : public GPU {
private:
    CUdevice device_;
    CUcontext context_;
    bool cuda_initialized_;
    
    void ensureCudaInitialized();
    
public:
    nv();
    ~nv() override;
    
    // ========== memory management ==========
    int allocate(void** ptr, size_t size) override;
    int deallocate(void* ptr) override;
    std::map<void*, size_t>& getMemoryMap() override;
    
    // ========== synchronization ==========
    int createStream(GPUStream* stream) override;
    int destroyStream(GPUStream stream) override;
    int createEvent(GPUEvent* event) override;
    int destroyEvent(GPUEvent event) override;
    int recordEvent(GPUEvent event, GPUStream stream) override;
    int synchronizeEvent(GPUEvent event) override;
    int memcpyAsync(void* dst, const void* src, size_t size, 
                   GPUMemcpyKind kind, GPUStream stream) override;
    int synchronizeStream(GPUStream stream) override;
    int syncAllKernels() override;
    int registerHostMemory(void* ptr, size_t size) override;
    
    // ========== checkpoint/restore memory management ==========
    int releasePhysicalMemory(void* ptr) override;
    int remapPhysicalMemory(void* ptr, size_t size) override;
    
    // ========== external tool interfaces ==========
    int externalCheckpoint(int pid) override;
    int externalRestore(int pid) override;
    
    GPUVendor getVendor() const override { return GPUVendor::NVIDIA; }
    std::string getVendorName() const override { return "NVIDIA"; }
};

extern "C" {
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaError_t cudaFree(void *devPtr);
}

#endif // NV_H