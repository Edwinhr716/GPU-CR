// Multi-threaded memcpy shared by the checkpoint/restore staging pipelines.
// Own translation unit (not vGPU.cpp, which needs the CUDA/HIP link) so the
// GPU-free unit suite can exercise it.

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#include "common.h"

// Helper function: multi-threaded memcpy
void memcpy_multi(void* dest, void* src, size_t size) {
    std::vector<std::thread> threads;
    size_t chunk_size = (size + NUM_COPY_THREADS - 1) / NUM_COPY_THREADS;
    for (int i = 0; i < NUM_COPY_THREADS; i++) {
        size_t offset = i * chunk_size;
        if (offset >= size) break;
        size_t this_chunk_size = std::min(chunk_size, size - offset);
        threads.emplace_back([=]() {
            memcpy((char*)dest + offset, (char*)src + offset, this_chunk_size);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}
