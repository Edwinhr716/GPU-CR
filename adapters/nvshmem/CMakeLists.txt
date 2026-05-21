# ---------------------------------------------------------------------------
# NVSHMEM checkpoint examples
#
# NVSHMEM itself does not need a patch — GPU-CR's `ipc_hooks.cpp` already
# intercepts the cuMem calls NVSHMEM uses (via dlsym + cuGetProcAddress_v2),
# so this directory only ships demonstrator examples + run scripts.
#
# Build with:
#   cmake -DGPU_VENDOR=NVIDIA \
#         -DGPU_CR_BUILD_NVSHMEM_ADAPTER=ON \
#         -DNVSHMEM_ROOT=/path/to/nvshmem ..
#
# Optional: -DNVSHMEM_PUBLIC_INCLUDE_DIR=... and -DNVSHMEM_LIBRARY=...
# ---------------------------------------------------------------------------

if(NOT NVSHMEM_ROOT)
    message(FATAL_ERROR
        "GPU_CR_BUILD_NVSHMEM_ADAPTER=ON requires -DNVSHMEM_ROOT=/path/to/nvshmem")
endif()

find_path(NVSHMEM_PUBLIC_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS
        ${NVSHMEM_ROOT}/include
        ${NVSHMEM_ROOT}/build/include
    NO_DEFAULT_PATH)
if(NOT NVSHMEM_PUBLIC_INCLUDE_DIR)
    message(FATAL_ERROR
        "Cannot find nvshmem.h under ${NVSHMEM_ROOT}/{include,build/include}. "
        "Override with -DNVSHMEM_PUBLIC_INCLUDE_DIR=...")
endif()
message(STATUS "NVSHMEM include dir: ${NVSHMEM_PUBLIC_INCLUDE_DIR}")

# pip-installed nvidia-nvshmem ships versioned .so only (libnvshmem_host.so.3,
# no unversioned symlink). Include the versioned filename in NAMES so
# find_library catches both layouts.
find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem_host libnvshmem_host.so.3 libnvshmem_host.so
    HINTS
        ${NVSHMEM_ROOT}/lib
        ${NVSHMEM_ROOT}/build/lib
    NO_DEFAULT_PATH)
find_library(NVSHMEM_DEVICE_LIBRARY
    NAMES nvshmem_device libnvshmem_device.a
    HINTS
        ${NVSHMEM_ROOT}/lib
        ${NVSHMEM_ROOT}/build/lib
    NO_DEFAULT_PATH)

# Host example — plain C++/CUDA-runtime, no device kernel.
if(NVSHMEM_HOST_LIBRARY)
    add_executable(nvshmem_host_test examples/nvshmem_host_test.cpp)
    target_include_directories(nvshmem_host_test PRIVATE
        ${NVSHMEM_PUBLIC_INCLUDE_DIR}
        ${GPU_INCLUDE_DIR}
    )
    target_link_directories(nvshmem_host_test PRIVATE ${GPU_LIB_DIR})
    target_link_libraries(nvshmem_host_test PRIVATE
        ${NVSHMEM_HOST_LIBRARY} ${GPU_LIBS} dl pthread
    )
    target_compile_options(nvshmem_host_test PRIVATE -O2 -g)
    message(STATUS "Building NVSHMEM host example with: ${NVSHMEM_HOST_LIBRARY}")
else()
    message(STATUS "libnvshmem_host not found — skipping nvshmem_host_test")
endif()

# Device example — has a CUDA kernel; needs CUDA language + nvshmem_device.
if(NVSHMEM_DEVICE_LIBRARY)
    # CUDA architectures: caller can override via -DCMAKE_CUDA_ARCHITECTURES=...
    # (Use bool test, not NOT DEFINED — CMake may auto-cache an empty value.)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;90" CACHE STRING "CUDA arches for NVSHMEM device kernel" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    add_executable(nvshmem_device_test examples/nvshmem_device_test.cu)
    set_target_properties(nvshmem_device_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(nvshmem_device_test PRIVATE
        ${NVSHMEM_PUBLIC_INCLUDE_DIR}
        ${GPU_INCLUDE_DIR}
    )
    target_link_directories(nvshmem_device_test PRIVATE ${GPU_LIB_DIR})
    target_link_libraries(nvshmem_device_test PRIVATE
        ${NVSHMEM_DEVICE_LIBRARY}
        $<IF:$<BOOL:${NVSHMEM_HOST_LIBRARY}>,${NVSHMEM_HOST_LIBRARY},>
        ${GPU_LIBS} dl pthread
    )
    target_compile_options(nvshmem_device_test PRIVATE -O2 -g)
    message(STATUS "Building NVSHMEM device example with: ${NVSHMEM_DEVICE_LIBRARY}")
else()
    message(STATUS "libnvshmem_device not found — skipping nvshmem_device_test")
endif()
