cmake_minimum_required(VERSION 3.18)
project(vGPU LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect GPU vendor from environment variable
# Usage: GPU_VENDOR=NVIDIA cmake .. or GPU_VENDOR=AMD cmake ..
if(DEFINED ENV{GPU_VENDOR})
    set(GPU_VENDOR $ENV{GPU_VENDOR})
else()
    set(GPU_VENDOR "NVIDIA")  # Default to NVIDIA
endif()

message(STATUS "Building for GPU vendor: ${GPU_VENDOR}")

# Set paths based on vendor
if(GPU_VENDOR STREQUAL "NVIDIA")
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
    set(GPU_INCLUDE_DIR "${CUDA_ROOT}/include")
    set(GPU_LIB_DIR "${CUDA_ROOT}/lib64")
    set(GPU_LIBS cudart cuda)
elseif(GPU_VENDOR STREQUAL "AMD")
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation directory")
    set(GPU_INCLUDE_DIR "${ROCM_ROOT}/include")
    set(GPU_LIB_DIR "${ROCM_ROOT}/lib")
    set(GPU_LIBS amdhip64)
    # Add HIP platform definition
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
else()
    message(FATAL_ERROR "Unknown GPU_VENDOR: ${GPU_VENDOR}. Must be NVIDIA or AMD")
endif()

# Collect source and header files automatically from src/
file(GLOB_RECURSE VGPU_SOURCE_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# 排除不需要编译的文件
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/cr_client\\.cpp$")
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "-pre\\.cpp$")     # 排除 *-pre.cpp 文件

# 根据 GPU vendor 排除对应的目录
if(GPU_VENDOR STREQUAL "NVIDIA")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/AMD/")       # 排除 AMD 目录
    message(STATUS "Excluding AMD files")
elseif(GPU_VENDOR STREQUAL "AMD")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/NVIDIA/")    # 排除 NVIDIA 目录
    message(STATUS "Excluding NVIDIA files")
endif()

file(GLOB_RECURSE VGPU_HEADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

set(SOURCES
    ${VGPU_SOURCE_FILES}
    ${VGPU_HEADER_FILES}
)

# Create shared library for LD_PRELOAD
add_library(vGPU SHARED ${SOURCES})

# Set library name based on vendor (vGPU-nvidia.so or vGPU-amd.so)
set_target_properties(vGPU PROPERTIES
    OUTPUT_NAME "vGPU-${GPU_VENDOR}"
    PREFIX ""
)

# Include directories
target_include_directories(vGPU PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link directories
target_link_directories(vGPU PRIVATE
    ${GPU_LIB_DIR}
)

# Link libraries
target_link_libraries(vGPU PRIVATE
    ${GPU_LIBS}
    dl
    stdc++
    pthread
)

# Compiler flags (matching gcc command: -fPIC -O3 -g)
target_compile_options(vGPU PRIVATE
    -fPIC
    -O3
    -g
)

# Linker flags (matching gcc command: -shared -fPIC)
target_link_options(vGPU PRIVATE
    -shared
    -fPIC
)

# Export all symbols for LD_PRELOAD (needed for constructor and signal handlers)
set_target_properties(vGPU PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

add_executable(cr_client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cr_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/comm/share_mem.cpp
)

target_include_directories(cr_client PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_directories(cr_client PRIVATE
    ${GPU_LIB_DIR}
)

target_link_libraries(cr_client PRIVATE
    ${GPU_LIBS}
    dl
    stdc++
    pthread
)

target_compile_options(cr_client PRIVATE
    -O3
    -g
)
