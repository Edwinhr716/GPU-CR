# ---------------------------------------------------------------------------
# NCCL checkpoint adapter
#
# Build with:
#   cmake -DGPU_CR_BUILD_NCCL_ADAPTER=ON -DNCCL_ROOT=/path/to/nccl ..
#
# Produces:
#   libgcr_preload.so           — runtime + IPC tracking, LD_PRELOAD into the worker
#   libnccl-checkpoint-gcr.so   — thin NCCL plugin that bridges to the runtime
#   two_proc_nccl_test          — optional example (built only if libnccl is found)
# ---------------------------------------------------------------------------

if(NOT NCCL_ROOT)
    message(FATAL_ERROR
        "GPU_CR_BUILD_NCCL_ADAPTER=ON requires -DNCCL_ROOT=/path/to/nccl "
        "(an NCCL source tree, ideally already built)")
endif()

# Public header (generated): nccl.h
find_path(NCCL_PUBLIC_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${NCCL_ROOT}/include
        ${NCCL_ROOT}/build/include
        ${NCCL_ROOT}/nccl_install_stage2/include
        ${NCCL_ROOT}/nccl_install_stage1/include
    NO_DEFAULT_PATH)
if(NOT NCCL_PUBLIC_INCLUDE_DIR)
    message(FATAL_ERROR
        "Cannot find nccl.h. Searched under ${NCCL_ROOT}/{include,build/include,"
        "nccl_install_stage{1,2}/include}. Override with -DNCCL_PUBLIC_INCLUDE_DIR=...")
endif()
message(STATUS "NCCL public header dir: ${NCCL_PUBLIC_INCLUDE_DIR}")

# Internal header (in source tree): nccl_common.h, needed by nccl_checkpoint.h
find_path(NCCL_INTERNAL_INCLUDE_DIR
    NAMES nccl_common.h
    HINTS ${NCCL_ROOT}/src/include
    NO_DEFAULT_PATH)
if(NOT NCCL_INTERNAL_INCLUDE_DIR)
    message(FATAL_ERROR
        "Cannot find nccl_common.h under ${NCCL_ROOT}/src/include. "
        "Override with -DNCCL_INTERNAL_INCLUDE_DIR=...")
endif()
message(STATUS "NCCL internal header dir: ${NCCL_INTERNAL_INCLUDE_DIR}")

# Optional library for the example binary.
find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        ${NCCL_ROOT}/lib
        ${NCCL_ROOT}/build/lib
        ${NCCL_ROOT}/nccl_install_stage2/lib
        ${NCCL_ROOT}/nccl_install_stage1/lib
    NO_DEFAULT_PATH)

# ---------------------------------------------------------------------------
# libgcr_preload.so — runtime + IPC tracking
# ---------------------------------------------------------------------------
add_library(gcr_preload SHARED
    runtime/gcr_checkpoint_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/ipc_hooks.cpp
    ${CMAKE_SOURCE_DIR}/src/ipc_fd_exchange.cpp
)
set_target_properties(gcr_preload PROPERTIES
    OUTPUT_NAME "gcr_preload"
    PREFIX "lib"
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)
target_include_directories(gcr_preload PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${GPU_INCLUDE_DIR}
)
target_link_directories(gcr_preload PRIVATE ${GPU_LIB_DIR})
target_link_libraries(gcr_preload PRIVATE ${GPU_LIBS} dl pthread)
target_compile_options(gcr_preload PRIVATE -fPIC -O2 -g)

# ---------------------------------------------------------------------------
# libnccl-checkpoint-gcr.so — NCCL plugin that dispatches into the runtime
# ---------------------------------------------------------------------------
add_library(nccl_checkpoint_gcr SHARED
    plugin/nccl_checkpoint_gcr_plugin.cpp
)
set_target_properties(nccl_checkpoint_gcr PROPERTIES
    OUTPUT_NAME "nccl-checkpoint-gcr"
    PREFIX "lib"
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)
target_include_directories(nccl_checkpoint_gcr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl_patches     # bundled nccl_checkpoint.h (patch source)
    ${NCCL_PUBLIC_INCLUDE_DIR}
    ${NCCL_INTERNAL_INCLUDE_DIR}
)
target_link_libraries(nccl_checkpoint_gcr PRIVATE dl)
target_compile_options(nccl_checkpoint_gcr PRIVATE -fPIC -O2 -g)

# ---------------------------------------------------------------------------
# Optional example: two_proc_nccl_test (needs libnccl.so)
# ---------------------------------------------------------------------------
if(NCCL_LIBRARY)
    message(STATUS "Building NCCL adapter example with: ${NCCL_LIBRARY}")
    add_executable(two_proc_nccl_test examples/two_proc_nccl_test.cpp)
    target_include_directories(two_proc_nccl_test PRIVATE
        ${NCCL_PUBLIC_INCLUDE_DIR}
        ${GPU_INCLUDE_DIR}
    )
    target_link_directories(two_proc_nccl_test PRIVATE ${GPU_LIB_DIR})
    target_link_libraries(two_proc_nccl_test PRIVATE
        ${NCCL_LIBRARY} ${GPU_LIBS} dl pthread
    )
    target_compile_options(two_proc_nccl_test PRIVATE -O2 -g)
else()
    message(STATUS "libnccl not found under ${NCCL_ROOT}; skipping two_proc_nccl_test "
        "(override with -DNCCL_LIBRARY=/path/to/libnccl.so)")
endif()
