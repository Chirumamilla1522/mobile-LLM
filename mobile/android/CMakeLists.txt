cmake_minimum_required(VERSION 3.22)
project(nanoedge_android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

add_library(nanoedge SHARED
    jni/nanoedge_jni.cpp
    ../../src/runtime/memory_mapped_model.cpp
    ../../src/runtime/arena_allocator.cpp
    ../../src/kernels/cpu/neon_gemv.cpp
)

find_library(LOG_LIB log)
target_link_libraries(nanoedge
    ${LOG_LIB}
)
