cmake_minimum_required(VERSION 3.20)
project(mllm_runtime LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Optimization flags
add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)

if(APPLE)
    # Enable ARC for Objective-C++ files
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Core runtime library
add_library(mllm_core STATIC
    src/runtime/memory_mapped_model.cpp
    src/runtime/arena_allocator.cpp
    src/runtime/transformer_engine.cpp
    src/kernels/cpu/neon_gemv.cpp
    src/kernels/metal/metal_backend.mm
)

if(APPLE)
    target_link_libraries(mllm_core PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
endif()

# Roofline benchmark executable
add_executable(mllm_benchmark
    bench/roofline_benchmark.cpp
)

target_link_libraries(mllm_benchmark PRIVATE mllm_core)

add_executable(test_transformer
    bench/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE mllm_core)

if(APPLE)
    add_executable(runtime_comparison bench/runtime_comparison.mm)
    target_link_libraries(runtime_comparison PRIVATE mllm_core)
endif()
