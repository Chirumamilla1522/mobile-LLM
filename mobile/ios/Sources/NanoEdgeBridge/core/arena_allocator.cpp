#include "arena_allocator.hpp"
#include <cstdlib>
#include <algorithm>
#include <string>

namespace mllm {

ArenaAllocator::ArenaAllocator(size_t capacity_bytes, size_t alignment)
    : capacity_(capacity_bytes) {
    // Allocate 64-byte aligned memory for cache-line alignment
    int res = ::posix_memalign(reinterpret_cast<void**>(&buffer_), alignment, capacity_);
    if (res != 0 || !buffer_) {
        throw std::bad_alloc();
    }
}

ArenaAllocator::~ArenaAllocator() {
    if (buffer_) {
        std::free(buffer_);
        buffer_ = nullptr;
    }
}

void* ArenaAllocator::allocate(size_t bytes, size_t alignment) {
    size_t current_addr = reinterpret_cast<size_t>(buffer_ + offset_);
    size_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
    size_t new_offset = (aligned_addr - reinterpret_cast<size_t>(buffer_)) + bytes;

    if (new_offset > capacity_) {
        throw std::runtime_error("ArenaAllocator out of memory: requested " + 
                                 std::to_string(bytes) + " bytes, available: " + 
                                 std::to_string(capacity_ - offset_));
    }

    offset_ = new_offset;
    peak_used_ = std::max(peak_used_, offset_);
    return reinterpret_cast<void*>(aligned_addr);
}

} // namespace mllm
