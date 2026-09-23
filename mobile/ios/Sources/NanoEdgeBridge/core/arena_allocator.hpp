#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>

namespace mllm {

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity_bytes, size_t alignment = 64);
    ~ArenaAllocator();

    // Non-copyable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Linear bump allocation
    void* allocate(size_t bytes, size_t alignment = 64);

    template<typename T>
    T* allocate_array(size_t count, size_t alignment = 64) {
        return static_cast<T*>(allocate(sizeof(T) * count, alignment));
    }

    // Reset watermark back to 0 (reusable for next token step)
    void reset() noexcept {
        offset_ = 0;
    }

    // Stats
    [[nodiscard]] size_t used_bytes() const noexcept { return offset_; }
    [[nodiscard]] size_t capacity_bytes() const noexcept { return capacity_; }
    [[nodiscard]] size_t peak_used_bytes() const noexcept { return peak_used_; }
    [[nodiscard]] void* base() const noexcept { return buffer_; }

private:
    uint8_t* buffer_{nullptr};
    size_t capacity_{0};
    size_t offset_{0};
    size_t peak_used_{0};
};

} // namespace mllm
