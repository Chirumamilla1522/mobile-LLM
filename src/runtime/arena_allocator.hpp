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

// Double-buffered ping-pong arena for layer activations
// Caps transient activation RAM strictly under 4 MB across all transformer layers
class PingPongArena {
public:
    explicit PingPongArena(size_t buffer_size_bytes = 2 * 1024 * 1024, size_t page_alignment = 16384);
    ~PingPongArena();

    PingPongArena(const PingPongArena&) = delete;
    PingPongArena& operator=(const PingPongArena&) = delete;

    [[nodiscard]] void* current_input() const noexcept { return buffer_a_is_input_ ? buffer_a_ : buffer_b_; }
    [[nodiscard]] void* current_output() const noexcept { return buffer_a_is_input_ ? buffer_b_ : buffer_a_; }

    void swap() noexcept { buffer_a_is_input_ = !buffer_a_is_input_; }
    void reset() noexcept { buffer_a_is_input_ = true; }

    [[nodiscard]] size_t buffer_size() const noexcept { return buffer_size_; }
    [[nodiscard]] size_t total_allocated_bytes() const noexcept { return buffer_size_ * 2; }

private:
    void* buffer_a_{nullptr};
    void* buffer_b_{nullptr};
    size_t buffer_size_{0};
    bool buffer_a_is_input_{true};
};

} // namespace mllm
