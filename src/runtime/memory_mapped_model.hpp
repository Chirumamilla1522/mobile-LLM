#pragma once

#include "mllm/model_format.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <system_error>

namespace mllm {

struct MemoryStats {
    uint64_t virtual_size_bytes{0};
    uint64_t resident_size_bytes{0};    // RSS
    uint64_t dirty_size_bytes{0};       // Actual modified physical memory
    uint64_t mapped_file_bytes{0};      // Total mmap address space
};

class MemoryMappedModel {
public:
    explicit MemoryMappedModel(const std::string& filepath);
    ~MemoryMappedModel();

    // Disable copy, enable move
    MemoryMappedModel(const MemoryMappedModel&) = delete;
    MemoryMappedModel& operator=(const MemoryMappedModel&) = delete;
    MemoryMappedModel(MemoryMappedModel&& other) noexcept;
    MemoryMappedModel& operator=(MemoryMappedModel&& other) noexcept;

    // Direct metadata access
    [[nodiscard]] const ModelHeader& header() const noexcept { return header_; }
    [[nodiscard]] const std::vector<TensorDescriptor>& descriptors() const noexcept { return descriptors_; }
    [[nodiscard]] size_t total_mapped_bytes() const noexcept { return file_size_; }

    // Fast tensor lookup
    [[nodiscard]] const TensorDescriptor* find_tensor(std::string_view name) const;
    [[nodiscard]] const TensorDescriptor* find_layer_tensor(int32_t layer_idx, TensorType type) const;

    // Zero-copy pointer resolution (zero allocations)
    [[nodiscard]] const void* get_tensor_data(const TensorDescriptor& desc) const noexcept;
    [[nodiscard]] const void* get_scales_data(const TensorDescriptor& desc) const noexcept;

    // Asynchronous virtual memory hints
    void prefetch_range(size_t offset, size_t size);
    void prefetch_tensor(const TensorDescriptor& desc);
    void prefetch_layer(int32_t layer_idx, int32_t lookahead = 1);
    void prefetch_all();
    void evict_layer(int32_t layer_idx);

    // Live OS memory metrics (Apple Mach task_info)
    static MemoryStats query_process_memory_stats();

private:
    int fd_{-1};
    void* mmap_base_{nullptr};
    size_t file_size_{0};
    ModelHeader header_{};
    std::vector<TensorDescriptor> descriptors_;
    std::unordered_map<std::string, size_t> name_to_index_;
};

} // namespace mllm
