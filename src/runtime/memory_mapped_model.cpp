#include "memory_mapped_model.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace mllm {

static bool range_valid(uint64_t offset, uint64_t size, size_t file_size) {
    return offset <= file_size && size <= file_size - static_cast<size_t>(offset);
}

static bool tensor_size_valid(const TensorDescriptor& desc) {
    const uint64_t rows = desc.rows, cols = desc.cols;
    if (!rows || !cols) return false;
    uint64_t units = 0, bytes_per_unit = 0;
    switch (static_cast<QuantType>(desc.quant_type)) {
        case QuantType::FP32: units = cols; bytes_per_unit = 4; break;
        case QuantType::FP16: units = cols; bytes_per_unit = 2; break;
        case QuantType::INT8: if (cols % 32) return false; units = cols / 32; bytes_per_unit = 34; break;
        case QuantType::Q4_0: if (cols % 32) return false; units = cols / 32; bytes_per_unit = 18; break;
        case QuantType::Q4_1: if (cols % 32) return false; units = cols / 32; bytes_per_unit = 20; break;
        case QuantType::MQ4_APPLE: if (cols % 256) return false; units = cols / 256; bytes_per_unit = 144; break;
        default: return false;
    }
    return units <= UINT64_MAX / rows && units * rows <= UINT64_MAX / bytes_per_unit &&
           units * rows * bytes_per_unit == desc.size_bytes;
}

MemoryMappedModel::MemoryMappedModel(const std::string& filepath) {
    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open model file: " + filepath + " (" + std::strerror(errno) + ")");
    }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        throw std::runtime_error("Failed to fstat model file: " + filepath);
    }
    file_size_ = static_cast<size_t>(st.st_size);

    if (file_size_ < sizeof(ModelHeader)) {
        ::close(fd_);
        throw std::runtime_error("File size is smaller than ModelHeader: " + filepath);
    }

    // MAP_PRIVATE gives copy-on-write semantics without modifying flash storage.
    mmap_base_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("Failed to mmap model file: " + filepath + " (" + std::strerror(errno) + ")");
    }

    // Parse ModelHeader directly from base
    std::memcpy(&header_, mmap_base_, sizeof(ModelHeader));

    auto fail = [&](const std::string& message) -> void {
        ::munmap(mmap_base_, file_size_);
        ::close(fd_);
        mmap_base_ = nullptr;
        fd_ = -1;
        throw std::runtime_error(message);
    };

    if (header_.magic != MLLM_MAGIC) {
        fail("Invalid MLLM magic: 0x" + std::to_string(header_.magic));
    }

    if (header_.version != MLLM_VERSION) {
        fail("Unsupported MLLM version: " + std::to_string(header_.version));
    }

    if (header_.total_file_size != file_size_) fail("Header file size does not match mapped file");

    // Parse Tensor Manifest
    if (!range_valid(header_.manifest_offset, header_.manifest_size, file_size_) ||
        header_.num_tensors > header_.manifest_size / sizeof(TensorDescriptor))
        fail("Corrupted manifest: exceeds file boundaries");

    descriptors_.resize(header_.num_tensors);
    const auto* manifest_ptr = static_cast<const uint8_t*>(mmap_base_) + header_.manifest_offset;
    std::memcpy(descriptors_.data(), manifest_ptr, sizeof(TensorDescriptor) * header_.num_tensors);

    for (size_t i = 0; i < descriptors_.size(); ++i) {
        const auto& desc = descriptors_[i];
        const size_t name_length = ::strnlen(desc.name, sizeof(desc.name));
        if (!tensor_size_valid(desc) || !range_valid(desc.offset, desc.size_bytes, file_size_) ||
            (desc.scales_bytes && !range_valid(desc.scales_offset, desc.scales_bytes, file_size_)))
            fail("Invalid tensor descriptor at index " + std::to_string(i));
        std::string name(desc.name, name_length);
        if (!name_to_index_.emplace(name, i).second) fail("Duplicate tensor name: " + name);
    }
}

MemoryMappedModel::~MemoryMappedModel() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        ::munmap(mmap_base_, file_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

MemoryMappedModel::MemoryMappedModel(MemoryMappedModel&& other) noexcept
    : fd_(other.fd_),
      mmap_base_(other.mmap_base_),
      file_size_(other.file_size_),
      header_(other.header_),
      descriptors_(std::move(other.descriptors_)),
      name_to_index_(std::move(other.name_to_index_)) {
    other.fd_ = -1;
    other.mmap_base_ = nullptr;
    other.file_size_ = 0;
}

MemoryMappedModel& MemoryMappedModel::operator=(MemoryMappedModel&& other) noexcept {
    if (this != &other) {
        if (mmap_base_ && mmap_base_ != MAP_FAILED) {
            ::munmap(mmap_base_, file_size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        mmap_base_ = other.mmap_base_;
        file_size_ = other.file_size_;
        header_ = other.header_;
        descriptors_ = std::move(other.descriptors_);
        name_to_index_ = std::move(other.name_to_index_);

        other.fd_ = -1;
        other.mmap_base_ = nullptr;
        other.file_size_ = 0;
    }
    return *this;
}

const TensorDescriptor* MemoryMappedModel::find_tensor(std::string_view name) const {
    auto it = name_to_index_.find(std::string(name));
    if (it != name_to_index_.end()) {
        return &descriptors_[it->second];
    }
    return nullptr;
}

const TensorDescriptor* MemoryMappedModel::find_layer_tensor(int32_t layer_idx, TensorType type) const {
    for (const auto& desc : descriptors_) {
        if (desc.layer_idx == layer_idx && desc.tensor_type == static_cast<uint32_t>(type)) {
            return &desc;
        }
    }
    return nullptr;
}

const void* MemoryMappedModel::get_tensor_data(const TensorDescriptor& desc) const noexcept {
    if (!range_valid(desc.offset, desc.size_bytes, file_size_)) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_base_) + desc.offset;
}

const void* MemoryMappedModel::get_scales_data(const TensorDescriptor& desc) const noexcept {
    if (desc.scales_offset == 0 || !range_valid(desc.scales_offset, desc.scales_bytes, file_size_)) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_base_) + desc.scales_offset;
}

void MemoryMappedModel::prefetch_range(size_t offset, size_t size) {
    if (offset > file_size_ || size > file_size_ - offset) return;
    void* ptr = static_cast<uint8_t*>(mmap_base_) + offset;
    ::madvise(ptr, size, MADV_WILLNEED);
}

void MemoryMappedModel::prefetch_tensor(const TensorDescriptor& desc) {
    prefetch_range(desc.offset, desc.size_bytes);
    if (desc.scales_bytes > 0) {
        prefetch_range(desc.scales_offset, desc.scales_bytes);
    }
}

void MemoryMappedModel::prefetch_layer(int32_t layer_idx, int32_t lookahead) {
    for (int32_t l = layer_idx; l <= layer_idx + lookahead && l < static_cast<int32_t>(header_.num_layers); ++l) {
        for (const auto& desc : descriptors_) {
            if (desc.layer_idx == l) {
                prefetch_tensor(desc);
            }
        }
    }
}

void MemoryMappedModel::prefetch_all() {
    ::madvise(mmap_base_, file_size_, MADV_WILLNEED);
}

void MemoryMappedModel::evict_layer(int32_t layer_idx) {
    for (const auto& desc : descriptors_) {
        if (desc.layer_idx == layer_idx) {
            void* ptr = static_cast<uint8_t*>(mmap_base_) + desc.offset;
            ::madvise(ptr, desc.size_bytes, MADV_DONTNEED);
        }
    }
}

MemoryStats MemoryMappedModel::query_process_memory_stats() {
    MemoryStats stats{};
#if defined(__APPLE__)
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (::task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &count) == KERN_SUCCESS) {
        stats.virtual_size_bytes = vm_info.virtual_size;
        stats.resident_size_bytes = vm_info.resident_size;
        stats.dirty_size_bytes = vm_info.phys_footprint; // Apple physical memory footprint
    }
#endif
    return stats;
}

} // namespace mllm
