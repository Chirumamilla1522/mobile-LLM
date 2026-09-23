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

    if (header_.magic != MLLM_MAGIC) {
        ::munmap(mmap_base_, file_size_);
        ::close(fd_);
        throw std::runtime_error("Invalid MLLM magic: 0x" + std::to_string(header_.magic));
    }

    if (header_.version != MLLM_VERSION) {
        ::munmap(mmap_base_, file_size_);
        ::close(fd_);
        throw std::runtime_error("Unsupported MLLM version: " + std::to_string(header_.version));
    }

    // Parse Tensor Manifest
    if (header_.manifest_offset + header_.manifest_size > file_size_) {
        ::munmap(mmap_base_, file_size_);
        ::close(fd_);
        throw std::runtime_error("Corrupted manifest: exceeds file boundaries");
    }

    descriptors_.resize(header_.num_tensors);
    const auto* manifest_ptr = static_cast<const uint8_t*>(mmap_base_) + header_.manifest_offset;
    std::memcpy(descriptors_.data(), manifest_ptr, sizeof(TensorDescriptor) * header_.num_tensors);

    for (size_t i = 0; i < descriptors_.size(); ++i) {
        name_to_index_[std::string(descriptors_[i].name)] = i;
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
    if (desc.offset + desc.size_bytes > file_size_) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_base_) + desc.offset;
}

const void* MemoryMappedModel::get_scales_data(const TensorDescriptor& desc) const noexcept {
    if (desc.scales_offset == 0 || desc.scales_offset + desc.scales_bytes > file_size_) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_base_) + desc.scales_offset;
}

void MemoryMappedModel::prefetch_range(size_t offset, size_t size) {
    if (offset + size > file_size_) return;
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
