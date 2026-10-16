#pragma once
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "vk/vk_gram.h"

namespace vkdiffusion
{
    struct BufferView {
        BufferView(const void* ptr, size_t size) : ptr(ptr), size(size) {
        }

        template <template <typename...> class C, typename T>
        BufferView(const C<T>& vec) : ptr(vec.data()), size(sizeof(T) * vec.size()) {
        }

        const void* ptr;
        size_t size;
    };

    enum class TensorDataType {
        F32, F64,
        I8, I16, I32, I64,
        U8, U16, U32, U64,
        UNKNOWN,
    };

    template <TensorDataType _DT>
    struct TensorDataTypeTraits {
        constexpr static std::string_view SafeTensorTypeAlias = "UNKNOWN";
        constexpr static std::string_view StandardTypeAlias = "unknown";
        constexpr static size_t Size = 0;
        using Type = void;
        constexpr static TensorDataType DT = _DT;
    };

#define DECLARE_TENSOR_DATA_TYPE_TRAIT(type, st_alias, std_alias, size, local_type) \
        template<> struct TensorDataTypeTraits<type> { \
            constexpr static std::string_view SafeTensorTypeAlias = st_alias; \
            constexpr static std::string_view StandardTypeAlias = std_alias; \
            constexpr static size_t Size = size; \
            using Type = local_type; \
            constexpr static TensorDataType DT = type; \
        };

    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::F32, "F32", "fp32", 4, float);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::F64, "F64", "fp64", 8, double);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::I8, "I8", "int8", 1, int8_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::I16, "I16", "int16", 2, int16_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::I32, "I32", "int32", 4, int32_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::I64, "I64", "int64", 8, int64_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::U8, "U8", "uint8", 1, uint8_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::U16, "U16", "uint16", 2, uint16_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::U32, "U32", "uint32", 4, uint32_t);
    DECLARE_TENSOR_DATA_TYPE_TRAIT(TensorDataType::U64, "U64", "uint64", 8, uint64_t);

    constexpr static std::tuple TensorTypes = {
        TensorDataTypeTraits<TensorDataType::F32>(),
        TensorDataTypeTraits<TensorDataType::F64>(),
        TensorDataTypeTraits<TensorDataType::I8>(),
        TensorDataTypeTraits<TensorDataType::I16>(),
        TensorDataTypeTraits<TensorDataType::I32>(),
        TensorDataTypeTraits<TensorDataType::I64>(),
        TensorDataTypeTraits<TensorDataType::U8>(),
        TensorDataTypeTraits<TensorDataType::U16>(),
        TensorDataTypeTraits<TensorDataType::U32>(),
        TensorDataTypeTraits<TensorDataType::U64>(),
    };

    static constexpr size_t ParseTensorTypeSize(TensorDataType type) {
        return std::apply([type](const auto&... traits) -> size_t {
            size_t size = 0;
            ((traits.DT == type ? (size = traits.Size, true) : false) || ...);
            return size;
        }, TensorTypes);
    }

    static constexpr std::string_view ParseTensorTypeAlias(TensorDataType type) {
        return std::apply([type](const auto&... traits) -> std::string_view {
            std::string_view alias;
            ((traits.DT == type ? (alias = traits.StandardTypeAlias, true) : false) || ...);
            return alias;
        }, TensorTypes);
    }

    static constexpr TensorDataType ParseTensorTypeWithSafeTensorAlias(const std::string& alias) {
        return std::apply([alias](const auto&... traits) -> TensorDataType {
            TensorDataType DT{TensorDataType::UNKNOWN};
            ((traits.SafeTensorTypeAlias == alias ? (DT = traits.DT, true) : false) || ...);
            return DT;
        }, TensorTypes);
    }

    class TensorView;

    class Tensor {
        friend TensorView;

    public:

    private:
        VkGRAMAllocator& allocator_;
        std::vector<uint32_t> shape_;
        std::vector<uint32_t> stride_;
        TensorDataType type_;
        size_t size_;
        VkGRAMHandle gpu_buffer_;
        std::string name_;

    public:
        Tensor(VkGRAMAllocator& allocator, TensorDataType type, const std::vector<uint32_t>& shape, const std::string& name = "") : allocator_(allocator), shape_(shape), type_(type), size_(1), name_(name) {
            for (auto s : shape) size_ *= s;
            size_ *= ParseTensorTypeSize(type);
            stride_.resize(shape_.size());
            size_t k = 1;
            for (int64_t i = shape_.size() - 1; i >= 0; --i) {
                stride_[i] = k;
                k *= shape_[i];
            }
            gpu_buffer_ = allocator.AllocateRemoteBuffer(size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, name);
        }

        NO_COPY(Tensor);
        Tensor(Tensor&& other) noexcept : allocator_(other.allocator_), shape_(std::move(other.shape_)), type_(other.type_), size_(other.size_), gpu_buffer_(other.gpu_buffer_), name_(std::move(other.name_)) {
            other.gpu_buffer_ = nullptr;
        }
        Tensor& operator=(TensorView&&) = delete;

        TensorDataType GetDataType() const {
            return type_;
        }

        const std::vector<uint32_t>& GetShape() const {
            return shape_;
        }

        const std::vector<uint32_t>& GetStride() const {
            return stride_;
        }

        size_t GetSize() const {
            return size_;
        }

        VkGRAMHandle GetGPUHandle() const {
            return gpu_buffer_;
        }

        ~Tensor() {
            if (gpu_buffer_ != nullptr) allocator_.MarkAsExpired(gpu_buffer_);
        }
    };

    class TensorView {
        Tensor& tensor_;
        VkGRAMAllocator& allocator_;
        VkGRAMHandle cpu_buffer_;
    public:
        TensorView(Tensor& tensor) : tensor_(tensor), allocator_(tensor.allocator_), cpu_buffer_(allocator_.AllocateLocalBuffer(tensor.size_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, tensor.name_ + "_View")) {
            memset(cpu_buffer_->GetMapped(), 0, tensor.size_);

        }
        NO_COPY(TensorView);
        TensorView(TensorView&& other) noexcept : tensor_(other.tensor_), allocator_(other.allocator_), cpu_buffer_(other.cpu_buffer_) {
            other.cpu_buffer_ = nullptr;
        }
        TensorView& operator=(TensorView&&) = delete;

        void Download() {
            allocator_.Download(tensor_.gpu_buffer_, cpu_buffer_);
        }

        void Upload() {
            allocator_.Upload(tensor_.gpu_buffer_, cpu_buffer_);
        }

        void LoadFromFile(const std::string& file, size_t offset) {
            auto ptr = cpu_buffer_->GetMapped();
            std::ifstream input(file, std::ios::in | std::ios::binary);
            input.seekg(static_cast<std::istream::off_type>(offset), std::ios::beg);
            if (!input) throw std::runtime_error(std::format("Cannot load tensor from file {} because file not exists", file));
            input.read(static_cast<char*>(ptr), static_cast<std::streamsize>(tensor_.size_));
        }

        void LoadFromMemory(BufferView buffer) {
            auto mapping = cpu_buffer_->GetMapped();
            memcpy(mapping, buffer.ptr, tensor_.size_);
        }

        void DumpToFile(const std::string& file) const {
            std::ofstream output(file, std::ios::out | std::ios::binary);
            output.write(static_cast<const char*>(cpu_buffer_->GetMapped()), tensor_.size_);
        }

        template<TensorDataType _DT>
        TensorDataTypeTraits<_DT>::Type& At(std::initializer_list<uint32_t> index) {
            using AccessType = TensorDataTypeTraits<_DT>::Type;
            if (_DT != tensor_.type_) throw std::runtime_error(std::format("TensorType of Tensor unmatched while access TensorView: Tensor {} / Tensor View {}", ParseTensorTypeAlias(tensor_.type_), TensorDataTypeTraits<_DT>::StandardTypeAlias));
            if (index.size() != tensor_.shape_.size()) {
                throw std::runtime_error(std::format("TensorView access error: shapes not equals. Access {}, Tensor {}", index.size(), tensor_.shape_.size()));
            }
            size_t offset = 0;
            for (auto i = 0; i < index.size(); ++i) {
                offset += index.data()[i] * tensor_.stride_[i];
            }
            if (offset * sizeof(TensorDataTypeTraits<_DT>::Type) >= tensor_.size_) throw std::runtime_error(std::format("Tensor View access out of range: index {} / size {}", offset, tensor_.size_));
            return static_cast<AccessType*>(cpu_buffer_->GetMapped())[offset];
        }

        template<TensorDataType _DT>
        const TensorDataTypeTraits<_DT>::Type& At(std::initializer_list<uint32_t> index) const {
            using AccessType = TensorDataTypeTraits<_DT>::Type;

            if (_DT != tensor_.type_) throw std::runtime_error(std::format("TensorType of Tensor unmatched while access TensorView: Tensor {} / Tensor View {}", ParseTensorTypeAlias(tensor_.type_), TensorDataTypeTraits<_DT>::StandardTypeAlias));
            if (index.size() != tensor_.shape_.size()) {
                throw std::runtime_error(std::format("TensorView access error: shapes not equals. Access {}, Tensor {}",
                                                     index.size(), tensor_.shape_.size()));
            }
            size_t offset = 0;
            for (auto i = 0; i < index.size(); ++i) {
                offset += index.data()[i] * tensor_.stride_[i];
            }
            return static_cast<const AccessType*>(cpu_buffer_->GetMapped())[offset];
        }

        ~TensorView() {
            if (cpu_buffer_ != nullptr) allocator_.MarkAsExpired(cpu_buffer_);
        }
    };

    inline Tensor LoadTensorFromFile(VkGRAMAllocator& allocator, TensorDataType type, const std::vector<uint32_t>& shape, const std::string& path, size_t offset, const std::string& name) {
        Tensor tensor(allocator, type, shape, name);
        TensorView tv(tensor);
        tv.LoadFromFile(path, offset);
        tv.Upload();
        return tensor;
    }
}
