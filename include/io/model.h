#pragma once
#include <ios>
#include <string>
#include <fstream>
#include <optional>
#include <json.hpp>

#include <vk/vk_gram.h>
#include <io/tensor.h>

namespace vkdiffusion
{
    class SafeTensorLoader;

    class SafeTensor {
        friend SafeTensorLoader;
    public:
        std::string header_;
        std::ifstream file_;
        std::vector<void*> segments_;
        std::unordered_map<std::string, Tensor> tensors_map_;

        SafeTensor() = default;
    public:
        SafeTensor(SafeTensor&) = delete;
        SafeTensor(SafeTensor&& other) noexcept : header_(std::move(other.header_)), file_(std::move(other.file_)), segments_(std::move(other.segments_)), tensors_map_(std::move(other.tensors_map_)) {}
        SafeTensor& operator=(SafeTensor&) = delete;
        SafeTensor& operator=(SafeTensor&&) = delete;

        Tensor& LoadTensor(const std::string& name) {
            if (!tensors_map_.contains(name)) throw std::runtime_error(std::format("Tensor {} not exists", name));
            return tensors_map_.at(name);
        }

        ~SafeTensor() =default;
    };

    class SafeTensorLoader {
        VkGRAMAllocator& allocator_;
    public:
        SafeTensorLoader(VkGRAMAllocator& allocator) : allocator_(allocator) {

        }
        std::optional<SafeTensor> Load(const std::string& path) {
            using namespace nlohmann;
            std::ifstream input(path, std::ios::in | std::ios::binary);
            if (!input) {
                return std::nullopt;
            }
            uint64_t header_size {};
            input.read(reinterpret_cast<char*>(&header_size), 8);
            std::string header(header_size, 0);
            input.read(&header[0], static_cast<std::streamsize>(header_size));
            json tensors = json::parse(header);
            SafeTensor safe_tensor;
            for (auto& [name, tensor] : tensors.items()) {
                if (name != "__metadata__") {
                    auto shape = tensor["shape"];
                    auto data_off = tensor["data_offsets"];
                    std::vector<uint32_t> shapes;
                    for (uint32_t s : shape) shapes.push_back(s);
                    auto [it, succ] = safe_tensor.tensors_map_.try_emplace(
                        name,
                        allocator_,
                        ParseTensorTypeWithSafeTensorAlias(tensor["dtype"]),
                        shapes,
                        name
                    );
                    if (!succ) throw std::runtime_error(std::format("Repeated Tensor {} in the safetensor file {}", name, path));
                    TensorView view(it->second);
                    view.LoadFromFile(path, 8 + header_size + static_cast<size_t>(data_off[0]));
                    view.Upload();
                }
            }
            return safe_tensor;
        }
    };
}
