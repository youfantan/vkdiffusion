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
    class SafetensorsLoader;

    class Safetensors {
        friend SafetensorsLoader;
        VkGRAMAllocator& allocator_;
        std::string header_;
        std::string path_;
        std::ifstream file_;
        std::vector<void*> segments_;
        std::unordered_map<std::string, Tensor> tensors_map_;
        struct TensorIndex {
            std::string name;
            uint64_t real_offset;
            TensorDataType data_type;
            std::vector<uint32_t> shapes;
        };
        std::unordered_map<std::string, TensorIndex> tensors_index_;

        Safetensors(VkGRAMAllocator& allocator) : allocator_(allocator) {

        }
    public:
        Safetensors(Safetensors&) = delete;
        Safetensors(Safetensors&& other) noexcept : allocator_(other.allocator_), header_(std::move(other.header_)), path_(std::move(other.path_)), file_(std::move(other.file_)), segments_(std::move(other.segments_)), tensors_map_(std::move(other.tensors_map_)) {}
        Safetensors& operator=(Safetensors&) = delete;
        Safetensors& operator=(Safetensors&&) = delete;

        Tensor& LoadTensor(const std::string& name) {
            if (!tensors_index_.contains(name)) throw std::runtime_error(std::format("Tensor {} not exists", name));
            auto& index = tensors_index_[name];
            if (!tensors_map_.contains(name)) {
                auto [it, ok] = tensors_map_.try_emplace(name, allocator_, index.data_type, index.shapes, name);
                TensorView view(it->second);
                view.LoadFromFile(path_, index.real_offset);
                view.Upload();
            }
            return tensors_map_.at(name);
        }

        ~Safetensors() =default;
    };

    class SafetensorsLoader {
        VkGRAMAllocator& allocator_;
    public:
        SafetensorsLoader(VkGRAMAllocator& allocator) : allocator_(allocator) {

        }
        Safetensors Load(const std::string& path) {
            using namespace nlohmann;
            std::ifstream input(path, std::ios::in | std::ios::binary);
            if (!input) {
                throw std::runtime_error(std::format("Cannot load safetensor because file {} not exists", path));
            }
            uint64_t header_size {};
            input.read(reinterpret_cast<char*>(&header_size), 8);
            std::string header(header_size, 0);
            input.read(&header[0], static_cast<std::streamsize>(header_size));
            json tensors = json::parse(header);
            Safetensors safetensors(allocator_);
            safetensors.path_ = path;
            for (auto& [name, tensor] : tensors.items()) {
                if (name != "__metadata__") {
                    auto shape = tensor["shape"];
                    uint64_t off = tensor["data_offsets"][0];
                    std::vector<uint32_t> shapes;
                    for (uint32_t s : shape) shapes.push_back(s);
                    auto [it, success] = safetensors.tensors_index_.try_emplace(name, name, 8 + header_size + off, ParseTensorTypeWithSafeTensorAlias(tensor["dtype"]), std::move(shapes));
                    if (!success) throw std::runtime_error(std::format("Repeated Tensor {} in the safetensors file {}", name, path));
                }
            }
            std::cout << std::format("safetensor {} is loaded into GRAM", path) << std::endl;
            return safetensors;
        }
    };
}
