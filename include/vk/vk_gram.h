#pragma once

#include <base.h>
#include <vk/vk_init.h>
#include <vk/vk_sync.h>

#include "utils.h"
#include "vk_commands.h"

namespace vkdiffusion
{
    class VkGRAMBuffer {
        friend VkGRAMAllocator;
        VkGRAMAllocator& allocator_;
        VkBuffer buffer_;
        VkDeviceMemory memory_;
        size_t size_;
        VkBufferUsageFlags usage_;
        VkMemoryPropertyFlags property_;
        void* mapped_;
        uint64_t fence_value_;
        std::string name_;

    public:
        VkGRAMBuffer(VkGRAMAllocator& allocator, VkBuffer buffer, VkDeviceMemory memory, size_t size, void* mapped, VkBufferUsageFlags usage, VkMemoryPropertyFlags property, const std::string& name) :
        allocator_(allocator), buffer_(buffer), memory_(memory), size_(size), mapped_(mapped), usage_(usage), property_(property), fence_value_(0), name_(name) {

        }
        NO_COPY(VkGRAMBuffer);
        VkGRAMBuffer(VkGRAMBuffer&& other) noexcept : allocator_(other.allocator_), buffer_(other.buffer_), memory_(other.memory_),
        size_(other.size_), usage_(other.usage_), property_(other.property_), mapped_(other.mapped_), fence_value_(other.fence_value_), name_(std::move(other.name_)) {
            other.buffer_ = VK_NULL_HANDLE;
            other.memory_ = VK_NULL_HANDLE;
            other.fence_value_ = 0;
        }
        VkGRAMBuffer& operator=(VkGRAMBuffer&& other) = delete;

        uint64_t& GetFenceValue() {
            return fence_value_;
        }

        size_t GetSize() const {
            return size_;
        }

        VkBuffer GetVkBuffer() const {
            return buffer_;
        }

        VkBufferUsageFlags GetUsage() const {
            return usage_;
        }

        VkMemoryPropertyFlags GetProperty() const {
            return property_;
        }

        void* GetMapped() const {
            return mapped_;
        }

        const std::string& GetName() const {
            return name_;
        }

        ~VkGRAMBuffer();
    };


    using VkGRAMHandle = VkGRAMBuffer*;

    class VkGRAMAllocator {
        friend VkGRAMBuffer;
    public:
    private:
        VkInitializer& vk_init_;
        VkCommandDispatcher& vk_cmd_dispatcher_;
        VkTimelineFence& fence_;
        VkGRAMHandle CreateBuffer(VkDeviceSize length, VkBufferUsageFlags usage, VkMemoryPropertyFlags property, bool mapped, const std::string& name) {
            VkDevice device = vk_init_.GetVkDevice();
            VkBufferCreateInfo buffer_create_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = length,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            VkBuffer buffer = VK_NULL_HANDLE;
            VKCHECK(vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer));
            VkMemoryRequirements requirements;
            vkGetBufferMemoryRequirements(device, buffer, &requirements);
            auto find = [&]() {
                const VkPhysicalDeviceMemoryProperties& memory_properties = vk_init_.GetMemoryProperties();
                for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
                    if ((requirements.memoryTypeBits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & property) == property) {
                        return i;
                    }
                }
                throw std::runtime_error("No matched memory type bits found while creating buffer");
            };
            VkMemoryAllocateInfo memory_allocate_info {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = requirements.size,
                .memoryTypeIndex = find()
            };
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VKCHECK(vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory));
            VKCHECK(vkBindBufferMemory(device, buffer, memory, 0));
            void* pointer = nullptr;
            if (mapped) {
                VKCHECK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &pointer));
            }
            auto handle = new VkGRAMBuffer(*this, buffer, memory, length, pointer, usage, property, name);
            return handle;
        }

        void FreeBuffer(VkGRAMHandle handle) {
            if (handle->mapped_ != nullptr) {
                vkUnmapMemory(vk_init_.GetVkDevice(), handle->memory_);
            }
            auto name = handle->GetName().empty() ? "[Anonymous]" : handle->GetName();
            if (handle->GetProperty() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                std::cout << std::format("Local VkBuffer '{}' is released", name) << std::endl;
            } else if (handle->GetProperty() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                std::cout << std::format("Remote VkBuffer '{}' is released", name) << std::endl;
            } else {
                std::cout << std::format("Unknown VkBuffer '{}' is released", name) << std::endl;
            }
            vkFreeMemory(vk_init_.GetVkDevice(), handle->memory_, nullptr);
            vkDestroyBuffer(vk_init_.GetVkDevice(), handle->buffer_, nullptr);
        }

        std::vector<VkGRAMHandle> gram_buffers_;
        std::vector<VkGRAMHandle> dying_buffers_;
    public:
        VkGRAMAllocator(VkInitializer& init, VkCommandDispatcher& dispatcher) : vk_init_(init), vk_cmd_dispatcher_(dispatcher), fence_(dispatcher.GetFence()) {

        }

        VkGRAMHandle AllocateLocalBuffer(size_t length, VkBufferUsageFlags usage, const std::string& name = "") {
            gram_buffers_.emplace_back(CreateBuffer(length, usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, name));
            return gram_buffers_.back();
        }
        VkGRAMHandle AllocateRemoteBuffer(size_t length, VkBufferUsageFlags usage, const std::string& name = "") {
            gram_buffers_.emplace_back(CreateBuffer(length, usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, name));
            return gram_buffers_.back();
        }

        void MarkAsExpired(VkGRAMHandle handle) {
            std::erase_if(gram_buffers_, [&](auto h) { return h == handle; });
            dying_buffers_.push_back(handle);
            auto name = handle->GetName().empty() ? "[Anonymous]" : handle->GetName();
            std::cout << std::format("VkBuffer '{}' is dying", name) << std::endl;
        }

        void CleanUp() {
            std::vector<VkGRAMHandle> erase_handles;
            for (auto h : dying_buffers_) {
                if (h->GetFenceValue() <= fence_.GetCompletedValue()) {
                    delete h;
                    erase_handles.push_back(h);
                }
            }
            std::erase_if(dying_buffers_, [&](auto w) {
                for (auto h : erase_handles) {
                    return h == w;
                }
                return false;
            });
        }

        void Upload(VkGRAMHandle remote, VkGRAMHandle local);
        void Download(VkGRAMHandle remote, VkGRAMHandle local);

        ~VkGRAMAllocator() {
            for (auto h : gram_buffers_) dying_buffers_.push_back(h);
            CleanUp();
        }
    };

}
