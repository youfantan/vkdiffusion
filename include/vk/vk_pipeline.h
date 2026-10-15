#pragma once

#include <base.h>
#include <map>
#include <vector>
#include <vk/vk_gram.h>

namespace vkdiffusion
{
    class VkComputePipeline {
        VkInitializer& initializer_;
        VkShaderModule shader_ { VK_NULL_HANDLE };
        std::vector<VkDescriptorSetLayoutBinding> bindings_;
        std::vector<VkPushConstantRange> push_constants_;
        VkDescriptorSetLayout set_layout_ { VK_NULL_HANDLE };
        VkPipelineLayout pipeline_layout_ { VK_NULL_HANDLE };
        VkPipeline pipeline_ { VK_NULL_HANDLE };

        void AddBinding(uint32_t binding, VkDescriptorType type);

    public:
        explicit VkComputePipeline(VkInitializer& initializer);
        VkComputePipeline(VkComputePipeline&& other) noexcept;
        NO_COPY(VkComputePipeline);
        VkComputePipeline& operator=(VkComputePipeline&&) = delete;
        VkComputePipeline& BindComputeShader(VkShaderModule sm);
        VkComputePipeline& BindStorageBuffer(uint32_t binding);
        VkComputePipeline& BindUniformBuffer(uint32_t binding);
        VkComputePipeline& BindStorageImage(uint32_t binding);
        VkComputePipeline& BindCombinedImageSampler(uint32_t binding);
        VkComputePipeline& PushConstant(uint32_t offset, uint32_t size, VkShaderStageFlags stage = VK_SHADER_STAGE_COMPUTE_BIT);

        void Build();

        VkPipeline GetVkPipeline() const {
            return pipeline_;
        }

        VkPipelineLayout GetVkPipelineLayout() const {
            return pipeline_layout_;
        }

        VkDescriptorSetLayout GetVkDescriptorSetLayout() const {
            return set_layout_;
        }

        const std::vector<VkDescriptorSetLayoutBinding>& GetBindings() const {
            return bindings_;
        }

        const std::vector<VkPushConstantRange>& GetPushConstants() const {
            return push_constants_;
        }

        ~VkComputePipeline();
    };

    class VkComputeTask {
        std::string pipeline_name_;
        std::vector<VkGRAMHandle> handles_;
        std::vector<uint64_t> fence_values_;
        std::map<uint32_t, VkDescriptorBufferInfo> buffers_;
        std::map<uint32_t, VkDescriptorImageInfo> images_;
        std::vector<std::byte> push_constants_;
        uint32_t push_constant_offset_ = 0;
        uint32_t group_count_[3] = { 1, 1, 1 };
    public:
        explicit VkComputeTask(std::string pipeline_name) : pipeline_name_(std::move(pipeline_name)) {

        }

        VkComputeTask& BindBuffer(uint32_t binding, VkGRAMHandle buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) {
            handles_.push_back(buffer);
            buffers_[binding] = VkDescriptorBufferInfo { buffer->GetVkBuffer(), offset, range };
            return *this;
        }

        VkComputeTask& BindImage(uint32_t binding, VkImageView view, VkSampler sampler = VK_NULL_HANDLE, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
            images_[binding] = VkDescriptorImageInfo { sampler, view, layout };
            return *this;
        }

        VkComputeTask& PushConstant(const void* data, size_t size, uint32_t offset = 0) {
            push_constant_offset_ = offset;
            push_constants_.resize(offset + size);
            std::memcpy(push_constants_.data() + offset, data, size);
            return *this;
        }

        VkComputeTask& Dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
            group_count_[0] = x;
            group_count_[1] = y;
            group_count_[2] = z;
            return *this;
        }

        const std::vector<uint64_t>& GetFenceValues() {
            fence_values_.clear();
            for (auto h : handles_) fence_values_.push_back(h->GetFenceValue());
            return fence_values_;
        }

        void UpdateFenceValues(uint64_t new_fence_value) const {
            for (auto h : handles_) h->GetFenceValue() = new_fence_value;
        }

        const std::string& GetPipelineName() const {
            return pipeline_name_;
        }

        const std::map<uint32_t, VkDescriptorBufferInfo>& GetBuffers() const {
            return buffers_;
        }

        const std::map<uint32_t, VkDescriptorImageInfo>& GetImages() const {
            return images_;
        }

        const std::vector<std::byte>& GetPushConstants() const {
            return push_constants_;
        }

        uint32_t GetPushConstantOffset() const {
            return push_constant_offset_;
        }

        uint32_t GetGroupCount(uint32_t axis) const {
            return group_count_[axis];
        }
    };
}