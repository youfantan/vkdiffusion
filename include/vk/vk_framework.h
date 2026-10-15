#pragma once

#include <base.h>
#include <stdexcept>
#include <utility>

#include <vk/vk_init.h>
#include <vk/vk_commands.h>
#include <vk/vk_gram.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_shader.h>

namespace vkdiffusion
{
    class VkComputeFramework {
        VkInitializer initializer_;
        Lazy<VkCommandDispatcher> dispatcher_;
        Lazy<VkShaderManager> shader_manager_;
        Lazy<VkGRAMAllocator> allocator_;
        std::map<std::string, VkComputePipeline> pipelines_;
        VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    public:
        VkComputeFramework() : initializer_(VkInitializer::Type::Compute) {
            initializer_.Create();
            dispatcher_.Construct(initializer_);
            shader_manager_.Construct(initializer_);
            allocator_.Construct(initializer_, dispatcher_);
            VkFenceCreateInfo fence_create_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT
            };
            VkDescriptorPoolSize pool_sizes[] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
            };
            VkDescriptorPoolCreateInfo pool_create_info {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .maxSets = 256,
                .poolSizeCount = 4,
                .pPoolSizes = pool_sizes
            };
            VKCHECK(vkCreateDescriptorPool(initializer_.GetVkDevice(), &pool_create_info, nullptr, &descriptor_pool_));
        }

        VkComputePipeline& CreatePipeline(const std::string& name) {
            if (pipelines_.contains(name)) throw std::runtime_error(std::format("Compute Pipeline {} already exists.", name));
            auto [it, ok] = pipelines_.try_emplace(name, initializer_);
            return it->second;
        }

        void ExecuteTask(VkComputeTask& task) {
            auto it = pipelines_.find(task.GetPipelineName());
            if (it == pipelines_.end()) {
                throw std::runtime_error(std::format("Compute Pipeline {} not found.", task.GetPipelineName()));
            }
            VkComputePipeline& pipeline = it->second;
            VkDevice device = initializer_.GetVkDevice();
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

            const auto& bindings = pipeline.GetBindings();
            if (!bindings.empty()) {
                VkDescriptorSetLayout set_layout = pipeline.GetVkDescriptorSetLayout();
                VkDescriptorSetAllocateInfo allocate_info {
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .pNext = nullptr,
                    .descriptorPool = descriptor_pool_,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &set_layout
                };
                VKCHECK(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set));

                std::vector<VkWriteDescriptorSet> writes;
                writes.reserve(bindings.size());
                for (const auto& binding : bindings) {
                    auto buffer_it = task.GetBuffers().find(binding.binding);
                    auto image_it = task.GetImages().find(binding.binding);
                    VkWriteDescriptorSet write {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = descriptor_set,
                        .dstBinding = binding.binding,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = binding.descriptorType,
                        .pImageInfo = image_it == task.GetImages().end() ? nullptr : &image_it->second,
                        .pBufferInfo = buffer_it == task.GetBuffers().end() ? nullptr : &buffer_it->second,
                        .pTexelBufferView = nullptr
                    };
                    writes.push_back(write);
                }
                vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }

            auto sync = dispatcher_->Dispatch([&](VkCommandBuffer command_buffer) {
                VkMemoryBarrier barrier {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                };
                vkCmdPipelineBarrier(
                    command_buffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &barrier,
                    0, nullptr, 0, nullptr);
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.GetVkPipeline());
                if (descriptor_set != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.GetVkPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);
                }
                if (!task.GetPushConstants().empty()) {
                    vkCmdPushConstants(command_buffer, pipeline.GetVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, task.GetPushConstantOffset(), static_cast<uint32_t>(task.GetPushConstants().size()), task.GetPushConstants().data());
                }
                vkCmdDispatch(command_buffer, task.GetGroupCount(0), task.GetGroupCount(1), task.GetGroupCount(2));
            }, task.GetFenceValues());
            task.UpdateFenceValues(sync);
            if (descriptor_set != VK_NULL_HANDLE) {
                VKCHECK(vkResetDescriptorPool(device, descriptor_pool_, 0));
            }
        }

        VkShaderManager& GetShaderManager() {
            return shader_manager_;
        }

        VkGRAMAllocator& GetAllocator() {
            return allocator_;
        }

        ~VkComputeFramework() {
            if (descriptor_pool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(initializer_.GetVkDevice(), descriptor_pool_, nullptr);
            }
        }
    };

}
