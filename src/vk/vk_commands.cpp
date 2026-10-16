#include <vk/vk_framework.h>
#include <vk/vk_commands.h>

#include <algorithm>

namespace vkdiffusion
{
    VkCommandDispatcher::InflightBuffer& VkCommandDispatcher::DispatchBuffer() {
        uint64_t min_i = 0;
        uint64_t min_fv = inflight_buffers_[min_i].fence_value;
        for (auto i = 1; i < inflight_buffers_.size(); ++i) {
            if (min_fv >= inflight_buffers_[i].fence_value) {
                min_i = i;
                min_fv = inflight_buffers_[min_i].fence_value;
            }
        }
        if (fence_.GetCompletedValue() <= min_fv) {
            fence_.CPUWait(min_fv);
        }
        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };
        VKCHECK(vkBeginCommandBuffer(inflight_buffers_[min_i].buffer, &begin_info));
        inflight_buffers_[min_i].fence_value = UINT64_MAX;
        return inflight_buffers_[min_i];
    }

    VkCommandDispatcher::VkCommandDispatcher(VkInitializer& init, uint64_t cmd_buffers_count) : vk_init_(init), queue_(init.GetVkQueue()), fence_(init), inflight_buffers_(cmd_buffers_count) {
        VkCommandPoolCreateInfo pool_create_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = static_cast<uint32_t>(vk_init_.GetSelectedQueueFamily())
        };
        VKCHECK(vkCreateCommandPool(vk_init_.GetVkDevice(), &pool_create_info, nullptr, &pool_));
        FORI(cmd_buffers_count) {
            VkCommandBufferAllocateInfo allocate_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = pool_,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
            VKCHECK(vkAllocateCommandBuffers(vk_init_.GetVkDevice(), &allocate_info, &inflight_buffers_[i].buffer));
        }
    }

    uint64_t VkCommandDispatcher::Dispatch(const CommandRecorder& recorder, uint64_t sync_value) {
        auto& inflight = DispatchBuffer();
        recorder(inflight.buffer);
        VKCHECK(vkEndCommandBuffer(inflight.buffer));
        uint64_t nval = fence_.AllocateValue();
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = sync_value ? 1u : 0u,
            .pWaitSemaphoreValues = sync_value ? &sync_value : nullptr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &nval
        };
        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_semaphore_submit_info,
            .waitSemaphoreCount = sync_value ? 1u : 0u,
            .pWaitSemaphores = sync_value ? &fence_.GetSemaphore() : nullptr,
            .pWaitDstStageMask = sync_value ? &wait_stage : nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &inflight.buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &fence_.GetSemaphore()
        };
        VKCHECK(vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
        inflight.fence_value = nval;
        return nval;
    }

    uint64_t VkCommandDispatcher::BatchedDispatch(uint64_t batch, const BatchedCommandRecorder& recorder, uint64_t sync_value) {
        if (batch > inflight_buffers_.size()) throw std::runtime_error(std::format("Size of batch is larger than size of command buffers, max {} / batch {}", inflight_buffers_.size(), batch));
        InflightBuffer& inflight= DispatchBuffer();
        for (uint64_t i = 0; i < batch; ++i) {
            recorder(i, inflight.buffer);
            VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT  | VK_ACCESS_SHADER_WRITE_BIT
            };
            vkCmdPipelineBarrier(
                inflight.buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        VKCHECK(vkEndCommandBuffer(inflight.buffer));
        uint64_t nval = fence_.AllocateValue();
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = sync_value ? 1u : 0u,
            .pWaitSemaphoreValues = sync_value ? &sync_value : nullptr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &nval
        };
        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_semaphore_submit_info,
            .waitSemaphoreCount = sync_value ? 1u : 0u,
            .pWaitSemaphores = sync_value ? &fence_.GetSemaphore() : nullptr,
            .pWaitDstStageMask = sync_value ? &wait_stage : nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &inflight.buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &fence_.GetSemaphore()
        };
        VKCHECK(vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
        inflight.fence_value = nval;
        return nval;
    }

    VkCommandDispatcher::~VkCommandDispatcher() {
        for (auto inflight : inflight_buffers_) {
            if (fence_.GetCompletedValue() <= inflight.fence_value) {
                fence_.CPUWait(inflight.fence_value);
            }
            vkFreeCommandBuffers(vk_init_.GetVkDevice(), pool_, 1, &inflight.buffer);
        }
        vkDestroyCommandPool(vk_init_.GetVkDevice(), pool_, nullptr);
    }

}
