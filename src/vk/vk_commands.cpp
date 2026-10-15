#include <vk/vk_framework.h>
#include <vk/vk_commands.h>

namespace vkdiffusion
{

    VkCommandDispatcher::VkCommandDispatcher(VkInitializer& init) : vk_init_(init), queue_(init.GetVkQueue()), fence_(init) {
        VkCommandPoolCreateInfo pool_create_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = static_cast<uint32_t>(vk_init_.GetSelectedQueueFamily())
        };
        VKCHECK(vkCreateCommandPool(vk_init_.GetVkDevice(), &pool_create_info, nullptr, &pool_));
    }

    uint64_t VkCommandDispatcher::Dispatch(const CommandRecorder& recorder, const std::vector<uint64_t>& wait_fence_values) {
        VkCommandBufferAllocateInfo allocate_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };

        VkCommandBuffer buffer;
        VKCHECK(vkAllocateCommandBuffers(vk_init_.GetVkDevice(), &allocate_info, &buffer));
        VKCHECK(vkBeginCommandBuffer(buffer, &begin_info));
        recorder(buffer);
        VKCHECK(vkEndCommandBuffer(buffer));
        uint64_t nval = fence_.AllocateValue();
        VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = static_cast<uint32_t>(wait_fence_values.size()),
            .pWaitSemaphoreValues = wait_fence_values.empty() ? nullptr : wait_fence_values.data(),
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &nval
        };
        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_semaphore_submit_info,
            .waitSemaphoreCount = wait_fence_values.empty() ? 0u : 1u,
            .pWaitSemaphores = wait_fence_values.empty() ? nullptr : &fence_.GetSemaphore(),
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &fence_.GetSemaphore()
        };
        VKCHECK(vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
        vkFreeCommandBuffers(vk_init_.GetVkDevice(), pool_, 1, &buffer);
        return nval;
    }

    VkCommandDispatcher::~VkCommandDispatcher() {
        vkDestroyCommandPool(vk_init_.GetVkDevice(), pool_, nullptr);
    }

}
