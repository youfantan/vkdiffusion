#include <vk/vk_gram.h>
#include <vk/vk_framework.h>

#include <cstring>
#include <stdexcept>

namespace vkdiffusion
{
    VkGRAMBuffer::~VkGRAMBuffer() {
        if (buffer_ != VK_NULL_HANDLE) allocator_.FreeBuffer(this);
    }

    void VkGRAMAllocator::Upload(VkGRAMHandle remote, VkGRAMHandle local) {
        VkBufferCopy copy_region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = local->GetSize()
        };

        auto nval = vk_cmd_dispatcher_.Dispatch([&](VkCommandBuffer command_buffer) {
            VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
            };
            vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1, &barrier,
                0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(command_buffer, local->GetVkBuffer(), remote->GetVkBuffer(), 1, &copy_region);
        });
        remote->GetFenceValue() = nval;
        local->GetFenceValue() = nval;
    }

    void VkGRAMAllocator::Download(VkGRAMHandle remote, VkGRAMHandle local) {
        VkBufferCopy copy_region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = remote->GetSize()
        };
        auto nval = vk_cmd_dispatcher_.Dispatch([&](VkCommandBuffer command_buffer) {
            VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
            };
            vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1, &barrier,
                0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(command_buffer, remote->GetVkBuffer(), local->GetVkBuffer(), 1, &copy_region);
        });
        fence_.CPUWait(nval);
        remote->GetFenceValue() = nval;
        local->GetFenceValue() = nval;
    }
}
