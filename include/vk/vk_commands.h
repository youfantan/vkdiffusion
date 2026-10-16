#pragma once

#include <base.h>
#include <functional>

#include "vk_sync.h"

namespace vkdiffusion
{
    class VkCommandDispatcher {
        VkInitializer& vk_init_;
        VkQueue queue_;
        VkCommandPool pool_ { VK_NULL_HANDLE };
        VkTimelineFence fence_;
        struct InflightBuffer {
            VkCommandBuffer buffer;
            uint64_t fence_value;
        };
        std::vector<InflightBuffer> inflight_buffers_;

        InflightBuffer& DispatchBuffer();
    public:
        explicit VkCommandDispatcher(VkInitializer& init, uint64_t cmd_buffers_count);
        NO_COPY_MOVE(VkCommandDispatcher);

        using CommandRecorder = std::function<void(VkCommandBuffer)>;
        uint64_t Dispatch(const CommandRecorder& recorder, uint64_t sync_value = 0);

        using BatchedCommandRecorder = std::function<void(uint64_t, VkCommandBuffer)>;
        uint64_t BatchedDispatch(uint64_t batch, const BatchedCommandRecorder& recorder, uint64_t sync_value = 0);
        VkTimelineFence& GetFence() {
            return fence_;
        }

        ~VkCommandDispatcher();
    };

}
