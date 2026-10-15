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
    public:
        explicit VkCommandDispatcher(VkInitializer& init);
        NO_COPY_MOVE(VkCommandDispatcher);

        using CommandRecorder = std::function<void(VkCommandBuffer)>;
        uint64_t Dispatch(const CommandRecorder& recorder, const std::vector<uint64_t>& wait_fence_values = {});

        VkTimelineFence& GetFence() {
            return fence_;
        }

        ~VkCommandDispatcher();
    };

}
