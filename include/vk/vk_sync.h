#pragma once

#include <base.h>
#include <vk/vk_init.h>

namespace vkdiffusion
{
    class VkTimelineFence {
        const VkInitializer& initializer_;
        VkSemaphore semaphore_ { VK_NULL_HANDLE };
        uint64_t fence_value_ {};
    public:
        VkTimelineFence(const VkInitializer& init) : initializer_(init) {
            VkSemaphoreTypeCreateInfo semaphore_type_create_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = nullptr,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = 0
            };
            VkSemaphoreCreateInfo semaphore_create_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &semaphore_type_create_info,
                .flags = 0
            };
            vkCreateSemaphore(init.GetVkDevice(), &semaphore_create_info, nullptr, &semaphore_);
        }

        uint64_t AllocateValue() {
            return ++fence_value_;
        }

        uint64_t GetCompletedValue() const {
            uint64_t v;
            vkGetSemaphoreCounterValue(initializer_.GetVkDevice(), semaphore_, &v);
            return v;
        }

        VkSemaphore& GetSemaphore() {
            return semaphore_;
        }

        void CPUWait(uint64_t val) const {
            VkSemaphoreWaitInfo semaphore_wait_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = 1,
                .pSemaphores = &semaphore_,
                .pValues = &val
            };
            vkWaitSemaphores(initializer_.GetVkDevice(), &semaphore_wait_info, UINT64_MAX);
        }

        void CPUSignal() const {
            VkSemaphoreSignalInfo semaphore_signal_info {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                .pNext = nullptr,
                .semaphore = semaphore_,
                .value = fence_value_
            };
            vkSignalSemaphore(initializer_.GetVkDevice(), &semaphore_signal_info);
        }

        ~VkTimelineFence() {
            vkDestroySemaphore(initializer_.GetVkDevice(), semaphore_, nullptr);
        }
    };
}