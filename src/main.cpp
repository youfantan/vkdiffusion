#include <platform/win32.h>

#include "vk/vk_framework.h"

using namespace vkdiffusion;
int main() {
    Window window(800, 600, 0, 0);
    VulkanDevice device(VulkanDevice::Type::Graphics_Compute);
    device
        .AddInstanceLayer("VK_LAYER_KHRONOS_validation")
        .AddInstanceExtension("VK_KHR_surface")
        .AddInstanceExtension("VK_KHR_win32_surface")
        .AddDeviceExtension("VK_KHR_swapchain")
        .Create(window);
    while (!window.Closed()) {
        while (window.FetchMessage()) {}
    }
}