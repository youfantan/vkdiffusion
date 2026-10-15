#include <vk/vk_init.h>

vkdiffusion::VkInitializer::VkInitializer(Type type) : vk_dev_typ_(type) {

}

vkdiffusion::VkInitializer& vkdiffusion::VkInitializer::AddInstanceExtension(const std::string& ext) {
    Add(vk_instance_extensions_, ext);
    return *this;
}

vkdiffusion::VkInitializer& vkdiffusion::VkInitializer::AddInstanceLayer(const std::string& layer) {
    Add(vk_instance_layers_, layer);
    return *this;
}

vkdiffusion::VkInitializer& vkdiffusion::VkInitializer::AddDeviceExtension(const std::string& ext) {
    Add(vk_device_extensions_, ext);
    return *this;
}

vkdiffusion::VkInitializer& vkdiffusion::VkInitializer::AddDeviceLayer(const std::string& layer) {
    Add(vk_device_layers_, layer);
    return *this;
}

vkdiffusion::VkInitializer::~VkInitializer() {
    if (device_ != VK_NULL_HANDLE) {
        for (auto v : fb_views_) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device_, v, nullptr);
        }
        if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
    }
}

void vkdiffusion::VkInitializer::CreateCommonInstance() {
    VkApplicationInfo app_info {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "vkdiffusion",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2
    };
    VkInstanceCreateInfo instance_create_info {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = static_cast<uint32_t>(vk_instance_layers_.size()),
        .ppEnabledLayerNames = vk_instance_layers_.empty() ? nullptr : &vk_instance_layers_[0],
        .enabledExtensionCount = static_cast<uint32_t>(vk_instance_extensions_.size()),
        .ppEnabledExtensionNames = vk_instance_extensions_.empty() ? nullptr : &vk_instance_extensions_[0]
    };
    VKCHECK(vkCreateInstance(&instance_create_info, nullptr, &instance_));
}
