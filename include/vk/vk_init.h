#pragma once

#include <base.h>

namespace vkdiffusion
{
    class VkInitializer {
    public:
        enum class Type {
            Graphics,
            Compute,
            Graphics_Compute
        };
    private:
        Type vk_dev_typ_;

        VkSurfaceKHR surface_ { VK_NULL_HANDLE };
        VkInstance instance_ { VK_NULL_HANDLE };
        VkDevice device_ { VK_NULL_HANDLE };
        VkQueue queue_ { VK_NULL_HANDLE };
        VkSwapchainKHR swapchain_ { VK_NULL_HANDLE };

        std::list<std::string> container_;
        std::vector<const char*> vk_instance_extensions_;
        std::vector<const char*> vk_instance_layers_;
        std::vector<const char*> vk_device_extensions_;
        std::vector<const char*> vk_device_layers_;
        std::vector<VkPhysicalDevice> physical_devices_;
        std::vector<VkQueueFamilyProperties> queue_family_properties_;
        VkPhysicalDevice selected_physical_device_ { VK_NULL_HANDLE };
        int64_t selected_queue_family { -1 };
        std::vector<VkImageView> fb_views_;
        VkSurfaceCapabilitiesKHR surface_capabilities_ {};
        VkPhysicalDeviceMemoryProperties memory_properties_ {};

        void Add(std::vector<const char*>& list, const std::string& name) {
            container_.push_back(name);
            list.emplace_back(container_.back().c_str());
        }

        void CreateCommonInstance();
    public:
        explicit VkInitializer(Type type);
        NO_COPY_MOVE(VkInitializer);

        VkInitializer& AddInstanceExtension(const std::string& ext);
        VkInitializer& AddInstanceLayer(const std::string& layer);
        VkInitializer& AddDeviceExtension(const std::string& ext);
        VkInitializer& AddDeviceLayer(const std::string& layer);

        template<typename Window>
        void Create(Window& window) {
            if (vk_dev_typ_ != Type::Graphics && vk_dev_typ_ != Type::Graphics_Compute) {
                throw std::runtime_error("Only Graphics or Graphics_Compute type device could call Create(Window)");
            }
            CreateCommonInstance();
            surface_ = window.CreateSurface(instance_);
            uint32_t phdev_count;
            VKCHECK(vkEnumeratePhysicalDevices(instance_, &phdev_count, nullptr));
            if (phdev_count == 0) {
                throw std::runtime_error("No available physical devices found");
            }
            physical_devices_.resize(phdev_count);
            VKCHECK(vkEnumeratePhysicalDevices(instance_, &phdev_count, &physical_devices_[0]));
            std::cout << "Detected physical devices: " << std::endl;
            FORI(phdev_count) {
                VkPhysicalDeviceProperties prop;
                vkGetPhysicalDeviceProperties(physical_devices_[i], &prop);
                std::cout << std::format("{}. {} [Vulkan API Version {}.{}.{}]", i, prop.deviceName, VK_API_VERSION_MAJOR(prop.apiVersion), VK_API_VERSION_MINOR(prop.apiVersion), VK_API_VERSION_PATCH(prop.apiVersion)) << std::endl;
            }
            selected_physical_device_ = physical_devices_[0];
            uint32_t queue_count;
            vkGetPhysicalDeviceQueueFamilyProperties(selected_physical_device_, &queue_count, nullptr);
            if (queue_count == 0) {
                throw std::runtime_error("Queue family count is 0");
            }
            queue_family_properties_.resize(queue_count);
            vkGetPhysicalDeviceQueueFamilyProperties(selected_physical_device_, &queue_count, &queue_family_properties_[0]);
            selected_queue_family = -1;
            FORI(queue_count) {
                VkBool32 support_present;
                vkGetPhysicalDeviceSurfaceSupportKHR(selected_physical_device_, i, surface_, &support_present);
                if (vk_dev_typ_ == Type::Graphics && queue_family_properties_[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && support_present) {
                    selected_queue_family = i;
                }
                if (vk_dev_typ_ == Type::Graphics_Compute && queue_family_properties_[i].queueFlags & VK_QUEUE_COMPUTE_BIT && queue_family_properties_[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && support_present) {
                    selected_queue_family = i;
                }
            }
            if (selected_queue_family == -1) {
               throw std::runtime_error("No available queue family found");
            }
            float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_create_info {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = static_cast<uint32_t>(selected_queue_family),
                .queueCount = 1,
                .pQueuePriorities = &priority
            };
            VkPhysicalDeviceVulkan12Features phy_features {};
            phy_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            phy_features.timelineSemaphore = true;
            VkPhysicalDeviceFeatures2 features {};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &phy_features;

            VkDeviceCreateInfo device_create_info {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = &features,
                .flags = 0,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &queue_create_info,
                .enabledLayerCount = static_cast<uint32_t>(vk_device_layers_.size()),
                .ppEnabledLayerNames = vk_device_layers_.empty() ? nullptr : &vk_device_layers_[0],
                .enabledExtensionCount = static_cast<uint32_t>(vk_device_extensions_.size()),
                .ppEnabledExtensionNames = vk_device_extensions_.empty() ? nullptr : &vk_device_extensions_[0],
                .pEnabledFeatures = nullptr
            };
            VKCHECK(vkCreateDevice(selected_physical_device_, &device_create_info, nullptr, &device_));
            vkGetDeviceQueue(device_, selected_queue_family, 0, &queue_);
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(selected_physical_device_, surface_, &surface_capabilities_);
            vkGetPhysicalDeviceMemoryProperties(selected_physical_device_, &memory_properties_);
            VkSwapchainCreateInfoKHR swapchain_create_info {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = nullptr,
                .flags = 0,
                .surface = surface_,
                .minImageCount = 2,
                .imageFormat = VK_FORMAT_R8G8B8A8_UNORM,
                .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                .imageExtent = surface_capabilities_.currentExtent,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR,
                .clipped = VK_TRUE,
                .oldSwapchain = VK_NULL_HANDLE
            };
            VKCHECK(vkCreateSwapchainKHR(device_, &swapchain_create_info, nullptr, &swapchain_));
            uint32_t image_count;
            VKCHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr));
            std::vector<VkImage> framebuffers(image_count);
            VKCHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, &framebuffers[0]));
            fb_views_.resize(image_count);
            FORI(image_count) {
                VkImageViewCreateInfo image_view_create_info {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .image = framebuffers[i],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = VK_FORMAT_R8G8B8A8_UNORM,
                    .components = {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY
                    },
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 1,
                        .layerCount = 1
                    }
                };
                VKCHECK(vkCreateImageView(device_, &image_view_create_info, nullptr, &fb_views_[i]));
            }
        }

        void Create() {
            if (vk_dev_typ_ != Type::Compute) {
                throw std::runtime_error("Only Compute type device could call Create()");
            }
            CreateCommonInstance();
            uint32_t phdev_count;
            VKCHECK(vkEnumeratePhysicalDevices(instance_, &phdev_count, nullptr));
            if (phdev_count == 0) {
                throw std::runtime_error("No available physical devices found");
            }
            physical_devices_.resize(phdev_count);
            VKCHECK(vkEnumeratePhysicalDevices(instance_, &phdev_count, &physical_devices_[0]));
            std::cout << "Detected physical devices: " << std::endl;
            FORI(phdev_count) {
                VkPhysicalDeviceProperties prop;
                vkGetPhysicalDeviceProperties(physical_devices_[i], &prop);
                std::cout << std::format("{}. {} [Vulkan API Version {}.{}.{}]", i, prop.deviceName, VK_API_VERSION_MAJOR(prop.apiVersion), VK_API_VERSION_MINOR(prop.apiVersion), VK_API_VERSION_PATCH(prop.apiVersion)) << std::endl;
            }
            selected_physical_device_ = physical_devices_[0];
            uint32_t queue_count;
            vkGetPhysicalDeviceQueueFamilyProperties(selected_physical_device_, &queue_count, nullptr);
            if (queue_count == 0) {
                throw std::runtime_error("Queue family count is 0");
            }
            queue_family_properties_.resize(queue_count);
            vkGetPhysicalDeviceQueueFamilyProperties(selected_physical_device_, &queue_count, &queue_family_properties_[0]);
            selected_queue_family = -1;
            FORI(queue_count) {
                if (queue_family_properties_[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    selected_queue_family = i;
                }
            }
            if (selected_queue_family == -1) {
                throw std::runtime_error("No available queue family found");
            }
            float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_create_info {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = static_cast<uint32_t>(selected_queue_family),
                .queueCount = 1,
                .pQueuePriorities = &priority
            };
            VkPhysicalDeviceVulkan12Features phy_features {};
            phy_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            phy_features.timelineSemaphore = true;
            VkPhysicalDeviceFeatures2 features {};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &phy_features;

            VkDeviceCreateInfo device_create_info {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = &features,
                .flags = 0,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &queue_create_info,
                .enabledLayerCount = static_cast<uint32_t>(vk_device_layers_.size()),
                .ppEnabledLayerNames = vk_device_layers_.empty() ? nullptr : &vk_device_layers_[0],
                .enabledExtensionCount = static_cast<uint32_t>(vk_device_extensions_.size()),
                .ppEnabledExtensionNames = vk_device_extensions_.empty() ? nullptr : &vk_device_extensions_[0],
                .pEnabledFeatures = nullptr
            };
            VKCHECK(vkCreateDevice(selected_physical_device_, &device_create_info, nullptr, &device_));
            vkGetDeviceQueue(device_, selected_queue_family, 0, &queue_);
            vkGetPhysicalDeviceMemoryProperties(selected_physical_device_, &memory_properties_);
        }

        VkInstance GetVkInstance() const {
            return instance_;
        }

        VkDevice GetVkDevice() const {
            return device_;
        }

        VkQueue GetVkQueue() const {
            return queue_;
        }

        VkPhysicalDevice GetVkPhysicalDevice() const {
            return selected_physical_device_;
        }

        int64_t GetSelectedQueueFamily() const {
            return selected_queue_family;
        }

        const VkSurfaceCapabilitiesKHR& GetVkSurfaceCapabilities() const {
            return surface_capabilities_;
        }

        const VkPhysicalDeviceMemoryProperties& GetMemoryProperties() const {
            return memory_properties_;
        }

        ~VkInitializer();
    };
}
