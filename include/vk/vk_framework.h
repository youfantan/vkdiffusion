#pragma once

#include <base.h>
#include <string>
#include <vector>
#include <format>
#include <list>

#include "utils.h"

namespace vkdiffusion
{

    class VulkanDevice {
    public:
        enum class Type {
            Graphics,
            Compute,
            Graphics_Compute
        };
    private:
        Type vk_dev_typ_;

        VkSurfaceKHR surface_;
        VkInstance instance_;
        VkDevice device_;
        VkQueue queue_;
        VkSwapchainKHR swapchain_;

        std::list<std::string> container_;
        std::vector<const char*> vk_instance_extensions_;
        std::vector<const char*> vk_instance_layers_;
        std::vector<const char*> vk_device_extensions_;
        std::vector<const char*> vk_device_layers_;
        std::vector<VkPhysicalDevice> physical_devices_;
        std::vector<VkQueueFamilyProperties> queue_family_properties_;
        VkPhysicalDevice selected_physical_device_;
        int64_t selected_queue_family;
        std::vector<VkImageView> fb_views_;
        VkSurfaceCapabilitiesKHR surface_capabilities_;

        void Add(std::vector<const char*>& list, const std::string& name) {
            container_.push_back(name);
            list.emplace_back(container_.back().c_str());
        }
    public:
        explicit VulkanDevice(Type type) : vk_dev_typ_(type) {

        }

        VulkanDevice& AddInstanceExtension(const std::string& ext) {
            Add(vk_instance_extensions_, ext);
            return *this;
        }

        VulkanDevice& AddInstanceLayer(const std::string& layer) {
            Add(vk_instance_layers_, layer);
            return *this;
        }

        VulkanDevice& AddDeviceExtension(const std::string& ext) {
            Add(vk_device_extensions_, ext);
            return *this;
        }

        VulkanDevice& AddDeviceLayer(const std::string& layer) {
            Add(vk_device_layers_, layer);
            return *this;
        }

        template<typename Window>
        bool Create(Window& window) {
            VkApplicationInfo app_info {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = nullptr,
                .pApplicationName = "vkdiffusion",
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = "No Engine",
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = VK_API_VERSION_1_0
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
            surface_ = window.CreateSurface(instance_);
            uint32_t phdev_count;
            VKCHECK(vkEnumeratePhysicalDevices(instance_, &phdev_count, nullptr));
            if (phdev_count == 0) {
                std::cout << "No available physical devices found" << std::endl;
                exit(EXIT_FAILURE);
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
                std::cout << "Queue family count is 0" << std::endl;
                exit(EXIT_FAILURE);
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
                if (vk_dev_typ_ == Type::Compute && queue_family_properties_[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    selected_queue_family = i;
                }

                if (vk_dev_typ_ == Type::Graphics_Compute && queue_family_properties_[i].queueFlags & VK_QUEUE_COMPUTE_BIT && queue_family_properties_[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && support_present) {
                    selected_queue_family = i;
                }
            }
            if (selected_queue_family == -1) {
                std::cout << "No available queue family found" << std::endl;
                exit(EXIT_FAILURE);
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
            VkPhysicalDeviceFeatures features {};
            VkDeviceCreateInfo device_create_info {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &queue_create_info,
                .enabledLayerCount = static_cast<uint32_t>(vk_device_layers_.size()),
                .ppEnabledLayerNames = vk_device_layers_.empty() ? nullptr : &vk_device_layers_[0],
                .enabledExtensionCount = static_cast<uint32_t>(vk_device_extensions_.size()),
                .ppEnabledExtensionNames = vk_device_extensions_.empty() ? nullptr : &vk_device_extensions_[0],
                .pEnabledFeatures = &features
            };
            VKCHECK(vkCreateDevice(selected_physical_device_, &device_create_info, nullptr, &device_));
            vkGetDeviceQueue(device_, selected_queue_family, 0, &queue_);
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(selected_physical_device_, surface_, &surface_capabilities_);
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

        VkDevice GetVkDevice() {
            return device_;
        }

        const VkSurfaceCapabilitiesKHR& GetVkSurfaceCapabilities() const {
            return surface_capabilities_;
        }

        ~VulkanDevice() {
            for (auto v : fb_views_) {
                vkDestroyImageView(device_, v, nullptr);
            }
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            vkDestroyDevice(device_, nullptr);
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            vkDestroyInstance(instance_, nullptr);
        }
    };

    class ShaderManager {
        VulkanDevice vk_dev_;
        std::list<std::string> container_;
    public:
        ShaderManager(VulkanDevice& vk_dev) : vk_dev_(vk_dev) {

        }

        std::optional<VkShaderModule> LoadShaderSPV(const std::string& shader_path) {
            auto bytes = ReadFileIntoString(shader_path);
            container_.emplace_back(std::move(bytes.value()));
            if (!bytes) return std::nullopt;
            VkShaderModuleCreateInfo module_create_info {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = bytes.value().size(),
                .pCode = reinterpret_cast<const uint32_t*>(&container_.back()[0])
            };
            VkShaderModule sm;
            vkCreateShaderModule(vk_dev_.GetVkDevice(), &module_create_info, nullptr, &sm);
            return sm;
        }
    };

    class VulkanGraphicsPipeline {
        VulkanDevice& vk_dev_;
        struct Shaders {
            VkPipelineShaderStageCreateInfo VS;
            VkPipelineShaderStageCreateInfo PS;
        } shaders_;
        std::vector<VkDynamicState> dynamic_states_;
        VkViewport vk_viewport_;
        VkGraphicsPipelineCreateInfo graphics_pipline_create_info_;
        VkPipelineShaderStageCreateInfo shader_stage_create_info_;
        VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info_;
        VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info_;
        VkPipelineTessellationStateCreateInfo tessellation_state_create_info_;
        VkPipelineViewportStateCreateInfo viewport_state_create_info_;
        VkPipelineRasterizationStateCreateInfo rasterization_state_create_info_;
        VkPipelineMultisampleStateCreateInfo multisample_state_create_info_;
        VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info_;
        VkPipelineColorBlendStateCreateInfo color_blend_state_create_info_;
        VkPipelineDynamicStateCreateInfo dynamic_state_create_info_;
        VkRect2D vk_scissor_;

    public:
        explicit VulkanGraphicsPipeline(VulkanDevice& vk_dev) : vk_dev_(vk_dev), shaders_(),
        vk_viewport_{
            .x = 0,
            .y = 0,
            .width = static_cast<float>(vk_dev_.GetVkSurfaceCapabilities().currentExtent.width),
            .height = static_cast<float>(vk_dev_.GetVkSurfaceCapabilities().currentExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        },
        vk_scissor_{
                    .offset = { 0, 0 },
                    .extent = vk_dev_.GetVkSurfaceCapabilities().currentExtent
        } {
            dynamic_state_create_info_ =  {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .dynamicStateCount = static_cast<uint32_t>(dynamic_states_.size()),
                .pDynamicStates = dynamic_states_.empty() ? nullptr : &dynamic_states_[0]
            };

            viewport_state_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .viewportCount = 1,
                .pViewports = &vk_viewport_,
                .scissorCount = 1,
                .pScissors = &vk_scissor_
            };

            rasterization_state_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .depthBiasConstantFactor = 0.0f,
                .depthBiasClamp = 0.0f,
                .depthBiasSlopeFactor = 0.0f,
                .lineWidth = 1.0f
            };

            multisample_state_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 1.0f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE
            };

            color_blend_state_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .logicOpEnable = ,
                .logicOp = ,
                .attachmentCount = ,
                .pAttachments = ,
                .blendConstants =
            }

            vertex_input_state_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .vertexBindingDescriptionCount = 0,
                .pVertexBindingDescriptions = nullptr,
                .vertexAttributeDescriptionCount = 0,
                .pVertexAttributeDescriptions = nullptr
            };

            input_assembly_state_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE
            };

            graphics_pipline_create_info_ = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stageCount = ,
                .pStages = ,
                .pVertexInputState = &vertex_input_state_create_info_,
                .pInputAssemblyState = &input_assembly_state_create_info_,
                .pTessellationState = nullptr,
                .pViewportState = &viewport_state_create_info_,
                .pRasterizationState = &rasterization_state_create_info_,
                .pMultisampleState = &multisample_state_create_info_,
                .pDepthStencilState = nullptr,
                .pColorBlendState = &color_blend_state_create_info_,
                .pDynamicState = &dynamic_state_create_info_,
                .layout = ,
                .renderPass = ,
                .subpass = ,
                .basePipelineHandle = ,
                .basePipelineIndex =
            };
        }

        VulkanGraphicsPipeline& BindVertexShader(VkShaderModule sm) {
            shaders_.VS = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = sm,
                .pName = "main",
                .pSpecializationInfo = nullptr
            };
        }

        VulkanGraphicsPipeline& BindPixelShader(VkShaderModule sm) {
            shaders_.VS = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = sm,
                .pName = "main",
                .pSpecializationInfo = nullptr
            };
        }

        void Build() {
        }
    };

}
