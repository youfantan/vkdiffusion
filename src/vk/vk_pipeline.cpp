#include <vk/vk_pipeline.h>
#include <vk/vk_init.h>

vkdiffusion::VkComputePipeline::VkComputePipeline(VkInitializer& initializer) : initializer_(initializer) {
}

vkdiffusion::VkComputePipeline::VkComputePipeline(VkComputePipeline&& other) noexcept :
    initializer_(other.initializer_),
    shader_(other.shader_), bindings_(std::move(other.bindings_)), push_constants_(std::move(other.push_constants_)),
    set_layout_(other.set_layout_), pipeline_layout_(other.pipeline_layout_),
    pipeline_(other.pipeline_) {
    other.pipeline_ = VK_NULL_HANDLE;
    other.pipeline_layout_ = VK_NULL_HANDLE;
    other.set_layout_ = VK_NULL_HANDLE;
}

vkdiffusion::VkComputePipeline& vkdiffusion::VkComputePipeline::BindComputeShader(VkShaderModule sm) {
    shader_ = sm;
    return *this;
}

vkdiffusion::VkComputePipeline& vkdiffusion::VkComputePipeline::BindStorageBuffer(uint32_t binding) {
    AddBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    return *this;
}

vkdiffusion::VkComputePipeline& vkdiffusion::VkComputePipeline::BindUniformBuffer(uint32_t binding) {
    AddBinding(binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    return *this;
}

vkdiffusion::VkComputePipeline& vkdiffusion::VkComputePipeline::BindStorageImage(uint32_t binding) {
    AddBinding(binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    return *this;
}

vkdiffusion::VkComputePipeline& vkdiffusion::VkComputePipeline::BindCombinedImageSampler(uint32_t binding) {
    AddBinding(binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    return *this;
}

vkdiffusion::VkComputePipeline& vkdiffusion::VkComputePipeline::PushConstant(
    uint32_t offset, uint32_t size, VkShaderStageFlags stage) {
    VkPushConstantRange range {
        .stageFlags = stage,
        .offset = offset,
        .size = size
    };
    push_constants_.push_back(range);
    return *this;
}

void vkdiffusion::VkComputePipeline::Build() {
    if (shader_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Compute pipeline has no shader bound");
    }
    if (!bindings_.empty()) {
        VkDescriptorSetLayoutCreateInfo set_layout_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(bindings_.size()),
            .pBindings = bindings_.data()
        };
        VKCHECK(
            vkCreateDescriptorSetLayout(initializer_.GetVkDevice(), &set_layout_create_info, nullptr, &set_layout_));
    }
    VkPipelineLayoutCreateInfo pipeline_layout_create_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = set_layout_ == VK_NULL_HANDLE ? 0u : 1u,
        .pSetLayouts = set_layout_ == VK_NULL_HANDLE ? nullptr : &set_layout_,
        .pushConstantRangeCount = static_cast<uint32_t>(push_constants_.size()),
        .pPushConstantRanges = push_constants_.empty() ? nullptr : push_constants_.data()
    };
    VKCHECK(vkCreatePipelineLayout(initializer_.GetVkDevice(), &pipeline_layout_create_info, nullptr, &pipeline_layout_));
    VkPipelineShaderStageCreateInfo shader_stage_create_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_,
        .pName = "main",
        .pSpecializationInfo = nullptr
    };

    VkComputePipelineCreateInfo compute_pipeline_create_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = shader_stage_create_info,
        .layout = pipeline_layout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };
    VKCHECK(
        vkCreateComputePipelines(initializer_.GetVkDevice(), VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr,
            &pipeline_));
}

void vkdiffusion::VkComputePipeline::AddBinding(uint32_t binding, VkDescriptorType type) {
    VkDescriptorSetLayoutBinding b {
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr
    };
    bindings_.push_back(b);
}

vkdiffusion::VkComputePipeline::~VkComputePipeline() {
    VkDevice device = initializer_.GetVkDevice();
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
}
