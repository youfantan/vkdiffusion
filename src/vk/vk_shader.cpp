#include <vk/vk_shader.h>
#include <vk/vk_framework.h>

VkShaderModule vkdiffusion::VkShaderManager::CompileShader(const std::string& shader_src_path, ShaderType type, const std::string& entry_func) {
    auto src = ReadFileIntoString(shader_src_path);
    if (!src) throw std::runtime_error(std::format("Shader source file {} not exists", shader_src_path));
    shaderc_compile_options_t opt = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(opt, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
    shaderc_compile_options_set_optimization_level(opt, shaderc_optimization_level_performance);
    shaderc_compilation_result_t r = shaderc_compile_into_spv(
        compiler_, src->c_str(), src->size(), ParseShaderType(type), shader_src_path.c_str(), entry_func.c_str(), opt
    );
    if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success) {
        shaderc_compile_options_release(opt);
        std::cout << std::format("Shader source compile error: \n{}", shaderc_result_get_error_message(r)) << std::endl;
        exit(EXIT_FAILURE);
    }
    container_.emplace_back(shaderc_result_get_bytes(r), shaderc_result_get_length(r));
    shaderc_result_release(r);
    shaderc_compile_options_release(opt);
    const std::string& spv = container_.back();
    VkShaderModuleCreateInfo module_create_info {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spv.size(),
        .pCode = reinterpret_cast<const uint32_t*>(spv.data())
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    VKCHECK(vkCreateShaderModule(vk_init_.GetVkDevice(), &module_create_info, nullptr, &sm));
    modules_.push_back(sm);
    return sm;
}

VkShaderModule vkdiffusion::VkShaderManager::LoadShaderSPV(const std::string& shader_path) {
    auto bytes = ReadFileIntoString(shader_path);
    if (!bytes) throw std::runtime_error(std::format("Shader spv file {} not exists", shader_path));
    container_.emplace_back(std::move(bytes.value()));
    const std::string& spv = container_.back();
    VkShaderModuleCreateInfo module_create_info {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spv.size(),
        .pCode = reinterpret_cast<const uint32_t*>(spv.data())
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    VKCHECK(vkCreateShaderModule(vk_init_.GetVkDevice(), &module_create_info, nullptr, &sm));
    modules_.push_back(sm);
    return sm;
}

vkdiffusion::VkShaderManager::~VkShaderManager() {
    for (auto sm : modules_) {
        vkDestroyShaderModule(vk_init_.GetVkDevice(), sm, nullptr);
    }
    shaderc_compiler_release(compiler_);
}
