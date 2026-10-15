#pragma once

#include <base.h>
#include <utils.h>
#include <list>
#include <optional>
#include <vector>
#include <shaderc/shaderc.hpp>

namespace vkdiffusion
{
    class VkShaderManager {
    public:
        enum class ShaderType {
            VERTEX_SHADER,
            PIXEL_SHADER,
            COMPUTE_SHADER,
        };
    private:
        VkInitializer& vk_init_;
        std::list<std::string> container_;
        std::vector<VkShaderModule> modules_;
        shaderc_compiler_t compiler_;
        shaderc_shader_kind ParseShaderType(ShaderType typ) {
            if (typ == ShaderType::VERTEX_SHADER) return shaderc_vertex_shader;
            if (typ == ShaderType::PIXEL_SHADER) return shaderc_fragment_shader;
            if (typ == ShaderType::COMPUTE_SHADER) return shaderc_compute_shader;
            return shaderc_vertex_shader;
        }
    public:
        explicit VkShaderManager(VkInitializer& vk_dev) : vk_init_(vk_dev), compiler_(shaderc_compiler_initialize()) {

        }
        NO_COPY_MOVE(VkShaderManager);
        VkShaderModule CompileShader(const std::string& shader_src_path, ShaderType type, const std::string& entry_func = "main");
        VkShaderModule LoadShaderSPV(const std::string& shader_path);
        ~VkShaderManager();
    };
}
