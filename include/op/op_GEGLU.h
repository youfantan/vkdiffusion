#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpGEGLU {

        struct GEGLUParams {
            uint32_t R;
            uint32_t F;
        };

        OpGEGLU() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpGEGLU")) {
                auto op_group_norm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpGEGLU.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpGEGLU")
                    .BindComputeShader(op_group_norm_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .PushConstant(0, sizeof(GEGLUParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& input) {
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetShape().size() == 2);
            RUNTIME_ASSERT(input.GetShape()[1] % 2 == 0);
#endif
            return { input.GetShape()[0], input.GetShape()[1] / 2 };
        }

        VkComputeTask operator()(const Tensor& input, Tensor& output) const {
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetShape().size() == 2 && output.GetShape().size() == 2);
            RUNTIME_ASSERT(input.GetShape()[0] == output.GetShape()[0]);
            RUNTIME_ASSERT(input.GetShape()[1] == 2 * output.GetShape()[1]);
            RUNTIME_ASSERT(input.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(output.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(output.GetShape()[1] > 0);
            RUNTIME_ASSERT(
                static_cast<uint64_t>(output.GetShape()[0]) * output.GetShape()[1] <= UINT32_MAX);
#endif
            const uint32_t R = output.GetShape()[0];
            const uint32_t F = output.GetShape()[1];
            const uint32_t element_count = R * F;
            GEGLUParams params {
                .R = R,
                .F = F
            };
            VkComputeTask task("OpGEGLU");
            task.BindBuffer(0, input.GetGPUHandle())
                .BindBuffer(1, output.GetGPUHandle())
                .PushConstant(&params, sizeof(GEGLUParams))
                .Dispatch((element_count + 255) / 256, 1, 1);
            return task;
        }
    };
}
