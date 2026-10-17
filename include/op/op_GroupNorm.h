#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpGroupNorm {

        struct GroupNormParams {
            uint32_t input_shape[4];
            uint32_t input_stride[4];
            uint32_t weight_shape;
            uint32_t groups;
            float eps;
        };

        OpGroupNorm() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpGroupNorm")) {
                auto op_group_norm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpGroupNorm.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpGroupNorm")
                    .BindComputeShader(op_group_norm_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .BindStorageBuffer(2)
                    .BindStorageBuffer(3)
                    .PushConstant(0, sizeof(GroupNormParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& input, const Tensor& weight, const Tensor& bias, uint32_t padding, uint32_t stride) {
            return input.GetShape();
        }

        VkComputeTask operator()(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output, uint32_t group) {
            const uint32_t N = input.GetShape()[0];
            const uint32_t G = group;
            GroupNormParams params {
                .input_shape = { input.GetShape()[0], input.GetShape()[1], input.GetShape()[2], input.GetShape()[3] },
                .input_stride = { input.GetStride()[0], input.GetStride()[1], input.GetStride()[2], input.GetStride()[3] },
                .weight_shape = weight.GetShape()[0],
                .groups = G,
                .eps = 1e-5
            };
            VkComputeTask task("OpGroupNorm");
            task.BindBuffer(0, input.GetGPUHandle())
                .BindBuffer(1, weight.GetGPUHandle())
                .BindBuffer(2, bias.GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(GroupNormParams))
                .Dispatch(N, G, 1);
            return task;
        }
    };
}
