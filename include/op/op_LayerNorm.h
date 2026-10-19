#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpLayerNorm {

        struct LayerNormParams {
            uint32_t input_shape[4];
            uint32_t input_stride[4];
            uint32_t weight_shape;
            float eps;
        };

        OpLayerNorm() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpLayerNorm")) {
                auto op_layer_norm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpLayerNorm.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpLayerNorm")
                    .BindComputeShader(op_layer_norm_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .BindStorageBuffer(2)
                    .BindStorageBuffer(3)
                    .PushConstant(0, sizeof(LayerNormParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& input, const Tensor& weight, const Tensor& bias) {
            return input.GetShape();
        }

        VkComputeTask operator()(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) const {
            const uint32_t N = input.GetShape()[0];
            const uint32_t L = input.GetShape()[1];
            const uint32_t C = input.GetShape()[2];
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetShape().size() == 3 && output.GetShape().size() == 3);
            RUNTIME_ASSERT(weight.GetShape().size() == 1 && bias.GetShape().size() == 1);
            RUNTIME_ASSERT(weight.GetShape()[0] == C && bias.GetShape()[0] == C);
            RUNTIME_ASSERT(output.GetShape() == input.GetShape());
#endif
            LayerNormParams params {
                .input_shape = { N, L, C, 0 },
                .input_stride = { input.GetStride()[0], input.GetStride()[1], input.GetStride()[2], 0 },
                .weight_shape = weight.GetShape()[0],
                .eps = 1e-5f
            };
            VkComputeTask task("OpLayerNorm");
            task.BindBuffer(0, input.GetGPUHandle())
                .BindBuffer(1, weight.GetGPUHandle())
                .BindBuffer(2, bias.GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(LayerNormParams))
                .Dispatch(N, L, 1);
            return task;
        }
    };
}
