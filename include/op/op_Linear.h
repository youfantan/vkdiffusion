#pragma once

#include <base.h>
#include <vk/vk_framework.h>
#include <io/tensor.h>

namespace vkdiffusion
{
    struct OpLinear {
        struct GEMMParams {
            uint32_t TensorAShape[2];
            uint32_t TensorAStride[2];
            uint32_t TensorBShape[2];
            uint32_t TensorBStride[2];
            uint32_t OutputShape[2];
            uint32_t OutputTensorStride[2];
            float alpha;
            float beta;
            uint32_t bias_tensor_shape;
            uint32_t trans_a;
            uint32_t trans_b;
        };

        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpGEMM")) {
                auto op_gemm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpGEMM.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpGEMM")
                     .BindComputeShader(op_gemm_shader)
                     .BindStorageBuffer(0)
                     .BindStorageBuffer(1)
                     .BindStorageBuffer(2)
                     .BindStorageBuffer(3)
                     .PushConstant(0, sizeof(GEMMParams))
                     .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& tensor, const Tensor& weight, const Tensor& bias) {
            uint32_t M = tensor.GetShape()[0];
            uint32_t N = weight.GetShape()[0];
            return { M, N };
        }

        VkComputeTask operator()(const Tensor& tensor, const Tensor& weight, const Tensor& bias, Tensor& output) const {
            uint32_t M = tensor.GetShape()[0];
            uint32_t K = tensor.GetShape()[1];
            uint32_t N = weight.GetShape()[0];
#ifndef NDEBUG
            RUNTIME_ASSERT(tensor.GetShape().size() == 2 && weight.GetShape().size() == 2 && bias.GetShape().size() == 1 && output.GetShape().size() == 2);
            RUNTIME_ASSERT(tensor.GetShape()[1] == weight.GetShape()[1]);
            RUNTIME_ASSERT(output.GetShape()[0] == M && output.GetShape()[1] == N);
#endif
            GEMMParams params {
                .TensorAShape = { tensor.GetShape()[0], tensor.GetShape()[1] },
                .TensorAStride = { tensor.GetStride()[0], tensor.GetStride()[1] },
                .TensorBShape = { weight.GetShape()[0], weight.GetShape()[1] },
                .TensorBStride = { weight.GetStride()[0], weight.GetStride()[1] },
                .OutputShape = { output.GetShape()[0], output.GetShape()[1] },
                .OutputTensorStride = { output.GetStride()[0], output.GetStride()[1] },
                .alpha = 1.0f,
                .beta = 0.0f,
                .bias_tensor_shape = bias.GetShape()[0],
                .trans_a = false,
                .trans_b = true
            };
            VkComputeTask task("OpGEMM");
            task.BindBuffer(0, tensor.GetGPUHandle())
                .BindBuffer(1, weight.GetGPUHandle())
                .BindBuffer(2, bias.GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(GEMMParams))
                .Dispatch((N + 15) / 16, (M + 15) / 16, 1);
            return task;
        }
    };
}
