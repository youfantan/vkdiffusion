#pragma once

#include <base.h>
#include <vk/vk_framework.h>
#include <io/tensor.h>

namespace vkdiffusion
{
    struct OpGEMM {
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

        OpGEMM() = default;
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

        static std::vector<uint32_t> GetResultShape(const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& bias, float alpha, float beta, bool trans_a, bool trans_b) {
            uint32_t M = trans_a ? tensor_a.GetShape()[1] : tensor_a.GetShape()[0];
            uint32_t N = trans_b ? tensor_b.GetShape()[0] : tensor_b.GetShape()[1];
            return { M, N };
        }

        std::vector<VkComputeTask> operator()(const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& bias, Tensor& output, float alpha, float beta, bool trans_a, bool trans_b) const {
            uint32_t M = trans_a ? tensor_a.GetShape()[1] : tensor_a.GetShape()[0];
            uint32_t K = trans_a ? tensor_a.GetShape()[0] : tensor_a.GetShape()[1];
            uint32_t N = trans_b ? tensor_b.GetShape()[0] : tensor_b.GetShape()[1];
#ifndef NDEBUG
            RUNTIME_ASSERT(tensor_a.GetShape().size() == 2 && tensor_b.GetShape().size() == 2 && bias.GetShape().size() == 1 && output.GetShape().size() == 2);
            RUNTIME_ASSERT((trans_a ? tensor_a.GetShape()[0] : tensor_a.GetShape()[1]) == (trans_b ? tensor_b.GetShape()[1] : tensor_b.GetShape()[0]));
            RUNTIME_ASSERT(output.GetShape()[0] == M && output.GetShape()[1] == N);
#endif
            GEMMParams params {
                .TensorAShape = { tensor_a.GetShape()[0], tensor_a.GetShape()[1] },
                .TensorAStride = { tensor_a.GetStride()[0], tensor_a.GetStride()[1] },
                .TensorBShape = { tensor_b.GetShape()[0], tensor_b.GetShape()[1] },
                .TensorBStride = { tensor_b.GetStride()[0], tensor_b.GetStride()[1] },
                .OutputShape = { output.GetShape()[0], output.GetShape()[1] },
                .OutputTensorStride = { output.GetStride()[0], output.GetStride()[1] },
                .alpha = alpha,
                .beta = beta,
                .bias_tensor_shape = bias.GetShape()[0],
                .trans_a = trans_a,
                .trans_b = trans_b
            };

            std::vector<VkComputeTask> tasks;
            tasks.emplace_back("OpGEMM");
            tasks.back().BindBuffer(0, tensor_a.GetGPUHandle())
                .BindBuffer(1, tensor_b.GetGPUHandle())
                .BindBuffer(2, bias.GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(GEMMParams))
                .Dispatch((N + 15) / 16, (M + 15) / 16, 1);
            return tasks;
        }
    };
}
