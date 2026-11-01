#pragma once

#include <base.h>
#include <vk/vk_framework.h>
#include <io/tensor.h>

namespace vkdiffusion
{
    struct OpGEMM {
        struct GEMMParams {
            uint32_t TensorAShape[4];
            uint32_t TensorAStride[4];
            uint32_t TensorBShape[2];
            uint32_t TensorBStride[2];
            uint32_t OutputShape[4];
            uint32_t OutputTensorStride[4];
            uint32_t tensor_a_rank;
            uint32_t output_rank;
            uint32_t batch;
            float alpha;
            float beta;
            uint32_t bias_tensor_shape;
            uint32_t trans_a;
            uint32_t trans_b;
        };

        static_assert(sizeof(GEMMParams) == 112);

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

        static std::vector<uint32_t> GetResultShape(const Tensor& tensor_a, const Tensor& tensor_b, float alpha, float beta, bool trans_a, bool trans_b) {
            uint32_t N = trans_b ? tensor_b.GetShape()[0] : tensor_b.GetShape()[1];
            if (tensor_a.GetShape().size() == 4) {
                return {
                    tensor_a.GetShape()[0],
                    tensor_a.GetShape()[2] * tensor_a.GetShape()[3],
                    N
                };
            }
            if (tensor_a.GetShape().size() == 3) {
                return { tensor_a.GetShape()[0], tensor_a.GetShape()[1], N };
            }
            uint32_t M = trans_a ? tensor_a.GetShape()[1] : tensor_a.GetShape()[0];
            return { M, N };
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& bias, float alpha, float beta, bool trans_a, bool trans_b) {
            return GetResultShape(tensor_a, tensor_b, alpha, beta, trans_a, trans_b);
        }

        VkComputeTask operator()(const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& bias, Tensor& output, float alpha, float beta, bool trans_a, bool trans_b) const {
            return CreateTask(tensor_a, tensor_b, &bias, output, alpha, beta, trans_a, trans_b);
        }

        VkComputeTask operator()(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& output, float alpha, float beta, bool trans_a, bool trans_b) const {
            return CreateTask(tensor_a, tensor_b, nullptr, output, alpha, beta, trans_a, trans_b);
        }

    private:
        VkComputeTask CreateTask(const Tensor& tensor_a, const Tensor& tensor_b, const Tensor* bias, Tensor& output, float alpha, float beta, bool trans_a, bool trans_b) const {
            const auto& a_shape_original = tensor_a.GetShape();
            const auto& b_shape = tensor_b.GetShape();
            const auto& output_shape_original = output.GetShape();
            const uint32_t a_rank = static_cast<uint32_t>(a_shape_original.size());
            const uint32_t output_rank = static_cast<uint32_t>(output_shape_original.size());
#ifndef NDEBUG
            RUNTIME_ASSERT((a_rank == 2 || a_rank == 3 || a_rank == 4) && b_shape.size() == 2);
            RUNTIME_ASSERT(bias == nullptr || bias->GetShape().size() == 1);
            RUNTIME_ASSERT(output_rank >= 2 && output_rank <= 4);
            if (a_rank == 2) {
                RUNTIME_ASSERT(output_rank == 2);
            } else {
                RUNTIME_ASSERT(!trans_a);
            }
#endif

            uint32_t M = 0;
            uint32_t K = 0;
            uint32_t N = trans_b ? b_shape[0] : b_shape[1];
            uint32_t batch_count = 1;
            uint32_t spatial_height = 0;
            uint32_t spatial_width = 0;

            if (a_rank == 2) {
                M = trans_a ? a_shape_original[1] : a_shape_original[0];
                K = trans_a ? a_shape_original[0] : a_shape_original[1];
            } else if (a_rank == 3) {
                batch_count = a_shape_original[0];
                M = a_shape_original[1];
                K = a_shape_original[2];
            } else {
                batch_count = a_shape_original[0];
                K = a_shape_original[1];
                spatial_height = a_shape_original[2];
                spatial_width = a_shape_original[3];
                M = spatial_height * spatial_width;
            }

#ifndef NDEBUG
            RUNTIME_ASSERT(K == (trans_b ? b_shape[1] : b_shape[0]));
            RUNTIME_ASSERT(bias == nullptr || bias->GetShape()[0] == N);
            if (a_rank == 2) {
                RUNTIME_ASSERT(output_rank == 2 && output_shape_original[0] == M && output_shape_original[1] == N);
            } else if (output_rank == 3) {
                RUNTIME_ASSERT(output_shape_original[0] == batch_count);
                RUNTIME_ASSERT(output_shape_original[1] == M);
                RUNTIME_ASSERT(output_shape_original[2] == N);
            } else if (output_rank == 4) {
                RUNTIME_ASSERT(output_shape_original[0] == batch_count);
                RUNTIME_ASSERT(output_shape_original[1] == N);
                if (spatial_height == 0) {
                    spatial_height = output_shape_original[2];
                    spatial_width = output_shape_original[3];
                    M = spatial_height * spatial_width;
                }
                RUNTIME_ASSERT(output_shape_original[2] == spatial_height);
                RUNTIME_ASSERT(output_shape_original[3] == spatial_width);
            } else {
                RUNTIME_ASSERT(false);
            }
#endif

            GEMMParams params {
                .TensorAShape = {
                    a_rank >= 1 ? tensor_a.GetShape()[0] : 0,
                    a_rank >= 2 ? tensor_a.GetShape()[1] : 0,
                    a_rank >= 3 ? tensor_a.GetShape()[2] : 0,
                    a_rank >= 4 ? tensor_a.GetShape()[3] : 0
                },
                .TensorAStride = {
                    a_rank >= 1 ? tensor_a.GetStride()[0] : 0,
                    a_rank >= 2 ? tensor_a.GetStride()[1] : 0,
                    a_rank >= 3 ? tensor_a.GetStride()[2] : 0,
                    a_rank >= 4 ? tensor_a.GetStride()[3] : 0
                },
                .TensorBShape = { b_shape[0], b_shape[1] },
                .TensorBStride = { tensor_b.GetStride()[0], tensor_b.GetStride()[1] },
                .OutputShape = {
                    output_rank >= 1 ? output_shape_original[0] : 0,
                    output_rank >= 2 ? output_shape_original[1] : 0,
                    output_rank >= 3 ? output_shape_original[2] : 0,
                    output_rank >= 4 ? output_shape_original[3] : 0
                },
                .OutputTensorStride = {
                    output_rank >= 1 ? output.GetStride()[0] : 0,
                    output_rank >= 2 ? output.GetStride()[1] : 0,
                    output_rank >= 3 ? output.GetStride()[2] : 0,
                    output_rank >= 4 ? output.GetStride()[3] : 0
                },
                .tensor_a_rank = a_rank,
                .output_rank = output_rank,
                .batch = batch_count,
                .alpha = alpha,
                .beta = beta,
                .bias_tensor_shape = bias == nullptr ? 0 : bias->GetShape()[0],
                .trans_a = static_cast<uint32_t>(trans_a),
                .trans_b = static_cast<uint32_t>(trans_b)
            };

            VkComputeTask task("OpGEMM");
            task.BindBuffer(0, tensor_a.GetGPUHandle())
                .BindBuffer(1, tensor_b.GetGPUHandle())
                .BindBuffer(2, bias == nullptr ? tensor_a.GetGPUHandle() : bias->GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(GEMMParams))
                .Dispatch((N + 15) / 16, (M + 15) / 16, batch_count);
            return task;
        }
    };
}
