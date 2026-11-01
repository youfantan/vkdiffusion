#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpConcat {

        struct ConcatParams {
            uint32_t a_shape[4];
            uint32_t a_stride[4];
            uint32_t b_shape[4];
            uint32_t b_stride[4];
            uint32_t output_stride[4];
        };

        OpConcat() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpConcat")) {
                auto op_concat_shader = vkcf.GetShaderManager().CompileShader("shaders/OpConcat.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpConcat")
                    .BindComputeShader(op_concat_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .BindStorageBuffer(2)
                    .PushConstant(0, sizeof(ConcatParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& a, const Tensor& b) {
#ifndef NDEBUG
            RUNTIME_ASSERT(a.GetShape().size() == 4 && b.GetShape().size() == 4);
            RUNTIME_ASSERT(a.GetShape()[0] == b.GetShape()[0]);
            RUNTIME_ASSERT(a.GetShape()[2] == b.GetShape()[2]);
            RUNTIME_ASSERT(a.GetShape()[3] == b.GetShape()[3]);
#endif
            return {
                a.GetShape()[0],
                a.GetShape()[1] + b.GetShape()[1],
                a.GetShape()[2],
                a.GetShape()[3]
            };
        }

        VkComputeTask operator()(const Tensor& a, const Tensor& b, Tensor& output) const {
#ifndef NDEBUG
            RUNTIME_ASSERT(a.GetShape().size() == 4 && b.GetShape().size() == 4 && output.GetShape().size() == 4);
            RUNTIME_ASSERT(a.GetShape()[0] == b.GetShape()[0] && a.GetShape()[0] == output.GetShape()[0]);
            RUNTIME_ASSERT(a.GetShape()[2] == b.GetShape()[2] && a.GetShape()[2] == output.GetShape()[2]);
            RUNTIME_ASSERT(a.GetShape()[3] == b.GetShape()[3] && a.GetShape()[3] == output.GetShape()[3]);
            RUNTIME_ASSERT(a.GetShape()[1] + b.GetShape()[1] == output.GetShape()[1]);
            RUNTIME_ASSERT(
                a.GetDataType() == TensorDataType::F32 &&
                b.GetDataType() == TensorDataType::F32 &&
                output.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(output.GetSize() / sizeof(float) <= UINT32_MAX);
#endif
            const uint32_t N = output.GetShape()[0];
            const uint32_t C = output.GetShape()[1];
            const uint32_t H = output.GetShape()[2];
            const uint32_t W = output.GetShape()[3];
            ConcatParams params {
                .a_shape = { a.GetShape()[0], a.GetShape()[1], a.GetShape()[2], a.GetShape()[3] },
                .a_stride = { a.GetStride()[0], a.GetStride()[1], a.GetStride()[2], a.GetStride()[3] },
                .b_shape = { b.GetShape()[0], b.GetShape()[1], b.GetShape()[2], b.GetShape()[3] },
                .b_stride = { b.GetStride()[0], b.GetStride()[1], b.GetStride()[2], b.GetStride()[3] },
                .output_stride = { output.GetStride()[0], output.GetStride()[1], output.GetStride()[2], output.GetStride()[3] }
            };
            VkComputeTask task("OpConcat");
            task.BindBuffer(0, a.GetGPUHandle())
                .BindBuffer(1, b.GetGPUHandle())
                .BindBuffer(2, output.GetGPUHandle())
                .PushConstant(&params, sizeof(ConcatParams))
                .Dispatch((W + 15) / 16, (H + 15) / 16, N * C);
            return task;
        }
    };
}
