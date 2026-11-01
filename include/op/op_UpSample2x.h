#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpUpSample2x {

        struct UpSample2xParams {
            uint32_t input_shape[4];
            uint32_t input_stride[4];
            uint32_t output_stride[4];
        };

        OpUpSample2x() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpUpSample2x")) {
                auto op_group_norm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpUpSample2x.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpUpSample2x")
                    .BindComputeShader(op_group_norm_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .PushConstant(0, sizeof(UpSample2xParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& input) {
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetShape().size() == 4);
#endif
            return { input.GetShape()[0], input.GetShape()[1], 2 * input.GetShape()[2], 2 * input.GetShape()[3] };
        }

        VkComputeTask operator()(const Tensor& input, Tensor& output) const {
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetShape().size() == 4 && output.GetShape().size() == 4);
            RUNTIME_ASSERT(input.GetShape()[0] == output.GetShape()[0]);
            RUNTIME_ASSERT(input.GetShape()[1] == output.GetShape()[1]);
            RUNTIME_ASSERT(2 * input.GetShape()[2] == output.GetShape()[2]);
            RUNTIME_ASSERT(2 * input.GetShape()[3] == output.GetShape()[3]);
            RUNTIME_ASSERT(input.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(output.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(input.GetShape()[2] > 0 && input.GetShape()[3] > 0);
            RUNTIME_ASSERT(output.GetSize() / sizeof(float) <= UINT32_MAX);
#endif
            const uint32_t N = input.GetShape()[0];
            const uint32_t C = input.GetShape()[1];
            UpSample2xParams params {
                .input_shape = { input.GetShape()[0], input.GetShape()[1], input.GetShape()[2], input.GetShape()[3] },
                .input_stride = { input.GetStride()[0], input.GetStride()[1], input.GetStride()[2], input.GetStride()[3] },
                .output_stride =  { output.GetStride()[0], output.GetStride()[1], output.GetStride()[2], output.GetStride()[3] }
            };
            VkComputeTask task("OpUpSample2x");
            task.BindBuffer(0, input.GetGPUHandle())
                .BindBuffer(1, output.GetGPUHandle())
                .PushConstant(&params, sizeof(UpSample2xParams))
                .Dispatch(N, C, 1);
            return task;
        }
    };
}
