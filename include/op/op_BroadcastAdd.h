#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpBroadcastAdd {

        struct AddParams {
            uint32_t output_shape[4];
            uint32_t tensor_a_stride[4];
            uint32_t tensor_b_stride[4];
            uint32_t output_stride[4];
        };

        OpBroadcastAdd() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpBroadcastAdd")) {
                auto op_broadcast_add_shader = vkcf.GetShaderManager().CompileShader("shaders/OpBroadcastAdd.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpBroadcastAdd")
                    .BindComputeShader(op_broadcast_add_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .BindStorageBuffer(2)
                    .PushConstant(0, sizeof(AddParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& a, const Tensor& b) {
            return a.GetShape();
        }

        VkComputeTask operator()(const Tensor& a, const Tensor& b, Tensor& output) const {
#ifndef NDEBUG
            RUNTIME_ASSERT(a.GetShape().size() == 4 && b.GetShape().size() == 2 && output.GetShape().size() == 4);
            RUNTIME_ASSERT(a.GetShape()[0] == b.GetShape()[0] && a.GetShape()[1] == b.GetShape()[1]);
            RUNTIME_ASSERT(a.GetShape()[2] == output.GetShape()[2] && a.GetShape()[3] == output.GetShape()[3]);
            RUNTIME_ASSERT(a.GetShape() == output.GetShape());
            RUNTIME_ASSERT(a.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(b.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(output.GetDataType() == TensorDataType::F32);
#endif
            AddParams params {
                .output_shape = { output.GetShape()[0], output.GetShape()[1], output.GetShape()[2], output.GetShape()[3] },
                .tensor_a_stride = { a.GetStride()[0], a.GetStride()[1], a.GetStride()[2], a.GetStride()[3] },
                .tensor_b_stride = { b.GetStride()[0], b.GetStride()[1], 0, 0 },
                .output_stride = { output.GetStride()[0], output.GetStride()[1], output.GetStride()[2], output.GetStride()[3] }
            };
            const uint32_t N = output.GetShape()[0];
            const uint32_t C = output.GetShape()[1];
            const uint32_t H = output.GetShape()[2];
            const uint32_t W = output.GetShape()[3];
            VkComputeTask task("OpBroadcastAdd");
            task.BindBuffer(0, a.GetGPUHandle())
                .BindBuffer(1, b.GetGPUHandle())
                .BindBuffer(2, output.GetGPUHandle())
                .PushConstant(&params, sizeof(AddParams))
                .Dispatch((W + 15) / 16, (H + 15) / 16, N * C);
            return task;
        }
    };
}
