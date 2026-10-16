#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpConv2d {

        struct Conv2dParams {
            uint32_t InputTensorShape[4];
            uint32_t InputTensorStride[4];
            uint32_t KernelTensorShape[4];
            uint32_t KernelTensorStride[4];
            uint32_t OutputTensorShape[4];
            uint32_t OutputTensorStride[4];
            uint32_t bias_tensor_shape;
            uint32_t padding;
            uint32_t stride;
            uint32_t _padding;
        };

        OpConv2d() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpConv2d")) {
                auto op_conv2d_shader = vkcf.GetShaderManager().CompileShader("shaders/OpConv2d.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpConv2d")
                    .BindComputeShader(op_conv2d_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .BindStorageBuffer(2)
                    .BindStorageBuffer(3)
                    .PushConstant(0, sizeof(Conv2dParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& input, const Tensor& kernel, const Tensor& bias, uint32_t padding, uint32_t stride) {
            uint32_t out_n = input.GetShape()[0];
            uint32_t out_c = kernel.GetShape()[0];
            uint32_t out_h = (input.GetShape()[2] + 2 * padding - kernel.GetShape()[2]) / stride + 1;
            uint32_t out_w = (input.GetShape()[3] + 2 * padding - kernel.GetShape()[3]) / stride + 1;
            return { out_n, out_c, out_h, out_w };
        }

        std::vector<VkComputeTask> operator()(const Tensor& input, const Tensor& kernel, const Tensor& bias, Tensor& output, uint32_t padding, uint32_t stride) {
            uint32_t out_n = input.GetShape()[0];
            uint32_t out_c = kernel.GetShape()[0];
            uint32_t out_h = (input.GetShape()[2] + 2 * padding - kernel.GetShape()[2]) / stride + 1;
            uint32_t out_w = (input.GetShape()[3] + 2 * padding - kernel.GetShape()[3]) / stride + 1;
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetShape().size() == 4 && kernel.GetShape().size() == 4 && output.GetShape().size() == 4 && bias.GetShape().size() == 1);
            RUNTIME_ASSERT(input.GetShape()[1] == kernel.GetShape()[1]);
            RUNTIME_ASSERT(out_n == output.GetShape()[0] && out_c == output.GetShape()[1] && out_h == output.GetShape()[2] && out_w == output.GetShape()[3]);
#endif
            Conv2dParams params {
                .InputTensorShape = { input.GetShape()[0], input.GetShape()[1], input.GetShape()[2], input.GetShape()[3] },
                .InputTensorStride = { input.GetStride()[0], input.GetStride()[1], input.GetStride()[2], input.GetStride()[3] },
                .KernelTensorShape = { kernel.GetShape()[0], kernel.GetShape()[1], kernel.GetShape()[2], kernel.GetShape()[3] },
                .KernelTensorStride = { kernel.GetStride()[0], kernel.GetStride()[1], kernel.GetStride()[2], kernel.GetStride()[3] } ,
                .OutputTensorShape = { output.GetShape()[0], output.GetShape()[1], output.GetShape()[2], output.GetShape()[3] } ,
                .OutputTensorStride = { output.GetStride()[0], output.GetStride()[1], output.GetStride()[2], output.GetStride()[3] },
                .bias_tensor_shape = bias.GetShape()[0],
                .padding = padding,
                .stride = stride,
                ._padding = 0
            };
            std::vector<VkComputeTask> tasks;
            tasks.emplace_back("OpConv2d");
            tasks.back().BindBuffer(0, input.GetGPUHandle())
                .BindBuffer(1, kernel.GetGPUHandle())
                .BindBuffer(2, bias.GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(Conv2dParams))
                .Dispatch((out_w + 15) / 16, (out_h + 15) / 16, out_n * out_c);
            return tasks;
        }
    };

}
