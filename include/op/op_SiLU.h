#pragma once

#include <base.h>
#include <vk/vk_framework.h>
#include <io/tensor.h>

namespace vkdiffusion
{
    struct OpSiLU {
        struct SiLUParams {
            uint32_t size;
            uint32_t block;
            uint32_t dispatch_width;
        };
    public:
        OpSiLU() = default;

        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpSiLU")) {
                auto op_gemm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpSiLU.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpSiLU")
                     .BindComputeShader(op_gemm_shader)
                     .BindStorageBuffer(0)
                     .BindStorageBuffer(1)
                     .PushConstant(0, sizeof(SiLUParams))
                     .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& input) {
            return input.GetShape();
        }

        VkComputeTask operator()(const Tensor& input, Tensor& output) const {
#ifndef NDEBUG
            RUNTIME_ASSERT(input.GetSize() == output.GetSize());
#endif
            constexpr uint32_t SingleThreadProcessNum = 4;
            const uint32_t element_count = static_cast<uint32_t>(input.GetSize() / ParseTensorTypeSize(input.GetDataType()));
            uint32_t width = 1;
            while (static_cast<uint64_t>(width) * width * 256 * SingleThreadProcessNum < element_count) {
                ++width;
            }
            SiLUParams params {
                .size = element_count,
                .block = SingleThreadProcessNum,
                .dispatch_width = width
            };
            VkComputeTask task("OpSiLU");
            task.BindBuffer(0, input.GetGPUHandle())
                .BindBuffer(1, output.GetGPUHandle())
                .PushConstant(&params, sizeof(SiLUParams))
                .Dispatch(width, width, 1);
            return task;
        }
    };
}