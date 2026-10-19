#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpAdd {

        struct AddParams {
            uint32_t size;
        };

        OpAdd() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpAdd")) {
                auto op_add_shader = vkcf.GetShaderManager().CompileShader("shaders/OpAdd.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpAdd")
                    .BindComputeShader(op_add_shader)
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
            RUNTIME_ASSERT(a.GetShape() == b.GetShape());
            RUNTIME_ASSERT(a.GetShape() == output.GetShape());
            RUNTIME_ASSERT(a.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(b.GetDataType() == TensorDataType::F32);
            RUNTIME_ASSERT(output.GetDataType() == TensorDataType::F32);
#endif
            const uint32_t element_count = static_cast<uint32_t>(a.GetSize() / sizeof(float));
            AddParams params {
                .size = element_count
            };
            VkComputeTask task("OpAdd");
            task.BindBuffer(0, a.GetGPUHandle())
                .BindBuffer(1, b.GetGPUHandle())
                .BindBuffer(2, output.GetGPUHandle())
                .PushConstant(&params, sizeof(AddParams))
                .Dispatch((element_count + 255) / 256, 1, 1);
            return task;
        }
    };
}
