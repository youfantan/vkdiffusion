#pragma once

#include <base.h>
#include <io/tensor.h>
#include <vk/vk_pipeline.h>
#include <vk/vk_framework.h>

namespace vkdiffusion
{
    struct OpAttention {

        struct AttentionParams {
            uint32_t query_shape[4];
            uint32_t query_stride[4];
            uint32_t key_shape[4];
            uint32_t key_stride[4];
            uint32_t heads;
            float scale;
        };

        OpAttention() = default;
        static void CreatePipeline(VkComputeFramework& vkcf) {
            if (!vkcf.GetPipeline("OpAttention")) {
                auto op_group_norm_shader = vkcf.GetShaderManager().CompileShader("shaders/OpAttention.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
                vkcf.CreatePipeline("OpAttention")
                    .BindComputeShader(op_group_norm_shader)
                    .BindStorageBuffer(0)
                    .BindStorageBuffer(1)
                    .BindStorageBuffer(2)
                    .BindStorageBuffer(3)
                    .PushConstant(0, sizeof(AttentionParams))
                    .Build();
            }
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& query, const Tensor& key, const Tensor& value, uint32_t heads, float scale) {
            return query.GetShape();
        }

        VkComputeTask operator()(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output, uint32_t heads, float scale) {
#ifndef NDEBUG
            RUNTIME_ASSERT(same(query.GetShape().size(), key.GetShape().size(), value.GetShape().size(), output.GetShape().size(), 3ull));
            RUNTIME_ASSERT(same(query.GetShape()[0], key.GetShape()[0], value.GetShape()[0], output.GetShape()[0]));
            RUNTIME_ASSERT(same(query.GetShape()[2], key.GetShape()[2], value.GetShape()[2], output.GetShape()[2]));
            RUNTIME_ASSERT(query.GetShape()[1] == output.GetShape()[1]);
            RUNTIME_ASSERT(key.GetShape()[1] == value.GetShape()[1]);
            RUNTIME_ASSERT(query.GetShape()[2] % heads == 0);
            RUNTIME_ASSERT(key.GetShape()[1] <= 4096);
#endif
            AttentionParams params {
                .query_shape = { query.GetShape()[0], query.GetShape()[1], query.GetShape()[2], 0 },
                .query_stride = { query.GetStride()[0], query.GetStride()[1], query.GetStride()[2], 0 },
                .key_shape = { key.GetShape()[0], key.GetShape()[1], key.GetShape()[2], 0 },
                .key_stride = { key.GetStride()[0], key.GetStride()[1], key.GetStride()[2], 0 },
                .heads = heads,
                .scale = scale
            };
            const uint32_t B = query.GetShape()[0];
            const uint32_t Lq = query.GetShape()[1];
            VkComputeTask task("OpAttention");
            task.BindBuffer(0, query.GetGPUHandle())
                .BindBuffer(1, key.GetGPUHandle())
                .BindBuffer(2, value.GetGPUHandle())
                .BindBuffer(3, output.GetGPUHandle())
                .PushConstant(&params, sizeof(AttentionParams))
                .Dispatch(B, Lq, heads);
            return task;
        }
    };
}
