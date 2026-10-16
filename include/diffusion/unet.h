#pragma once

#include <vk/vk_framework.h>
#include <io/tensor.h>
#include <io/model.h>
#include <op/op_SiLU.h>
#include <op/op_linear.h>


namespace vkdiffusion
{
    template<TensorDataType _DT>
    class UNet {
        VkComputeFramework& vkcf_;
        Safetensors& sd_1_5;

        OpSiLU SiLU;
        OpLinear Linear;
    public:
        UNet(VkComputeFramework& vkcf, Safetensors& sd_1_5) : vkcf_(vkcf), sd_1_5(sd_1_5) {
            OpSiLU::CreatePipeline(vkcf);
            OpLinear::CreatePipeline(vkcf);
        }

        Tensor TimestepEmbedding(TensorDataTypeTraits<_DT>::Type t, uint32_t batch, uint32_t dims) {
            Tensor embedding(vkcf_.GetAllocator(), TensorDataType::F32, { batch, dims }, "TimestepEmbedding");
            uint32_t half_dim = embedding.GetShape()[1] / 2;
            TensorView embedding_view(embedding);
            for (uint32_t i = 0; i < half_dim; ++i) {
                const double exponent = -std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half_dim);
                const double v = static_cast<double>(t) * std::exp(exponent);
                embedding_view.At<_DT>({ 0, i }) = static_cast<TensorDataTypeTraits<_DT>::Type>(std::cos(v));
                embedding_view.At<_DT>({ 0, half_dim + i }) = static_cast<TensorDataTypeTraits<_DT>::Type>(std::sin(v));
            }
            embedding_view.Upload();
            Tensor a(vkcf_.GetAllocator(), TensorDataType::F32, { 1, 1280 }, "a");
            Tensor b(vkcf_.GetAllocator(), TensorDataType::F32, { 1, 1280 }, "b");
            auto& time_embed_0_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.weight");
            auto& time_embed_0_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.bias");
            auto& time_embed_2_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.weight");
            auto& time_embed_2_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.bias");
            vkcf_.ExecuteTasks(
                    Linear(embedding, time_embed_0_weight, time_embed_0_bias, a),
                    SiLU(a, a),
                    Linear(a, time_embed_2_weight, time_embed_2_bias, b)
            );
            return b;
        }
    };
}

