#pragma once

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <vk/vk_framework.h>
#include <io/tensor.h>
#include <io/model.h>
#include <op/op_Add.h>
#include <op/op_BroadcastAdd.h>
#include <op/op_Conv2d.h>
#include <op/op_GroupNorm.h>
#include <op/op_Linear.h>
#include <op/op_SiLU.h>


namespace vkdiffusion
{
    template<TensorDataType _DT>
    class UNet {
        static constexpr uint32_t NumGroups = 32;

        VkComputeFramework& vkcf_;
        Safetensors& sd_1_5;

        OpAdd Add;
        OpBroadcastAdd BroadcastAdd;
        OpConv2d Conv2d;
        OpGroupNorm GroupNorm;
        OpLinear Linear;
        OpSiLU SiLU;

        Tensor MakeTensor(const std::vector<uint32_t>& shape, const std::string& name) {
            return Tensor(vkcf_.GetAllocator(), TensorDataType::F32, shape, name);
        }
    public:
        UNet(VkComputeFramework& vkcf, Safetensors& sd_1_5) : vkcf_(vkcf), sd_1_5(sd_1_5) {
            OpAdd::CreatePipeline(vkcf);
            OpBroadcastAdd::CreatePipeline(vkcf);
            OpConv2d::CreatePipeline(vkcf);
            OpGroupNorm::CreatePipeline(vkcf);
            OpLinear::CreatePipeline(vkcf);
            OpSiLU::CreatePipeline(vkcf);
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
            Tensor mid(vkcf_.GetAllocator(), TensorDataType::F32, { 1, 1280 }, "timestep_embedding_mid");
            Tensor t_emb(vkcf_.GetAllocator(), TensorDataType::F32, { 1, 1280 }, "t_emb");
            auto& time_embed_0_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.weight");
            auto& time_embed_0_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.bias");
            auto& time_embed_2_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.weight");
            auto& time_embed_2_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.bias");
            vkcf_.ExecuteTasks(
                    Linear(embedding, time_embed_0_weight, time_embed_0_bias, mid),
                    SiLU(mid, mid),
                    Linear(mid, time_embed_2_weight, time_embed_2_bias, t_emb)
            );
            return t_emb;
        }

        Tensor ConvLatent(const Tensor& latent) {
            Tensor conv_in(vkcf_.GetAllocator(), TensorDataType::F32, { 1, 320, 64, 64 }, "conv_in");
            auto& weight = sd_1_5.LoadTensor("model.diffusion_model.input_blocks.0.0.weight");
            auto& bias = sd_1_5.LoadTensor("model.diffusion_model.input_blocks.0.0.bias");
            vkcf_.WaitFor(vkcf_.ExecuteTasks(Conv2d(latent, weight, bias, conv_in, 1, 1)));
            return conv_in;
        }

        Tensor ResNetBlock2D(const std::string& prefix, const Tensor& input, const Tensor& temb, uint32_t out_channels) {
            const uint32_t batch = input.GetShape()[0];
            const uint32_t in_channels = input.GetShape()[1];
            const uint32_t height = input.GetShape()[2];
            const uint32_t width = input.GetShape()[3];

            auto& in_norm_weight = sd_1_5.LoadTensor(prefix + ".in_layers.0.weight");
            auto& in_norm_bias = sd_1_5.LoadTensor(prefix + ".in_layers.0.bias");
            auto& conv1_weight = sd_1_5.LoadTensor(prefix + ".in_layers.2.weight");
            auto& conv1_bias = sd_1_5.LoadTensor(prefix + ".in_layers.2.bias");
            auto& emb_weight = sd_1_5.LoadTensor(prefix + ".emb_layers.1.weight");
            auto& emb_bias = sd_1_5.LoadTensor(prefix + ".emb_layers.1.bias");
            auto& out_norm_weight = sd_1_5.LoadTensor(prefix + ".out_layers.0.weight");
            auto& out_norm_bias = sd_1_5.LoadTensor(prefix + ".out_layers.0.bias");
            auto& conv2_weight = sd_1_5.LoadTensor(prefix + ".out_layers.3.weight");
            auto& conv2_bias = sd_1_5.LoadTensor(prefix + ".out_layers.3.bias");

            Tensor norm1 = MakeTensor(input.GetShape(), "resnet_block_norm1");
            Tensor act1 = MakeTensor(input.GetShape(), "resnet_block_act1");
            Tensor conv1 = MakeTensor({ batch, out_channels, height, width }, "resnet_block_conv1");
            Tensor temb_act = MakeTensor(temb.GetShape(), "resnet_block_temb_act");
            Tensor temb_proj = MakeTensor({ temb.GetShape()[0], out_channels }, "resnet_block_temb_proj");
            Tensor with_time = MakeTensor(conv1.GetShape(), "resnet_block_with_time");
            Tensor norm2 = MakeTensor(conv1.GetShape(), "resnet_block_norm2");
            Tensor act2 = MakeTensor(conv1.GetShape(), "resnet_block_act2");
            Tensor conv2 = MakeTensor(conv1.GetShape(), "resnet_block_conv2");
            Tensor output = MakeTensor(conv1.GetShape(), "resnet_block");

            std::optional<Tensor> shortcut;
            if (in_channels != out_channels) {
                auto& skip_weight = sd_1_5.LoadTensor(prefix + ".skip_connection.weight");
                auto& skip_bias = sd_1_5.LoadTensor(prefix + ".skip_connection.bias");
                shortcut.emplace(MakeTensor(conv1.GetShape(), "resnet_block_shortcut"));
                vkcf_.ExecuteTasks(Conv2d(input, skip_weight, skip_bias, *shortcut, 0, 1));
            }

            const Tensor& residual = shortcut.has_value() ? *shortcut : input;

            vkcf_.ExecuteTasks(
                    GroupNorm(input, in_norm_weight, in_norm_bias, norm1, NumGroups),
                    SiLU(norm1, act1),
                    Conv2d(act1, conv1_weight, conv1_bias, conv1, 1, 1),
                    SiLU(temb, temb_act),
                    Linear(temb_act, emb_weight, emb_bias, temb_proj),
                    BroadcastAdd(conv1, temb_proj, with_time)
            );

            vkcf_.ExecuteTasks(
                    GroupNorm(with_time, out_norm_weight, out_norm_bias, norm2, NumGroups),
                    SiLU(norm2, act2),
                    Conv2d(act2, conv2_weight, conv2_bias, conv2, 1, 1),
                    Add(conv2, residual, output)
            );

            return output;
        }
    };
}
