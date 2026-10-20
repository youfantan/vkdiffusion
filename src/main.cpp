#include <vk/vk_framework.h>
#include <io/model.h>
#include <platform/win32.h>
#include <../include/op/op_Conv2d.h>
#include <diffusion/unet.h>

using namespace vkdiffusion;

int main() {
    MicroTrace::InitializeProfile profile{
        .enable_console = true,
        .enable_console_ansi_out = true,
        .console_filter_level = MicroTrace::Level::DEBUG,
        .log_file_name = "vkdiffusion.log",
    };
    MicroTrace::Initialize(profile);
    VkComputeFramework vkcf;
    SafetensorsLoader loader(vkcf.GetAllocator());
    Safetensors sd_1_5 = loader.Load("models/v1-5-pruned-emaonly.safetensors");
    UNet<TensorDataType::F32> unet(vkcf, sd_1_5);
    auto latent = LoadTensorFromFile(vkcf.GetAllocator(), TensorDataType::F32, { 1, 4, 64, 64 }, "case0/latent.f32.bin", 0, "latent");
    auto text_embedding = LoadTensorFromFile(vkcf.GetAllocator(), TensorDataType::F32, { 1, 77, 768 }, "case0/context.f32.bin", 0, "text_embedding");
    auto timestep_str = ReadFileIntoString("case0/timestep.i64.bin").value();
    uint64_t timestep = *reinterpret_cast<uint64_t*>(&timestep_str[0]);
    auto& input_blocks_0_0_weight = sd_1_5.LoadTensor("model.diffusion_model.input_blocks.0.0.weight");
    auto& input_blocks_0_0_bias = sd_1_5.LoadTensor("model.diffusion_model.input_blocks.0.0.bias");
    auto& time_embed_0_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.weight");
    auto& time_embed_0_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.bias");
    auto& time_embed_2_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.weight");
    auto& time_embed_2_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.bias");
    HighPerformanceCounter counter;
    counter.Start();
    auto t_emb = unet.TimestepEmbedding(static_cast<float>(timestep), 1, 320);
    auto conv_in = unet.ConvLatent(latent);
    auto resnet_block = unet.ResNetBlock2D("model.diffusion_model.input_blocks.1.0", conv_in, t_emb, 320);
    counter.Stop();
    MLOG_INFO("ResNetBlock2D time resuming: {}", counter.GetDurationUs());
    TensorView v_view(resnet_block);
    v_view.Download();
    v_view.DumpToFile("resnet.f32.bin");
}
