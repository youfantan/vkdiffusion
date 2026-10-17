#include <vk/vk_framework.h>
#include <io/model.h>
#include <platform/win32.h>
#include <../include/op/op_Conv2d.h>
#include <diffusion/unet.h>

using namespace vkdiffusion;

int main() {
    VkComputeFramework vkcf;
    SafetensorsLoader loader(vkcf.GetAllocator());
    Safetensors sd_1_5 = loader.Load("models/v1-5-pruned-emaonly.safetensors");
    auto latent = LoadTensorFromFile(vkcf.GetAllocator(), TensorDataType::F32, { 1, 4, 64, 64 }, "case0/latent.f32.bin", 0, "Latent");
    auto text_embedding = LoadTensorFromFile(vkcf.GetAllocator(), TensorDataType::F32, { 1, 77, 768 }, "case0/context.f32.bin", 0, "TextEmbedding");
    auto timestep_str = ReadFileIntoString("case0/timestep.i64.bin").value();
    uint64_t timestep = *reinterpret_cast<uint64_t*>(&timestep_str[0]);
    auto& input_blocks_0_0_weight = sd_1_5.LoadTensor("model.diffusion_model.input_blocks.0.0.weight");
    auto& input_blocks_0_0_bias = sd_1_5.LoadTensor("model.diffusion_model.input_blocks.0.0.bias");
    auto& time_embed_0_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.weight");
    auto& time_embed_0_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.0.bias");
    auto& time_embed_2_weight = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.weight");
    auto& time_embed_2_bias = sd_1_5.LoadTensor("model.diffusion_model.time_embed.2.bias");
    Tensor conv_in(vkcf.GetAllocator(), TensorDataType::F32, { 1, 320, 64, 64 }, "ConvIn");
    UNet<TensorDataType::F32> unet(vkcf, sd_1_5);
    HighPerformanceCounter counter;
    counter.Start();
    auto v = unet.TimestepEmbedding(static_cast<float>(timestep), 1, 320);
    counter.Stop();
    std::cout << "TimeEmbedding time resuming: " << counter.GetDurationUs() << std::endl;
    TensorView v_view(v);
    v_view.Download();
    v_view.DumpToFile("proj.f32.bin");



}
