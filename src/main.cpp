#include <vk/vk_framework.h>
#include <io/model.h>

using namespace vkdiffusion;

struct PushConstants {
    uint32_t width;
    uint32_t height;
};

int main() {
    constexpr uint32_t width = 1024;
    constexpr uint32_t height = 1024;
    constexpr uint32_t local_size = 16;
    constexpr size_t element_count = static_cast<size_t>(width) * height;
    std::vector<float> host_a(element_count);
    std::vector<float> host_b(element_count);
    std::vector<float> host_c(element_count, 0.0f);
    for (size_t i = 0; i < element_count; ++i) {
        host_a[i] = static_cast<float>(i % 97) * 0.5f;
        host_b[i] = static_cast<float>(i % 31) + 1.0f;
    }
    VkComputeFramework vkcf;
    auto comp_shader = vkcf.GetShaderManager().CompileShader("shaders/mul.comp", VkShaderManager::ShaderType::COMPUTE_SHADER);
    vkcf.CreatePipeline("mul")
        .BindComputeShader(comp_shader)
        .BindStorageBuffer(0)
        .BindStorageBuffer(1)
        .BindStorageBuffer(2)
        .PushConstant(0, sizeof(PushConstants))
        .Build();

    VkGRAMAllocator& allocator = vkcf.GetAllocator();
    Tensor tensor_a(allocator, TensorDataType::F32, { 1024, 1024 }, "TensorA");
    Tensor tensor_b(allocator, TensorDataType::F32, { 1024, 1024 }, "TensorB");
    Tensor tensor_c(allocator, TensorDataType::F32, { 1024, 1024 }, "TensorC");
    TensorView tensor_a_view(tensor_a);
    TensorView tensor_b_view(tensor_b);
    tensor_a_view.LoadFromMemory(host_a);
    tensor_a_view.Upload();
    tensor_b_view.LoadFromMemory(host_b);
    tensor_b_view.Upload();
    PushConstants push_constants { width, height };
    VkComputeTask task("mul");
    task.BindBuffer(0, tensor_a.GetGPUHandle())
        .BindBuffer(1, tensor_b.GetGPUHandle())
        .BindBuffer(2, tensor_c.GetGPUHandle())
        .PushConstant(&push_constants, sizeof(push_constants))
        .Dispatch(width / local_size, height / local_size);
    vkcf.ExecuteTask(task);
    TensorView tensor_c_view(tensor_c);
    tensor_c_view.Download();
    auto* proxy = &tensor_c_view.At<TensorDataType::F32>({ 0, 0 });
}
