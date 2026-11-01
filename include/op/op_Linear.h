#pragma once

#include <base.h>
#include <vk/vk_framework.h>
#include <io/tensor.h>
#include <op/op_GEMM.h>

namespace vkdiffusion
{
    struct OpLinear {
        static void CreatePipeline(VkComputeFramework& vkcf) {
            OpGEMM::CreatePipeline(vkcf);
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& tensor, const Tensor& weight, const Tensor& bias) {
            return OpGEMM::GetResultShape(tensor, weight, bias, 1.0f, 0.0f, false, true);
        }

        static std::vector<uint32_t> GetResultShape(const Tensor& tensor, const Tensor& weight) {
            return OpGEMM::GetResultShape(tensor, weight, 1.0f, 0.0f, false, true);
        }

        VkComputeTask operator()(const Tensor& tensor, const Tensor& weight, const Tensor& bias, Tensor& output) const {
            OpGEMM gemm;
            return gemm(tensor, weight, bias, output, 1.0f, 0.0f, false, true);
        }

        VkComputeTask operator()(const Tensor& tensor, const Tensor& weight, Tensor& output) const {
            OpGEMM gemm;
            return gemm(tensor, weight, output, 1.0f, 0.0f, false, true);
        }
    };
}
