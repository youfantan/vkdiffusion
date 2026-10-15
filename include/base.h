#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <format>
#include <string>
#include <optional>
#include <unordered_map>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <functional>

#ifdef WIN32

#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX

#include <Windows.h>

#endif

#define VKCHECK(expr) _VkCheck(#expr, (expr), __FILE__, __LINE__)

#define FORI(number) for (uint64_t i = 0; i < number; ++i)
#define FORX(number, x) for (uint64_t x = 0; x < number; ++x)

#define NO_COPY_MOVE(cls) \
    cls(cls&) = delete; \
    cls(cls&&) = delete; \
    cls& operator=(cls&) = delete; \
    cls& operator=(cls&&) = delete;

#define NO_COPY(cls) \
    cls(cls&) = delete; \
    cls& operator=(cls&) = delete;

inline VkResult _VkCheck(const char* expr, VkResult result, const char* file_name, int line) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::format("Vulkan calling error: [ {} ] returned {} at {}:{}", expr, static_cast<int>(result), file_name, line));
    }
    return result;
}

namespace vkdiffusion
{
    class VkInitializer;
    class VkCommandDispatcher;
    class VkGRAMBuffer;
    class VkGRAMAllocator;
    class VkShaderManager;
}
