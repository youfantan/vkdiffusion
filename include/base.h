#pragma once

#include <iostream>

#ifdef WIN32

#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX

#include <Windows.h>
#include <vulkan/vulkan.h>
#include <cstdint>

#endif

#define VKCHECK(expr) _VkCheck(#expr, (expr), __FILE__, __LINE__)

#define FORI(number) for (uint64_t i = 0; i < number; ++i)
#define FORX(number, x) for (uint64_t x = 0; x < number; ++x)

inline VkResult _VkCheck(const char* expr, VkResult result, const char* file_name, int line) {
    if (result != VK_SUCCESS) {
        std::cout << "Vulkan calling error: [" << expr << "] returned " << result << " at " << file_name << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}