cmake_minimum_required(VERSION 4.2)
project(vkdiffusion)

set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(UNICODE)

include_directories(thirdparty/headers)
include_directories(include)

find_package(Vulkan REQUIRED)

message(STATUS "Vulkan include path: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")

add_executable(vkdiffusion
        src/main.cpp
        include/platform/win32.h
        include/vk/vk_framework.h
        include/utils.h
        include/io/safetensor.h
)

target_link_libraries(vkdiffusion PRIVATE Vulkan::Vulkan)

if (WIN32)
    set(VULKAN_LOADER_DLL "")
    foreach(_candidate "$ENV{VULKAN_SDK}/Bin/vulkan-1.dll" "$ENV{SystemRoot}/System32/vulkan-1.dll")
        if (EXISTS "${_candidate}")
            set(VULKAN_LOADER_DLL "${_candidate}")
            break()
        endif()
    endforeach()

    add_custom_command(TARGET vkdiffusion POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:vkdiffusion>
                    ${VULKAN_LOADER_DLL}
                    "$<TARGET_FILE_DIR:vkdiffusion>"
            COMMAND_EXPAND_LISTS
            COMMENT "Packaging Vulkan runtime DLLs"
    )
endif()
