cmake_minimum_required(VERSION 4.2)
project(vkdiffusion)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.h HINTS "$ENV{VULKAN_SDK}/Include")
find_library(SHADERC_LIBRARY_RELEASE NAMES shaderc_combined  HINTS "$ENV{VULKAN_SDK}/Lib")
find_library(SHADERC_LIBRARY_DEBUG NAMES shaderc_combinedd HINTS "$ENV{VULKAN_SDK}/Lib")

message(STATUS "Vulkan include path: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")

add_executable(vkdiffusion
        src/main.cpp
        include/platform/win32.h
        include/utils.h
        include/io/model.h
        include/io/tensor.h

        src/vk/vk_commands.cpp
        src/vk/vk_gram.cpp
        src/vk/vk_shader.cpp
        src/vk/vk_pipeline.cpp
        src/vk/vk_init.cpp
        src/vk/vk_framework.cpp

        include/vk/vk_framework.h
        include/vk/vk_commands.h
        include/vk/vk_gram.h
        include/vk/vk_shader.h
        include/vk/vk_init.h
        include/vk/vk_pipeline.h
        include/vk/vk_sync.h
)

target_include_directories(vkdiffusion PUBLIC include)
target_include_directories(vkdiffusion PRIVATE thirdparty/headers)
target_include_directories(vkdiffusion PRIVATE ${SHADERC_INCLUDE_DIR})
target_link_libraries(vkdiffusion PRIVATE Vulkan::Vulkan
        optimized "${SHADERC_LIBRARY_RELEASE}"
        debug "${SHADERC_LIBRARY_DEBUG}"
)

target_compile_definitions(vkdiffusion PRIVATE UNICODE)

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

add_custom_command(TARGET vkdiffusion POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:vkdiffusion>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:vkdiffusion>/shaders"
        COMMENT "Copying shaders to runtime directory"
        VERBATIM
)