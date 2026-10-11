#pragma once

#include <base.h>
#include <fstream>
#include <optional>

inline std::optional<size_t> ReadFileLength(const std::string& file_path) {
    std::ifstream input(file_path, std::ios::in | std::ios::binary);
    if (!input) return std::nullopt;
    size_t now = input.tellg();
    input.seekg(0, std::ios::end);
    size_t size = input.tellg();
    input.seekg(static_cast<std::istream::off_type>(now), std::ios::beg);
    return size;
}

inline std::optional<std::string> ReadFileIntoString(const std::string& file_path) {
    auto len = ReadFileLength(file_path);
    if (!len) return std::nullopt;
    std::ifstream input(file_path, std::ios::in | std::ios::binary);
    std::string str(len.value(), 0);
    input.read(&str[0], static_cast<std::streamsize>(len.value()));
    return str;
}
