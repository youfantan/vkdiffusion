#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <dbghelp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>

#ifdef _MSC_VER
#pragma comment(lib, "dbghelp.lib")
#endif

namespace MicroTrace
{
    enum class Level {
        TRACE,
        DEBUG,
        INFO,
        WARN,
        ERR,
        FATAL
    };

    struct InitializeProfile {
        bool enable_console;
        bool enable_console_ansi_out;
        Level console_filter_level;
        const char* log_file_name;
    };

    struct Instance {
        HANDLE h_stdout = INVALID_HANDLE_VALUE;
        HANDLE h_file = INVALID_HANDLE_VALUE;
    };

    enum ANSIColor {
        BLACK,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE,
        RESET
    };

    template<ANSIColor _Background, ANSIColor _Foreground>
    struct ANSITraits {
        constexpr static std::string_view Prefix = "\033[0m";
    };

#define MICROTRACE_ANSI_TRAITS(background, foreground, prefix) \
    template<> struct ANSITraits<background, foreground> { \
        constexpr static std::string_view Prefix = prefix; \
    }

    MICROTRACE_ANSI_TRAITS(BLACK, BLACK, "\033[40;30m");
    MICROTRACE_ANSI_TRAITS(BLACK, RED, "\033[40;31m");
    MICROTRACE_ANSI_TRAITS(BLACK, GREEN, "\033[40;32m");
    MICROTRACE_ANSI_TRAITS(BLACK, YELLOW, "\033[40;33m");
    MICROTRACE_ANSI_TRAITS(BLACK, BLUE, "\033[40;34m");
    MICROTRACE_ANSI_TRAITS(BLACK, MAGENTA, "\033[40;35m");
    MICROTRACE_ANSI_TRAITS(BLACK, CYAN, "\033[40;36m");
    MICROTRACE_ANSI_TRAITS(BLACK, WHITE, "\033[40;37m");

    MICROTRACE_ANSI_TRAITS(RED, BLACK, "\033[41;30m");
    MICROTRACE_ANSI_TRAITS(RED, RED, "\033[41;31m");
    MICROTRACE_ANSI_TRAITS(RED, GREEN, "\033[41;32m");
    MICROTRACE_ANSI_TRAITS(RED, YELLOW, "\033[41;33m");
    MICROTRACE_ANSI_TRAITS(RED, BLUE, "\033[41;34m");
    MICROTRACE_ANSI_TRAITS(RED, MAGENTA, "\033[41;35m");
    MICROTRACE_ANSI_TRAITS(RED, CYAN, "\033[41;36m");
    MICROTRACE_ANSI_TRAITS(RED, WHITE, "\033[41;37m");

    MICROTRACE_ANSI_TRAITS(GREEN, BLACK, "\033[42;30m");
    MICROTRACE_ANSI_TRAITS(GREEN, RED, "\033[42;31m");
    MICROTRACE_ANSI_TRAITS(GREEN, GREEN, "\033[42;32m");
    MICROTRACE_ANSI_TRAITS(GREEN, YELLOW, "\033[42;33m");
    MICROTRACE_ANSI_TRAITS(GREEN, BLUE, "\033[42;34m");
    MICROTRACE_ANSI_TRAITS(GREEN, MAGENTA, "\033[42;35m");
    MICROTRACE_ANSI_TRAITS(GREEN, CYAN, "\033[42;36m");
    MICROTRACE_ANSI_TRAITS(GREEN, WHITE, "\033[42;37m");

    MICROTRACE_ANSI_TRAITS(YELLOW, BLACK, "\033[43;30m");
    MICROTRACE_ANSI_TRAITS(YELLOW, RED, "\033[43;31m");
    MICROTRACE_ANSI_TRAITS(YELLOW, GREEN, "\033[43;32m");
    MICROTRACE_ANSI_TRAITS(YELLOW, YELLOW, "\033[43;33m");
    MICROTRACE_ANSI_TRAITS(YELLOW, BLUE, "\033[43;34m");
    MICROTRACE_ANSI_TRAITS(YELLOW, MAGENTA, "\033[43;35m");
    MICROTRACE_ANSI_TRAITS(YELLOW, CYAN, "\033[43;36m");
    MICROTRACE_ANSI_TRAITS(YELLOW, WHITE, "\033[43;37m");

    MICROTRACE_ANSI_TRAITS(BLUE, BLACK, "\033[44;30m");
    MICROTRACE_ANSI_TRAITS(BLUE, RED, "\033[44;31m");
    MICROTRACE_ANSI_TRAITS(BLUE, GREEN, "\033[44;32m");
    MICROTRACE_ANSI_TRAITS(BLUE, YELLOW, "\033[44;33m");
    MICROTRACE_ANSI_TRAITS(BLUE, BLUE, "\033[44;34m");
    MICROTRACE_ANSI_TRAITS(BLUE, MAGENTA, "\033[44;35m");
    MICROTRACE_ANSI_TRAITS(BLUE, CYAN, "\033[44;36m");
    MICROTRACE_ANSI_TRAITS(BLUE, WHITE, "\033[44;37m");

    MICROTRACE_ANSI_TRAITS(MAGENTA, BLACK, "\033[45;30m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, RED, "\033[45;31m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, GREEN, "\033[45;32m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, YELLOW, "\033[45;33m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, BLUE, "\033[45;34m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, MAGENTA, "\033[45;35m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, CYAN, "\033[45;36m");
    MICROTRACE_ANSI_TRAITS(MAGENTA, WHITE, "\033[45;37m");

    MICROTRACE_ANSI_TRAITS(CYAN, BLACK, "\033[46;30m");
    MICROTRACE_ANSI_TRAITS(CYAN, RED, "\033[46;31m");
    MICROTRACE_ANSI_TRAITS(CYAN, GREEN, "\033[46;32m");
    MICROTRACE_ANSI_TRAITS(CYAN, YELLOW, "\033[46;33m");
    MICROTRACE_ANSI_TRAITS(CYAN, BLUE, "\033[46;34m");
    MICROTRACE_ANSI_TRAITS(CYAN, MAGENTA, "\033[46;35m");
    MICROTRACE_ANSI_TRAITS(CYAN, CYAN, "\033[46;36m");
    MICROTRACE_ANSI_TRAITS(CYAN, WHITE, "\033[46;37m");

    MICROTRACE_ANSI_TRAITS(WHITE, BLACK, "\033[47;30m");
    MICROTRACE_ANSI_TRAITS(WHITE, RED, "\033[47;31m");
    MICROTRACE_ANSI_TRAITS(WHITE, GREEN, "\033[47;32m");
    MICROTRACE_ANSI_TRAITS(WHITE, YELLOW, "\033[47;33m");
    MICROTRACE_ANSI_TRAITS(WHITE, BLUE, "\033[47;34m");
    MICROTRACE_ANSI_TRAITS(WHITE, MAGENTA, "\033[47;35m");
    MICROTRACE_ANSI_TRAITS(WHITE, CYAN, "\033[47;36m");
    MICROTRACE_ANSI_TRAITS(WHITE, WHITE, "\033[47;37m");

    inline InitializeProfile g_profile{};
    inline Instance g_instance{};
    inline bool g_ansi_out = false;
    inline std::mutex g_log_mutex;

    inline bool IsValidHandle(HANDLE handle) {
        return handle != nullptr && handle != INVALID_HANDLE_VALUE;
    }

    inline bool IsGuiSubsystem() {
        const auto* base = reinterpret_cast<const BYTE*>(GetModuleHandleW(nullptr));
        if (base == nullptr) {
            return false;
        }

        const auto* dos_header = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        if (dos_header->e_magic != IMAGE_DOS_SIGNATURE) {
            return false;
        }

        const auto* nt_headers = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos_header->e_lfanew);
        if (nt_headers->Signature != IMAGE_NT_SIGNATURE) {
            return false;
        }

        return nt_headers->OptionalHeader.Subsystem == IMAGE_SUBSYSTEM_WINDOWS_GUI;
    }

    inline void WriteToHandle(HANDLE handle, std::string_view text) {
        if (!IsValidHandle(handle) || text.empty()) {
            return;
        }

        DWORD written = 0;
        if (GetFileType(handle) == FILE_TYPE_CHAR) {
            WriteConsoleA(handle, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
            return;
        }

        size_t offset = 0;
        while (offset < text.size()) {
            const size_t remaining = text.size() - offset;
            const DWORD chunk = remaining > MAXDWORD ? MAXDWORD : static_cast<DWORD>(remaining);
            if (!WriteFile(handle, text.data() + offset, chunk, &written, nullptr) || written == 0) {
                break;
            }
            offset += written;
        }
    }

    inline void BindStandardStreams() {
        FILE* stream = nullptr;
        freopen_s(&stream, "CONOUT$", "w", stdout);
        freopen_s(&stream, "CONOUT$", "w", stderr);
        freopen_s(&stream, "CONIN$", "r", stdin);
    }

    inline void Initialize(const InitializeProfile& profile) {
        std::lock_guard<std::mutex> lock(g_log_mutex);

        g_profile = profile;
        g_ansi_out = false;

        if (IsValidHandle(g_instance.h_file)) {
            CloseHandle(g_instance.h_file);
        }
        g_instance.h_file = INVALID_HANDLE_VALUE;
        g_instance.h_stdout = INVALID_HANDLE_VALUE;

        if (profile.enable_console) {
            HANDLE stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
            const bool is_pipe = IsValidHandle(stdout_handle) && GetFileType(stdout_handle) == FILE_TYPE_PIPE;

            if (!is_pipe) {
                const bool has_stdout = IsValidHandle(stdout_handle) && GetFileType(stdout_handle) != FILE_TYPE_UNKNOWN;
                if (!has_stdout && !GetConsoleWindow() && IsGuiSubsystem()) {
                    if (AllocConsole()) {
                        BindStandardStreams();
                    }
                }
                if (!IsValidHandle(stdout_handle)) {
                    stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
                }
            }

            g_instance.h_stdout = stdout_handle;

            if (profile.enable_console_ansi_out && IsValidHandle(stdout_handle)) {
                const DWORD type = GetFileType(stdout_handle);
                if (type == FILE_TYPE_CHAR) {
                    DWORD mode = 0;
                    if (GetConsoleMode(stdout_handle, &mode)) {
                        SetConsoleOutputCP(CP_UTF8);
                        if (SetConsoleMode(stdout_handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                            g_ansi_out = true;
                        }
                    }
                } else if (type == FILE_TYPE_PIPE) {
                    g_ansi_out = true;
                }
            }
        }

        if (profile.log_file_name != nullptr && profile.log_file_name[0] != '\0') {
            g_instance.h_file = CreateFileA(
                profile.log_file_name,
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr,
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);
            if (!IsValidHandle(g_instance.h_file)) {
                g_instance.h_file = INVALID_HANDLE_VALUE;
            }
        }
    }

    inline std::string GetStackTrace() {
        std::string result;
        constexpr USHORT max_frames = 64;
        void* frames[max_frames] = {};
        const USHORT frame_count = CaptureStackBackTrace(1, max_frames, frames, nullptr);
        const HANDLE process = GetCurrentProcess();
        static std::once_flag symbol_once;
        std::call_once(symbol_once, [] {
            SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        });
        constexpr DWORD max_name_length = 256;
        alignas(SYMBOL_INFO) char symbol_buffer[sizeof(SYMBOL_INFO) + max_name_length * sizeof(char)] = {};
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = max_name_length;
        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        for (USHORT i = 0; i < frame_count; ++i) {
            const DWORD64 address = reinterpret_cast<DWORD64>(frames[i]);
            DWORD64 displacement = 0;
            DWORD line_displacement = 0;
            result += std::format("  #{} 0x{:016X}", i, address);
            if (SymFromAddr(process, address, &displacement, symbol)) {
                result += std::format(" {}+0x{:X}", symbol->Name, displacement);
            }
            result += '\n';
        }
        return result;
    }

    inline void FileOut(const std::string& log_str) {
        WriteToHandle(g_instance.h_file, log_str);
    }

    template<ANSIColor _Background, ANSIColor _Foreground>
    inline void ConsoleOut(const std::string& log_str) {
        if (!IsValidHandle(g_instance.h_stdout)) {
            return;
        }
        if (g_ansi_out) {
            WriteToHandle(g_instance.h_stdout, std::format(
                "{}{}{}",
                ANSITraits<_Background, _Foreground>::Prefix,
                log_str,
                ANSITraits<RESET, RESET>::Prefix));
        } else {
            WriteToHandle(g_instance.h_stdout, log_str);
        }
    }

    template<Level Lv>
    constexpr std::string_view LevelName() {
        if constexpr (Lv == Level::TRACE) {
            return "TRACE";
        } else if constexpr (Lv == Level::DEBUG) {
            return "DEBUG";
        } else if constexpr (Lv == Level::INFO) {
            return "INFO";
        } else if constexpr (Lv == Level::WARN) {
            return "WARN";
        } else if constexpr (Lv == Level::ERR) {
            return "ERR";
        } else {
            return "FATAL";
        }
    }

    inline std::string FormatTime(const time_t& tp) {
        std::tm local_time{};
        localtime_s(&local_time, &tp);
        char buffer[32] = {};
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time);
        return buffer;
    }

    template<Level Lv, typename... Args>
    inline void LogOut(const std::source_location& loc, const time_t& tp, const std::string& fmt_string, Args&&... args) {
        std::string message = std::vformat(fmt_string, std::make_format_args(args...));

        const std::string log_line = std::format(
            "[{}][{}][{}:{}]: {}\n",
            FormatTime(tp),
            LevelName<Lv>(),
            loc.file_name(),
            loc.line(),
            message);

        std::string stack_trace;
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            const int level = static_cast<int>(Lv);
            const bool console_enabled =
                g_profile.enable_console &&
                level >= static_cast<int>(g_profile.console_filter_level);
            if (console_enabled) {
                if constexpr (Lv == Level::TRACE) {
                    ConsoleOut<BLACK, GREEN>(log_line);
                } else if constexpr (Lv == Level::DEBUG) {
                    ConsoleOut<BLACK, CYAN>(log_line);
                } else if constexpr (Lv == Level::INFO) {
                    ConsoleOut<BLACK, WHITE>(log_line);
                } else if constexpr (Lv == Level::WARN) {
                    ConsoleOut<BLACK, YELLOW>(log_line);
                } else if constexpr (Lv == Level::ERR) {
                    ConsoleOut<BLACK, MAGENTA>(log_line);
                } else {
                    ConsoleOut<WHITE, RED>(log_line);
                }
            }

            FileOut(log_line);

            if constexpr (Lv == Level::FATAL) {
                stack_trace = "Stack trace:\n" + GetStackTrace();
                if (console_enabled) {
                    ConsoleOut<WHITE, RED>(stack_trace);
                }
                FileOut(stack_trace);

                if (IsValidHandle(g_instance.h_file)) {
                    FlushFileBuffers(g_instance.h_file);
                }
            }
        }

        if constexpr (Lv == Level::FATAL) {
            std::exit(EXIT_FAILURE);
        }
    }
}

#define MLOG_TRACE(...) ::MicroTrace::LogOut<::MicroTrace::Level::TRACE>(std::source_location::current(), std::time(nullptr), __VA_ARGS__)
#define MLOG_DEBUG(...) ::MicroTrace::LogOut<::MicroTrace::Level::DEBUG>(std::source_location::current(), std::time(nullptr), __VA_ARGS__)
#define MLOG_INFO(...) ::MicroTrace::LogOut<::MicroTrace::Level::INFO>(std::source_location::current(), std::time(nullptr), __VA_ARGS__)
#define MLOG_WARN(...) ::MicroTrace::LogOut<::MicroTrace::Level::WARN>(std::source_location::current(), std::time(nullptr), __VA_ARGS__)
#define MLOG_ERROR(...) ::MicroTrace::LogOut<::MicroTrace::Level::ERR>(std::source_location::current(), std::time(nullptr), __VA_ARGS__)
#define MLOG_FATAL(...) ::MicroTrace::LogOut<::MicroTrace::Level::FATAL>(std::source_location::current(), std::time(nullptr), __VA_ARGS__)
