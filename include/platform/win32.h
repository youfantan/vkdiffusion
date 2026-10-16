#pragma once

#include <base.h>
#include <chrono>
#include <string>

namespace vkdiffusion
{
    class Window {
    private:
        uint32_t width_;
        uint32_t height_;
        uint32_t x_;
        uint32_t y_;
        HWND hwnd_;
        bool closed_;

        static LRESULT CALLBACK WindowProcess(HWND hwnd, UINT msg, WPARAM w, LPARAM l);
    public:
        Window(uint32_t width, uint32_t height, uint32_t x, uint32_t y) : width_(width), height_(height), x_(x), y_(y), closed_(false) {
            std::wstring class_name = L"vkdiffusion";
            WNDCLASS wc {};
            wc.lpfnWndProc = WindowProcess;
            wc.hInstance = nullptr;
            wc.lpszClassName = class_name.c_str();
            RegisterClass(&wc);
            hwnd_ = CreateWindowEx(0, class_name.c_str(), L"vkdiffusion", WS_OVERLAPPEDWINDOW, x_, y_, width_, height_, nullptr, nullptr, nullptr, this);
            if (hwnd_ == nullptr) {

            }
            ShowWindow(hwnd_, 1);
        }

        const HWND GetHandle() const {
            return hwnd_;
        }

        const bool Closed() const {
            return closed_;
        }

        const uint32_t GetWidth() const {
            return width_;
        }

        const uint32_t GetHeight() const {
            return height_;
        }

        bool FetchMessage() {
            MSG msg {};
            if (PeekMessage(&msg, hwnd_, 0, 0, PM_REMOVE) != 0) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
                return true;
            }
            return false;
        }

        VkSurfaceKHR CreateSurface(VkInstance inst) {
            VkWin32SurfaceCreateInfoKHR info {
                .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                .pNext = nullptr,
                .flags = 0,
                .hinstance = GetModuleHandle(nullptr),
                .hwnd = hwnd_,
            };
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            VKCHECK(vkCreateWin32SurfaceKHR(inst, &info, nullptr, &surface));
            return surface;
        }

        Window(Window&) = delete;
        Window(Window&&) = delete;
        Window& operator=(Window&) = delete;
        Window& operator=(Window&&) = delete;
        ~Window() {
            DestroyWindow(hwnd_);
        }
    };

    inline LRESULT Window::WindowProcess(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
        Window* window {};
        if (msg != WM_NCCREATE) window = reinterpret_cast<Window*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
        switch (msg) {
            case WM_NCCREATE: {
                auto* cs = reinterpret_cast<CREATESTRUCT*>(l);
                window = static_cast<Window*>(cs->lpCreateParams);
                SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
                break;
            }

            case WM_DESTROY: {
                PostQuitMessage(0);
                window->closed_ = true;
                return 0;
            }

            case WM_ACTIVATE: {
                break;
            }
        }
        return DefWindowProc(hwnd, msg, w, l);
    }

    class HighPerformanceCounter {
    private:
        LARGE_INTEGER frequency_ {};
        LARGE_INTEGER start_ {};
        LARGE_INTEGER end_ {};
        double duration_us_;
    public:
        HighPerformanceCounter() : duration_us_(0.0) {
            QueryPerformanceFrequency(&frequency_);
            start_.QuadPart = 0;
            end_.QuadPart = 0;
        }

        void Start() {
            start_.QuadPart = 0;
            end_.QuadPart = 0;
            duration_us_ = 0.0;
            QueryPerformanceCounter(&start_);
        }

        void Stop() {
            QueryPerformanceCounter(&end_);
            duration_us_ += static_cast<double>(end_.QuadPart - start_.QuadPart) * 1000000.0 / static_cast<double>(frequency_.QuadPart);
        }

        std::chrono::microseconds GetDurationUs() const {
            return std::chrono::microseconds(static_cast<long long>(duration_us_));
        }
    };
}