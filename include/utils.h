#pragma once

#include <base.h>
#include <fstream>
#include <optional>

namespace vkdiffusion
{
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

    template<typename Container, typename Pred>
    void IterativeRemove(Container& container, Pred pred) {
        for (auto it = container.begin(); it != container.end(); ) {
            if (pred(*it))
                it = container.erase(it);
            else
                ++it;
        }
    }

    /* Package of C++ class that enabled lazy initializing */
    template<typename T>
    class Lazy {
    private:
        alignas(T) unsigned char object_[sizeof(T)]{};
        bool initialized_{};
    public:
        Lazy() = default;

        Lazy(Lazy&) = delete;
        Lazy(Lazy&& obj) noexcept : initialized_(obj.initialized_) {
            memcpy(object_, obj.object_, sizeof(T));
            obj.initialized_ = false;
        }

        /* When initialize params are ready, call Construct() to initialize the class */
        template<typename... Args>
        void Construct(Args&& ...args) {
            if (!initialized_) {
                new(object_) T(std::forward<Args>(args)...);
                initialized_ = true;
            }
        }

        /* After called Construct(), call Get() to get the reference of the class */
        T& Get() {
#ifndef NDEBUG
            if (!initialized_) {
                throw std::runtime_error("Cannot execute Get() of a Lazy object, the Lazy object not constructed yet");
            }
#endif
            return *reinterpret_cast<T*>(object_);
        }

        operator T&() {
            return Get();
        }

        T* operator->() {
#ifndef NDEBUG
            if (!initialized_) {
                throw std::runtime_error("Cannot execute Get() of a Lazy object, the Lazy object not constructed yet");
            }
#endif
            return reinterpret_cast<T*>(object_);
        }

        /* Indicates whether called Construct() */
        bool Constructed() {
            return initialized_;
        }

        /* Lazy objects has the same life time like common classes */
        ~Lazy() {
            if (initialized_) reinterpret_cast<T*>(object_)->~T();
        }
    };
}