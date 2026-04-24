#pragma once

// #include <cstdint>
// #include <cmath>

// Common utility functions for linear algebra operations
namespace linalg
{
    // Add common utility functions here as needed

    enum class VecOrient
    {
        Column,
        Row
    };
}

template <typename T>
inline T max(const T &a, const T &b)
{
    return a > b ? a : b;
}

template <typename T>
inline T min(const T &a, const T &b)
{
    return a < b ? a : b;
}

template <typename T>
inline T clamp(T a, T _min, T _max)
{
    // return hls::clamp(a, _min, _max);
    return min(max(a, _min), _max);
}