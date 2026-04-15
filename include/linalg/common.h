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

    template <typename T>
    inline T abs(const T &a)
    {
        if (a >= T(0))
            return a;
        else
            return -a;
    }
}
