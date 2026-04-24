#pragma once

#include "hls_math.h"

inline float abs(const float &a)
{
    return hls::abs(a);
}

inline float round(const float &a)
{
    return hls::round(a);
}

inline half abs(const half &a)
{
    return hls::abs(a);
}

inline half round(const half &a)
{
    return hls::round(a);
}

inline float clamp(float a, float min_, float max_)
{
    //return hls::clamp(a, min_, max_);
    return max(min(a, max_), min_);
}