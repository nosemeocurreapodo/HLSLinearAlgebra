#pragma once

#include "hls_math.h"

template <typename T>
inline T fabs(const T &a)
{
    return hls::abs(a);
}

template <typename T>
inline T round(const T &a)
{
    return hls::round(a);
}

template <typename T>
inline T floor(const T &a)
{
    return hls::floor(a);
}

template <typename T>
inline T ceil(const T &a)
{
    return hls::ceil(a);
}

template <typename T>
inline T fmod(const T &a, const T &b)
{
    return hls::fmod(a, b);
}

template <typename T>
inline T exp(const T &a)
{
    return hls::exp(a);
}
