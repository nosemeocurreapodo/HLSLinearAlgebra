#pragma once

#include "ap_float.h"

template <int W, int E>
class wap_float
{
public:
    ap_float<W, E> value_;

    wap_float() = default;

    wap_float(const ap_float<W, E> &v)
        : value_(v)
    {
    }

    template <int in_nbits>
    wap_float(const ap_int<in_nbits> &c)
    {
        value_ = float(c);
    }

    wap_float(const float &c)
    {
        value_ = c;
    }

    operator ap_float<W, E>() const
    {
        return value_;
    }

    template <int out_bits>
    operator ap_int<out_bits>() const
    {
        return ap_int<out_bits>(float(value_));
    }

    operator int() const
    {
        return int(float(value_));
    }

    wap_float<W, E> operator-() const
    {
        wap_float<W, E> res;
        res.value_ = -res.value_;
        return res;
    }

    wap_float<W, E> operator+(const wap_float<W, E> &rhs) const
    {
        wap_float<W, E> res;
        res.value_ = value_ + rhs.value_;
        return res;
    }

    wap_float<W, E> operator-(const wap_float<W, E> &rhs) const
    {
        wap_float<W, E> res;
        res.value_ = value_ - rhs.value_;
        return res;
    }

    wap_float<W, E> operator*(const wap_float<W, E> &rhs) const
    {
        wap_float<W, E> res;
        res.value_ = value_ * rhs.value_;
        return res;
    }

    wap_float<W, E> operator/(const wap_float<W, E> &rhs) const
    {
        wap_float<W, E> res;
        res.value_ = value_ / rhs.value_;
        return res;
    }

    wap_float<W, E> &operator+=(const wap_float<W, E> &rhs) 
    {
        value_ = value_ + rhs.value_;
        return *this;
    }

    wap_float<W, E> &operator-=(const wap_float<W, E> &rhs)
    {
        value_ = value_ - rhs.value_;
        return *this;
    }

    wap_float<W, E> &operator*=(const wap_float<W, E> &rhs)
    {
        value_ = value_ * rhs.value_;
        return *this;
    }

    wap_float<W, E> &operator/=(const wap_float<W, E> &rhs)
    {
        value_ = value_ / rhs.value_;
        return *this;
    }

    bool operator<(const wap_float<W, E> &rhs) const
    {
        return value_ < rhs.value_;
    }

    bool operator>(const wap_float<W, E> &rhs) const
    {
        return value_ > rhs.value_;
    }

    bool operator<=(const wap_float<W, E> &rhs) const
    {
        return value_ <= rhs.value_;
    }

    bool operator>=(const wap_float<W, E> &rhs) const
    {
        return value_ >= rhs.value_;
    }

    bool operator==(const wap_float<W, E> &rhs) const
    {
        return value_ == rhs.value_;
    }
};

template <int nbits, int ebits>
inline wap_float<nbits, ebits> round(const wap_float<nbits, ebits> &a)
{
    int integer = int(a);
    wap_float<nbits, ebits> diff = a - wap_float<nbits, ebits>(integer);
    if (diff > wap_float<nbits, ebits>(0.5))
        integer++;
    return wap_float<nbits, ebits>(integer);
}

template <int nbits, int ebits>
inline wap_float<nbits, ebits> floor(const wap_float<nbits, ebits> &a)
{
    int integer = int(a);
    return wap_float<nbits, ebits>(integer);
}

template <int nbits, int ebits>
inline wap_float<nbits, ebits> ceil(const wap_float<nbits, ebits> &a)
{
    int integer = int(a);
    wap_float<nbits, ebits> diff = a - wap_float<nbits, ebits>(integer);
    if (diff > wap_float<nbits, ebits>(0.0))
        integer++;
    return wap_float<nbits, ebits>(integer);
}

template <int nbits, int ebits>
inline wap_float<nbits, ebits> mod(const wap_float<nbits, ebits> &a, const wap_float<nbits, ebits> &b)
{
    wap_float<nbits, ebits> c = a / b;
    wap_float<nbits, ebits> d = a - floor(c) * b;
    return b;
}