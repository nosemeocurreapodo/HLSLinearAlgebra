#pragma once

#include "ap_float.h"
#include "hls_math.h"

template <int W, int E>
class wap_float
{
public:
    ap_float<W, E> value_;

    wap_float() = default;

    /*
    wap_float(const ap_float<W, E> &v)
        : value_(v)
    {
    }
    */

    template <int in_nbits>
    wap_float(const ap_int<in_nbits> &c)
    {
        value_ = c;
    }

    wap_float(const ap_int<16> &c)
    {
        value_ = c;
    }

    template <int in_nbits>
    wap_float(const ap_uint<in_nbits> &c)
    {
        value_ = c;
    }

    wap_float(const int &c)
    {
        value_ = float(c);
    }

    wap_float(const unsigned int &c)
    {
        value_ = float(c);
    }

    wap_float(const float &c)
    {
        value_ = c;
    }

    wap_float(const double &c)
    {
        value_ = c;
    }

    /*
    operator ap_float<W, E>() const
    {
        return value_;
    }
    */

    template <int out_bits>
    operator ap_int<out_bits>() const
    {
        return ap_int<out_bits>(value_);
    }

    operator ap_int<16>() const
    {
        return ap_int<16>(value_);
    }

    template <int out_bits>
    operator ap_uint<out_bits>() const
    {
        return ap_uint<out_bits>(value_);
    }

    operator int() const
    {
        return int(float(value_));
    }

    operator float() const
    {
        return float(value_);
    }

    operator double() const
    {
        return double(value_);
    }

    wap_float<W, E> operator-() const
    {
        wap_float<W, E> res;
        res.value_ = -value_;
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

    bool operator!=(const wap_float<W, E> &rhs) const
    {
        return value_ != rhs.value_;
    }
};

template <int nbits, int ibits>
inline wap_float<nbits, ibits> fabs(const wap_float<nbits, ibits> &a)
{
    wap_float<nbits, ibits> res = a;
    if (res.value_ < 0.0)
        res = -res;
    return res;
}

template <int nbits, int ibits>
inline wap_float<nbits, ibits> floor(const wap_float<nbits, ibits> &a)
{
    wap_float<nbits, ibits> b = int(a);
    return b;
}

template <int nbits, int ibits>
inline wap_float<nbits, ibits> ceil(const wap_float<nbits, ibits> &a)
{
    wap_float<nbits, ibits> b = int(a);
    wap_float<nbits, ibits> diff = a - b;
    if (diff > wap_float<nbits, ibits>(0))
        b += wap_float<nbits, ibits>(1);
    return b;
}

template <int nbits, int ibits>
inline wap_float<nbits, ibits> round(const wap_float<nbits, ibits> &a)
{
    wap_float<nbits, ibits> b = int(a);
    wap_float<nbits, ibits> diff = a - b;
    if (diff > wap_float<nbits, ibits>(0.5))
        b += wap_float<nbits, ibits>(1);
    return b;
}

template <int nbits, int ibits>
inline wap_float<nbits, ibits> fmod(const wap_float<nbits, ibits> &a, const wap_float<nbits, ibits> &b)
{
    wap_float<nbits, ibits> c = a / b;
    wap_float<nbits, ibits> d = a - floor(c) * b;
    return d;
}

template <int nbits, int ibits>
inline wap_float<nbits, ibits> exp(const wap_float<nbits, ibits> &a)
{
    return wap_float<nbits, ibits>(1);
}
