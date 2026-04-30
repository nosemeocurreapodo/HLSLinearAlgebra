#pragma once

#include "ap_fixed.h"
#include "hls_math.h"

template <int W, int E>
class wap_fixed
{
public:
    ap_fixed<W, E> value_;

    wap_fixed() = default;

    /*
    wap_fixed(const ap_fixed<W, E> &v)
        : value_(v)
    {
    }
    */

    template <int in_nbits>
    wap_fixed(const ap_int<in_nbits> &c)
    {
        value_ = c;
    }

    template <int in_nbits>
    wap_fixed(const ap_uint<in_nbits> &c)
    {
        value_ = c;
    }

    wap_fixed(const int &c)
    {
        value_ = c;
    }

    wap_fixed(const unsigned int &c)
    {
        value_ = c;
    }

    wap_fixed(const float &c)
    {
        value_ = c;
    }

    wap_fixed(const double &c)
    {
        value_ = c;
    }

    /*
    operator ap_fixed<W, E>() const
    {
        return value_;
    }
    */

    template <int out_bits>
    operator ap_int<out_bits>() const
    {
        return ap_int<out_bits>(value_);
    }

    template <int out_bits>
    operator ap_uint<out_bits>() const
    {
        return ap_uint<out_bits>(value_);
    }

    operator int() const
    {
        return int(value_);
    }

    operator float() const
    {
        return float(value_);
    }

    operator double() const
    {
        return double(value_);
    }

    wap_fixed<W, E> operator-() const
    {
        wap_fixed<W, E> res;
        res.value_ = -value_;
        return res;
    }

    wap_fixed<W, E> operator+(const wap_fixed<W, E> &rhs) const
    {
        wap_fixed<W, E> res;
        res.value_ = value_ + rhs.value_;
        return res;
    }

    wap_fixed<W, E> operator-(const wap_fixed<W, E> &rhs) const
    {
        wap_fixed<W, E> res;
        res.value_ = value_ - rhs.value_;
        return res;
    }

    wap_fixed<W, E> operator*(const wap_fixed<W, E> &rhs) const
    {
        wap_fixed<W, E> res;
        res.value_ = value_ * rhs.value_;
        return res;
    }

    wap_fixed<W, E> operator/(const wap_fixed<W, E> &rhs) const
    {
        wap_fixed<W, E> res;
        if (rhs.value_ == 0)
        {
            res.value_[W - 1] = 1;
            res.value_(W - 2, 1) = 0;
            res.value_[0] = 1;
            if (value_ >= 0)
                res.value_ = -res.value_;
        }
        else
            res.value_ = value_ / rhs.value_;
        return res;
    }

    wap_fixed<W, E> &operator+=(const wap_fixed<W, E> &rhs)
    {
        value_ = value_ + rhs.value_;
        return *this;
    }

    wap_fixed<W, E> &operator-=(const wap_fixed<W, E> &rhs)
    {
        value_ = value_ - rhs.value_;
        return *this;
    }

    wap_fixed<W, E> &operator*=(const wap_fixed<W, E> &rhs)
    {
        value_ = value_ * rhs.value_;
        return *this;
    }

    wap_fixed<W, E> &operator/=(const wap_fixed<W, E> &rhs)
    {
        value_ = value_ / rhs.value_;
        return *this;
    }

    bool operator<(const wap_fixed<W, E> &rhs) const
    {
        return value_ < rhs.value_;
    }

    bool operator>(const wap_fixed<W, E> &rhs) const
    {
        return value_ > rhs.value_;
    }

    bool operator<=(const wap_fixed<W, E> &rhs) const
    {
        return value_ <= rhs.value_;
    }

    bool operator>=(const wap_fixed<W, E> &rhs) const
    {
        return value_ >= rhs.value_;
    }

    bool operator==(const wap_fixed<W, E> &rhs) const
    {
        return value_ == rhs.value_;
    }

    bool operator!=(const wap_fixed<W, E> &rhs) const
    {
        return value_ != rhs.value_;
    }
};

template <int nbits, int ibits>
inline wap_fixed<nbits, ibits> fabs(const wap_fixed<nbits, ibits> &a)
{
    wap_fixed<nbits, ibits> res = a;
    if (res.value_ < 0.0)
        res = -res;
    return res;

    // wap_fixed<nbits, ibits> res;
    // res.value_ = hls::abs(a.value_);
    // return res;
}

template <int nbits, int ibits>
inline wap_fixed<nbits, ibits> floor(const wap_fixed<nbits, ibits> &a)
{
    wap_fixed<nbits, ibits> b = int(a);
    return b;

    // wap_fixed<nbits, ibits> res;
    // res.value_ = hls::floor(a.value_);
    // return res;
}

template <int nbits, int ibits>
inline wap_fixed<nbits, ibits> ceil(const wap_fixed<nbits, ibits> &a)
{
    wap_fixed<nbits, ibits> b = int(a);
    wap_fixed<nbits, ibits> diff = a - b;
    if (diff > wap_fixed<nbits, ibits>(0))
        b += wap_fixed<nbits, ibits>(1);
    return b;

    // wap_fixed<nbits, ibits> res;
    // res.value_ = hls::ceil(a.value_);
    // return res;
}

template <int nbits, int ibits>
inline wap_fixed<nbits, ibits> round(const wap_fixed<nbits, ibits> &a)
{
    wap_fixed<nbits, ibits> b = int(a);
    wap_fixed<nbits, ibits> diff = a - b;
    if (diff > wap_fixed<nbits, ibits>(0.5))
        b += wap_fixed<nbits, ibits>(1);
    return b;

    // wap_fixed<nbits, ibits> res;
    // res.value_ = hls::round(a.value_);
    // return res;
}

template <int nbits, int ibits>
inline wap_fixed<nbits, ibits> fmod(const wap_fixed<nbits, ibits> &a, const wap_fixed<nbits, ibits> &b)
{
    wap_fixed<nbits, ibits> c = a / b;
    wap_fixed<nbits, ibits> d = a - floor(c) * b;
    return d;

    // wap_fixed<nbits, ibits> res;
    // res.value_ = hls::fmod(a.value_, b.value_);
    // return res;
}

template <int nbits, int ibits>
inline wap_fixed<nbits, ibits> exp(const wap_fixed<nbits, ibits> &a)
{
    return wap_fixed<nbits, ibits>(1);
    // wap_fixed<nbits, ibits> res;
    // res.value_ = hls::exp(a.value_);
    // return res;
}
