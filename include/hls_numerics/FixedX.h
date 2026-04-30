#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

#include "common.h"
#include "FloatX.h"

template <int nbits, int ibits>
class FixedX
{
public:
    static constexpr int fbits = nbits - ibits;

    FixedX()
    {
    }

    template <int in_nbits, int in_ibits>
    FixedX(const FixedX<in_nbits, in_ibits> &other)
    {
        // #pragma HLS INLINE off

        ap_int<nbits> bits;
        if (fbits >= other.fbits)
            bits = (ap_int<nbits>)other.bits_ << (fbits - other.fbits);
        else
            bits = other.bits_ >> (other.fbits - fbits);
        bits_ = bits;
    }

    template <int in_nbits, int in_ibits>
    FixedX &operator=(const FixedX<in_nbits, in_ibits> &other)
    {
        // #pragma HLS INLINE off

        // if (this != &other)
        {
            ap_int<nbits> bits;
            if (fbits >= other.fbits)
                bits = (ap_int<nbits>)other.bits_ << (fbits - other.fbits);
            else
                bits = other.bits_ >> (other.fbits - fbits);
            bits_ = bits;
        }
        return *this;
    }

    template <int in_nbits>
    FixedX(ap_int<in_nbits> c)
    {
        ap_int<in_nbits + fbits> b = (ap_int<in_nbits + fbits>)c << fbits;
        bits_ = b;
    }

    template <int in_nbits>
    FixedX(ap_uint<in_nbits> c)
    {
        ap_int<in_nbits + fbits> b = (ap_int<in_nbits + fbits>)c << fbits;
        bits_ = b;
    }

    FixedX(int c)
    {
        bits_ = (ap_int<nbits>)c << fbits;
    }

    FixedX(unsigned int c)
    {
        bits_ = (ap_int<nbits>)c << fbits;
    }

    FixedX(float c)
    {
        // #pragma HLS INLINE off

        if (c == 0)
        {
            bits_ = 0;
            return;
        }

        ap_uint<32> bits = bitcast_u32(c);
        FloatXUnpacked<32, 8> float_unpacked;
        float_unpacked.decode(bits);

        int shift = -23 + (float_unpacked.exp_ - fbias<8>::value) + fbits;
        ap_int<25 + nbits> base = (ap_uint<2>(0b01), float_unpacked.mant_);
        if (float_unpacked.sign_)
            base = -base;

        base = base << shift;
        bits_ = base;
        // ap_int<nbits> fixed;
        // if (shift >= 0)
        //     fixed = base << shift;
        // else
        //     fixed = base >> (-shift);
        // bits_ = fixed;
    }

    FixedX(double c)
    {
        // #pragma HLS INLINE off

        if (c == 0)
        {
            bits_ = 0;
            return;
        }

        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<64, 11> float_unpacked;
        float_unpacked.decode(bits);

        int shift = -52 + (float_unpacked.exp_ - fbias<11>::value) + fbits;
        ap_int<54 + nbits> base = (ap_uint<2>(0b01), float_unpacked.mant_);
        if (float_unpacked.sign_)
            base = -base;

        base = base << shift;
        bits_ = base;

        // ap_int<nbits> fixed;
        // if (shift >= 0)
        //     fixed = base << shift;
        // else
        //     fixed = base >> (-shift);
        // bits_ = fixed;
    }

    template <int out_nbits>
    operator ap_int<out_nbits>() const
    {
        ap_int<out_nbits> res = (ap_int<out_nbits>)bits_ >> fbits;
        return res;
    }

    operator int() const
    {
        ap_int<nbits> aux = bits_;
        ap_int<ibits> res = aux(nbits - 1, fbits);
        if (aux < 0 && aux(fbits - 1, 0) > 0)
            res += 1;
        return res;
    }

    operator unsigned int() const
    {
        int res = int(*this);
        return (unsigned int)res;
    }

    operator float() const
    {
        // #pragma HLS INLINE off

        if (bits_ == 0)
            return 0.0f;

        bool sign = bits_ < 0 ? 1 : 0;
        ap_int<nbits> abs_bits = bits_;
        if (sign)
            abs_bits = -abs_bits;
        int shift = count_leading_zeros((ap_uint<nbits>)abs_bits);

        abs_bits = abs_bits << (shift + 1);

        FloatXUnpacked<32, 8> float_unpacked;
        float_unpacked.zero_ = 0;
        float_unpacked.inf_ = 0;
        float_unpacked.sign_ = sign;
        float_unpacked.exp_ = -fbits + nbits - shift + fbias<8>::value - 1;
        float_unpacked.mant_ = 0;

        if (nbits <= 23)
            float_unpacked.mant_(22, 23 - nbits) = abs_bits;
        else
            float_unpacked.mant_ = abs_bits(nbits - 1, nbits - 23);
        ap_uint<32> bits = float_unpacked.encode();

        // float fresult = *reinterpret_cast<float *>(&bits);
        // return fresult;
        return bitcast_f32(bits);
    }

    operator double() const
    {
        // #pragma HLS INLINE off

        if (bits_ == 0)
            return 0.0;

        bool sign = bits_ < 0 ? 1 : 0;

        ap_int<nbits> abs_bits = bits_;

        if (sign)
            abs_bits = -abs_bits;

        int shift = count_leading_zeros((ap_uint<nbits>)abs_bits);
        abs_bits = abs_bits << (shift + 1);

        FloatXUnpacked<64, 11> float_unpacked;
        float_unpacked.zero_ = 0;
        float_unpacked.inf_ = 0;
        float_unpacked.sign_ = sign;
        float_unpacked.exp_ = -fbits + nbits - shift + fbias<11>::value - 1;
        float_unpacked.mant_ = 0;

        if (nbits <= 52)
            float_unpacked.mant_(51, 52 - nbits) = abs_bits(nbits - 1, 0);
        else
            float_unpacked.mant_ = abs_bits(nbits - 1, nbits - 52);
        ap_uint<64> bits = float_unpacked.encode();

        // float fresult = *reinterpret_cast<float *>(&bits);
        // return fresult;
        return bitcast_f64(bits);
    }

    FixedX<nbits + 1, ibits + 1> operator+(const FixedX &rhs) const
    {
        // #pragma HLS INLINE off

        FixedX<nbits + 1, ibits + 1> res;
        res.bits_ = bits_ + rhs.bits_;
        return res;
    }

    FixedX<nbits + 1, ibits + 1> operator-(const FixedX &rhs) const
    {
        // #pragma HLS INLINE off

        FixedX<nbits + 1, ibits + 1> res;
        res.bits_ = bits_ - rhs.bits_;
        return res;
    }

    template <int in_nbits, int in_ibits>
    FixedX<nbits + in_nbits, ibits + in_ibits> operator*(const FixedX<in_nbits, in_ibits> &rhs) const
    {
        // #pragma HLS INLINE off
        FixedX<nbits + in_nbits, ibits + in_ibits> res;
        res.bits_ = bits_ * rhs.bits_;
        return res;
    }

    /*
    FixedX<nbits * 2, ibits * 2> operator*(const FixedX &rhs) const
    {
#pragma HLS INLINE off

        ap_int<nbits * 2> res = bits_ * rhs.bits_;
        return FixedX<nbits * 2, ibits * 2>(res);
    }
    */

    FixedX<nbits, ibits> operator/(const FixedX &rhs) const
    {
        // #pragma HLS INLINE off
        FixedX<nbits, ibits> res;
        ap_int<nbits * 2> num = (ap_int<nbits * 2>)bits_ << fbits;
        res.bits_ = num / rhs.bits_;
        return res;
    }

    FixedX operator-() const
    {
        // #pragma HLS INLINE off

        FixedX res;
        res.bits_ = -bits_;
        return res;
    }

    template <int in_nbits, int in_ibits>
    FixedX &operator*=(const FixedX<in_nbits, in_ibits> &rhs)
    {
        FixedX<nbits + in_nbits, ibits + in_ibits> res = *this * rhs;
        *this = res;
        return *this;
    }

    template <int in_nbits, int in_ibits>
    FixedX &operator/=(const FixedX<in_nbits, in_ibits> &rhs)
    {
        FixedX<nbits + in_nbits, ibits + in_ibits> res = *this / rhs;
        *this = res;
        return *this;
    }

    FixedX &operator+=(const FixedX &rhs)
    {
        FixedX res = *this + rhs;
        *this = res;
        return *this;
    }

    FixedX &operator-=(const FixedX &rhs)
    {
        FixedX res = *this - rhs;
        *this = res;
        return *this;
    }

    bool operator==(const FixedX &rhs) const
    {
        return bits_ == rhs.bits_;
    }

    bool operator!=(const FixedX &rhs) const
    {
        return bits_ != rhs.bits_;
    }

    bool operator<(const FixedX &rhs) const
    {
        return bits_ < rhs.bits_;
    }

    bool operator>(const FixedX &rhs) const
    {
        return rhs.bits_ < bits_;
    }

    bool operator<=(const FixedX &rhs) const
    {
        return bits_ == rhs.bits_ || bits_ < rhs.bits_;
    }

    bool operator>=(const FixedX &rhs) const
    {
        return bits_ == rhs.bits_ || rhs.bits_ < bits_;
    }

    // private:
    ap_int<nbits> bits_;
};

template <int nbits, int ibits>
inline FixedX<nbits, ibits> fabs(const FixedX<nbits, ibits> &a)
{
    return a.bits_ >= 0 ? a : -a;
}

template <int nbits, int ibits>
inline FixedX<nbits, ibits> floor(const FixedX<nbits, ibits> &a)
{
    FixedX<nbits, ibits> b = int(a);
    return b;
}

template <int nbits, int ibits>
inline FixedX<nbits, ibits> ceil(const FixedX<nbits, ibits> &a)
{
    FixedX<nbits, ibits> b = int(a);
    FixedX<nbits, ibits> diff = a - b;
    if (diff > FixedX<nbits, ibits>(0))
        b += FixedX<nbits, ibits>(1);
    return b;
}

template <int nbits, int ibits>
inline FixedX<nbits, ibits> round(const FixedX<nbits, ibits> &a)
{
    FixedX<nbits, ibits> b = int(a);
    FixedX<nbits, ibits> diff = a - b;
    if (diff > FixedX<nbits, ibits>(0.5))
        b += FixedX<nbits, ibits>(1);
    return b;
}

template <int nbits, int ibits>
inline FixedX<nbits, ibits> fmod(const FixedX<nbits, ibits> &a, const FixedX<nbits, ibits> &b)
{
    FixedX<nbits, ibits> c = a / b;
    FixedX<nbits, ibits> d = a - floor(c) * b;
    return d;
}

template <int nbits, int ibits>
inline FixedX<nbits, ibits> exp(const FixedX<nbits, ibits> &a)
{
    return FixedX<nbits, ibits>(1);
}
