#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

#include "common.h"

template <int nbits, int ebits>
class FloatXUnpacked
{
public:
    static const int fbits = nbits - ebits - 1;

    FloatXUnpacked()
    {
    }

    template <int onbits, int oebits>
    FloatXUnpacked(const FloatXUnpacked<onbits, oebits> &other)
    {
#pragma HLS INLINE off

        ap_int<oebits + 1> oexp = (ap_int<oebits + 1>)(ap_uint<1>(0), other.exp_) -
                                  (ap_int<oebits + 1>)fbias<oebits>::value;
        ap_int<ebits + 1> exp = oexp + (ap_int<ebits + 1>)fbias<ebits>::value;

        // ap_int<oebits + 1> exp = (ap_int<oebits + 1>)(ap_uint<1>(0), other.exp_) -
        //                          (ap_int<oebits + 1>)fbias<oebits>::value +
        //                          (ap_int<oebits + 1>)fbias<ebits>::value;

        zero_ = other.zero_;
        inf_ = other.inf_;
        sign_ = other.sign_;
        exp_ = exp(ebits - 1, 0);
        if (fbits >= other.fbits)
        {
            // no rounding needed
            mant_(fbits - 1, fbits - other.fbits) = other.mant_(other.fbits - 1, 0);
            mant_(fbits - other.fbits - 1, 0) = 0;
        }
        else
        {
            // round as well
            // mant_(fbits - 1, 0) = other.mant_(ofbits - 1, ofbits - fbits);

            ap_uint<fbits + 2> rfrac = (ap_uint<1>(0), other.mant_(other.fbits - 1, other.fbits - fbits - 1));
            if (rfrac[0] == 1)
                rfrac += 1;
            if (rfrac[fbits + 1] == 1)
            {
                rfrac = rfrac >> 1;
                exp_++;
            }
            mant_(fbits - 1, 0) = rfrac(fbits, 1);
        }
    }

    template <int in_nbits>
    FloatXUnpacked(ap_int<in_nbits> c)
    {
        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            exp_ = 0;
            mant_ = 0;
            return;
        }

        zero_ = 0;
        inf_ = 0;

        bool psign = c < 0;
        ap_uint<32> i = hls::abs(c);
        int lz = count_leading_zeros(i);
        i = i << (lz + 1);

        sign_ = psign;
        exp_ = 31 - lz + fbias<ebits>::value;
        if constexpr (fbits <= 32)
            mant_ = i(31, 32 - fbits);
        else
            mant_(fbits - 1, fbits - 32) = i;
    }

    template <int in_nbits>
    FloatXUnpacked(ap_uint<in_nbits> c)
    {
        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            exp_ = 0;
            mant_ = 0;
            return;
        }

        zero_ = 0;
        inf_ = 0;

        ap_uint<32> i = hls::abs(c);
        int lz = count_leading_zeros(i);
        i = i << (lz + 1);

        sign_ = 0;
        exp_ = 31 - lz + fbias<ebits>::value;
        if constexpr (fbits <= 32)
            mant_ = i(31, 32 - fbits);
        else
            mant_(fbits - 1, fbits - 32) = i;
    }

    FloatXUnpacked(int c)
    {
        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            exp_ = 0;
            mant_ = 0;
            return;
        }

        zero_ = 0;
        inf_ = 0;

        bool psign = c < 0;
        ap_uint<32> i = hls::abs(c);
        int lz = count_leading_zeros(i);
        i = i << (lz + 1);

        sign_ = psign;
        exp_ = 31 - lz + fbias<ebits>::value;
        if constexpr (fbits <= 32)
            mant_ = i(31, 32 - fbits);
        else
            mant_(fbits - 1, fbits - 32) = i;
    }

    FloatXUnpacked(unsigned int c)
    {
        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            exp_ = 0;
            mant_ = 0;
            return;
        }

        zero_ = 0;
        inf_ = 0;

        ap_uint<32> i = c;
        int lz = count_leading_zeros(i);
        i = i << (lz + 1);

        sign_ = 0;
        exp_ = 31 - lz + fbias<ebits>::value;
        if constexpr (fbits <= 32)
            mant_ = i(31, 32 - fbits);
        else
            mant_(fbits - 1, fbits - 32) = i;
    }

    template <int in_nbits>
    operator ap_int<in_nbits>() const
    {
        if (zero_ == 0)
            return 0;

        int exp = exp_ - fbias<ebits>::value;
        ap_int<in_nbits> res = (ap_uint<2>(0b01), mant_) << exp;
        if (sign_)
        {
            res = -res;
        }
        return res;
    }

    operator int() const
    {
        if (zero_ == 0)
            return 0;

        int exp = exp_ - fbias<ebits>::value;
        int res = (ap_uint<2>(0b01), mant_) << exp;
        if (sign_)
        {
            res = -res;
        }
        return res;
    }

    operator unsigned int() const
    {
        int exp = exp_ - fbias<ebits>::value;
        int res = (ap_uint<2>(0b01), mant_) << exp;
        return res;
    }

    /*
    FloatXUnpacked &operator=(const FloatXUnpacked &other)
    {
        // #pragma HLS INLINE

        // if (this != &other)
        {
            bits_ = other.bits_;
        }
        return *this;
    }
    */

    void decode(const ap_uint<nbits> &bits)
    {
#pragma HLS INLINE off

        if (bits == 0)
            zero_ = 1;
        else
            zero_ = 0;

        inf_ = 0;

        sign_ = bits[nbits - 1];
        exp_ = bits(nbits - 2, fbits);
        mant_ = bits(fbits - 1, 0);

        // if (exp_ == ap_uint<ebits>(-1) && mant_ == 0)
        //     inf_ = 1;
        // else
        //     inf_ = 0;
    }

    ap_uint<nbits> encode() const
    {
#pragma HLS INLINE off

        ap_uint<nbits> bits;

        if (zero_ == 1)
        {
            bits = 0;
        }
        else
        {
            bits[nbits - 1] = sign_;
            bits(nbits - 2, fbits) = exp_;
            bits(fbits - 1, 0) = mant_;
        }

        return bits;
    }

    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE off

        // if (sign_ == 0 && exp_ == 0 && mant_ == 0)
        if (zero_)
            return rhs;
        // if (rhs.sign_ == 0 && rhs.exp_ == 0 && rhs.mant_ == 0)
        if (rhs.zero_)
            return *this;

        /*
        if (inf_ && rhs.inf_)
        {
            // define your policy here
            // for now, if opposite signs, return zero or NaN-like state
            if (sign_ != rhs.sign_)
            {
                out.zero_ = 1;
                out.inf_ = 0;
                out.sign_ = 0;
                return out;
            }
            return *this;
        }
        if (inf_ == 1)
            return *this;
        if (rhs.inf_ == 1)
            return rhs;
        */

        FloatXUnpacked in1;
        FloatXUnpacked in2;

        bool lhs_ge_rhs =
            (exp_ > rhs.exp_) ||
            (exp_ == rhs.exp_ && mant_ >= rhs.mant_);

        // muxes
        if (lhs_ge_rhs)
        {
            in1 = *this;
            in2 = rhs;
        }
        else
        {
            in1 = rhs;
            in2 = *this;
        }

        // sure to be possitibe
        ap_uint<ebits> diff_texp = in1.exp_ - in2.exp_;

        ap_uint<fbits + 2> frac1 = (ap_uint<1>(1), in1.mant_, ap_uint<1>(0));
        ap_uint<fbits + 2> frac2 = (ap_uint<1>(1), in2.mant_, ap_uint<1>(0));
        frac2 = frac2 >> diff_texp;

        // do addition (result is sure to be positive)
        ap_uint<fbits + 3> frac;

        if (in1.sign_ == in2.sign_)
            frac = frac1 + frac2;
        else
            frac = frac1 - frac2;

        // if (in1.sign_ != in2.sign_)
        //     frac2 = -frac2;

        // frac = frac1 + frac2;

        // #pragma HLS BIND_OP variable = frac op = add impl = dsp latency = -1

        if (frac == 0)
        {
            in1.zero_ = 1;
        }
        else
        {
            in1.zero_ = 0;
        }

        // normalize
        ap_uint<clog2<nbits>::value> lz = count_leading_zeros(frac);

        in1.exp_ = in1.exp_ - lz + 1;

        // round
        // if (frac[0] == 1)
        //     frac += 2;

        // if (frac[fbits + 2] == 1)
        //{
        //     frac = frac >> 1;
        //     exp++;
        // }

        frac = frac << lz;
        in1.mant_ = frac(fbits + 1, 2);

        return in1;
    }

    FloatXUnpacked operator-(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked in2;
        in2.zero_ = rhs.zero_;
        in2.inf_ = rhs.inf_;
        in2.sign_ = !rhs.sign_;
        in2.exp_ = rhs.exp_;
        in2.mant_ = rhs.mant_;

        FloatXUnpacked result = (*this) + in2;
        return result;
    }

    FloatXUnpacked operator*(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked out;

        out.zero_ = zero_ | rhs.zero_;
        out.inf_ = inf_ | rhs.inf_;
        out.sign_ = sign_ ^ rhs.sign_;

        ap_uint<ebits + 1> exp = exp_ + rhs.exp_ - (ap_uint<ebits + 1>)fbias<ebits>::value;
        ap_uint<fbits * 2 + 2> frac = (ap_uint<2>(0b01), mant_) * (ap_uint<2>(0b01), rhs.mant_);

        // normalize
        if (frac[fbits * 2 + 1] == 1)
        {
            frac = frac >> 1;
            exp++;
        }

        // round
        /*
        if (frac[fbits - 1] == 1)
            frac += 1;

        // normalize (again)
        if (frac[fbits * 2 + 1] == 1)
        {
            frac = frac >> 1;
            exp++;
        }
        */

        out.exp_ = exp(ebits - 1, 0);
        out.mant_ = frac(fbits * 2, fbits);

        return out;
    }

    FloatXUnpacked operator/(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked out;

        out.zero_ = zero_;
        out.inf_ = rhs.zero_;
        out.sign_ = sign_ ^ rhs.sign_;

        ap_uint<fbits + 1> num = (ap_uint<1>(1), mant_);
        ap_uint<fbits + 1> den = (ap_uint<1>(1), rhs.mant_);

        ap_uint<ebits + 1> exp = exp_ - rhs.exp_ + (ap_uint<ebits + 1>)fbias<ebits>::value;

        ap_uint<fbits * 2 + 2> num1 = (ap_uint<fbits * 2 + 2>)num << (fbits + 1);
        ap_uint<fbits * 2 + 2> frac = num1 / den;

        // normalize
        if (frac[fbits + 1] == 0)
        {
            frac = frac << 1;
            exp--;
        }

        // ap_ufixed<fbits * 2 + 1, 1> rfrac = round_to(frac, fbits - 1);

        // normalize
        // if (rfrac < 1)
        //{
        //    rfrac = rfrac << 1;
        //    exp--;
        //}

        // if (rfrac == 0)
        //{
        //     exp = 0;
        //     frac = 0;
        // }

        out.exp_ = exp(ebits - 1, 0);
        out.mant_ = frac(fbits, 1);

        return out;
    }

    FloatXUnpacked operator-() const
    {
#pragma HLS INLINE off

        FloatXUnpacked result;
        result.sign_ = !sign_;
        result.exp_ = exp_;
        result.mant_ = mant_;

        return result;
    }

    bool operator==(const FloatXUnpacked &rhs)
    {
        if (sign_ != rhs.sign_)
        {
            return false;
        }

        if (exp_ != rhs.exp_)
        {
            return false;
        }

        if (mant_ != rhs.mant_)
        {
            return false;
        }

        return true;
    }

    bool operator<(const FloatXUnpacked &rhs) const
    {
        if (zero_)
        {
            if (rhs.zero_)
                return false;
            else
                return rhs.sign_;
        }

        if (sign_ != rhs.sign_)
        {
            return rhs.sign_;
        }

        if (exp_ != rhs.exp_)
        {
            return sign_ != (exp_ < rhs.exp_);
        }

        if (mant_ != rhs.mant_)
        {
            return sign_ != (mant_ < rhs.mant_);
        }

        return false;
    }

    bool operator>(const FloatXUnpacked &rhs) const
    {
        return rhs < *this;
    }

    bool sign_;
    ap_uint<ebits> exp_;
    ap_uint<fbits> mant_;
    bool zero_;
    bool inf_;
};

template <int nbits, int ebits>
class FloatX
{
public:
    static constexpr int fbits = nbits - ebits - 1;

    FloatX()
    {
    }

    FloatX(const FloatX &other)
    {
#pragma HLS INLINE off

        bits_ = other.bits_;
    }

    FloatX &operator=(const FloatX &other)
    {
#pragma HLS INLINE off

        // if (this != &other)
        {
            bits_ = other.bits_;
        }
        return *this;
    }

    template <int in_nbits>
    FloatX(ap_int<in_nbits> c)
    {
        FloatXUnpacked<nbits, ebits> float_unpacked(c);
        bits_ = float_unpacked.encode();
    }

    template <int in_nbits>
    FloatX(ap_uint<in_nbits> c)
    {
        FloatXUnpacked<nbits, ebits> float_unpacked(c);
        bits_ = float_unpacked.encode();
    }

    FloatX(int c)
    {
        FloatXUnpacked<nbits, ebits> float_unpacked(c);
        bits_ = float_unpacked.encode();
    }

    FloatX(unsigned int c)
    {
        FloatXUnpacked<nbits, ebits> float_unpacked(c);
        bits_ = float_unpacked.encode();
    }

    FloatX(float c)
    {
#pragma HLS INLINE off

        // ap_uint<32> bits = *reinterpret_cast<ap_uint<32> *>(&c);
        ap_uint<32> bits = bitcast_u32(c);
        FloatXUnpacked<32, 8> float_unpacked;
        float_unpacked.decode(bits);
        FloatXUnpacked<nbits, ebits> floatx_unpacked(float_unpacked);
        bits_ = floatx_unpacked.encode();
    }

    FloatX(double c)
    {
#pragma HLS INLINE off

        // ap_uint<64> bits = *reinterpret_cast<ap_uint<64> *>(&c);
        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<64, 11> double_unpacked;
        double_unpacked.decode(bits);
        FloatXUnpacked<nbits, ebits> floatx_unpacked(double_unpacked);
        bits_ = floatx_unpacked.encode();
    }

    FloatX(const FloatXUnpacked<nbits, ebits> &c)
    {
#pragma HLS INLINE off

        bits_ = c.encode();
    }

    template <int in_nbits>
    operator ap_int<in_nbits>() const
    {
        if (bits_ == 0)
            return 0;

        FloatXUnpacked<nbits, ebits> floatx_unpacked;
        floatx_unpacked.decode(bits_);

        return floatx_unpacked;
    }

    operator int() const
    {
        if (bits_ == 0)
            return 0;

        FloatXUnpacked<nbits, ebits> floatx_unpacked;
        floatx_unpacked.decode(bits_);

        return int(floatx_unpacked);
    }

    operator float() const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> floatx_unpacked;
        floatx_unpacked.decode(bits_);
        FloatXUnpacked<32, 8> float_unpacked(floatx_unpacked);
        ap_uint<32> bits = float_unpacked.encode();

        // float fresult = *reinterpret_cast<float *>(&bits);
        // return fresult;
        return bitcast_f32(bits);
    }

    operator double() const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> floatx_unpacked;
        floatx_unpacked.decode(bits_);
        FloatXUnpacked<64, 11> double_unpacked(floatx_unpacked);
        ap_uint<64> bits = double_unpacked.encode();

        // double fresult = *reinterpret_cast<double *>(&bits);
        // return fresult;
        return bitcast_f64(bits);
    }

    operator FloatXUnpacked<nbits, ebits>() const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked;
    }

    FloatXUnpacked<nbits, ebits> operator+(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> in1;
        in1.decode(bits_);
        FloatXUnpacked<nbits, ebits> res = in1 + rhs;
        return res;
    }

    FloatXUnpacked<nbits, ebits> operator-(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> in1;
        in1.decode(bits_);
        FloatXUnpacked<nbits, ebits> res = in1 - rhs;
        return res;
    }

    FloatXUnpacked<nbits, ebits> operator*(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> in1;
        in1.decode(bits_);
        FloatXUnpacked<nbits, ebits> res = in1 * rhs;
        return res;
    }

    FloatXUnpacked<nbits, ebits> operator/(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> in1;
        in1.decode(bits_);
        FloatXUnpacked<nbits, ebits> res = in1 / rhs;
        return res;
    }

    FloatXUnpacked<nbits, ebits> operator-() const
    {
#pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits> res;
        res.decode(bits_);
        return -res;
    }

    FloatX &operator*=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        FloatXUnpacked<nbits, ebits> res;
        res.decode(bits_);
        res = res * rhs;
        bits_ = res.encode();
        return *this;
    }

    FloatX &operator/=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        FloatXUnpacked<nbits, ebits> res;
        res.decode(bits_);
        res = res / rhs;
        bits_ = res.encode();
        return *this;
    }

    FloatX &operator+=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        FloatXUnpacked<nbits, ebits> res;
        res.decode(bits_);
        res = res + rhs;
        bits_ = res.encode();
        return *this;
    }

    FloatX &operator-=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        FloatXUnpacked<nbits, ebits> res;
        res.decode(bits_);
        res = res - rhs;
        bits_ = res.encode();
        return *this;
    }

    bool operator==(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
        FloatXUnpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked == rhs;
    }

    bool operator<(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
        FloatXUnpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked < rhs;
    }

    bool operator>(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
        FloatXUnpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return rhs < unpacked;
    }

    bool operator<=(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
        FloatXUnpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked == rhs || unpacked < rhs;
    }

    bool operator>=(const FloatXUnpacked<nbits, ebits> &rhs) const
    {
        FloatXUnpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked == rhs || rhs < unpacked;
    }

    // private:
    ap_uint<nbits> bits_;
};

template <int nbits, int ebits>
inline FloatX<nbits, ebits> abs(const FloatX<nbits, ebits> &a)
{
    FloatX<nbits, ebits> b = a;
    if (b.bits_[nbits - 1] == 1)
        b.bits_[nbits - 1] == 0;
    return b;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> round(const FloatX<nbits, ebits> &a)
{
    int integer = int(a);
    FloatX<nbits, ebits> diff = a - FloatX<nbits, ebits>(integer);
    if (diff > FloatX<nbits, ebits>(0.5))
        integer++;
    return FloatX<nbits, ebits>(integer);
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> floor(const FloatX<nbits, ebits> &a)
{
    int integer = int(a);
    return FloatX<nbits, ebits>(integer);
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> ceil(const FloatXUnpacked<nbits, ebits> &a)
{
    int integer = int(a);
    FloatXUnpacked<nbits, ebits> diff = a - FloatXUnpacked<nbits, ebits>(integer);
    if (diff > FloatXUnpacked<nbits, ebits>(0))
        integer++;
    return FloatX<nbits, ebits>(integer);
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> mod(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    FloatX<nbits, ebits> c = a / b;
    FloatX<nbits, ebits> d = a - floor(c) * b;
    return b;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> exp(const FloatX<nbits, ebits> &a)
{
    return a;
}
