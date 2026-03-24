#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

#include "common.h"

template <int ebits, int fbits>
class FloatXUnpacked
{
public:
    static const int nbits = ebits + fbits + 1;

    FloatXUnpacked()
    {
    }

    // FloatXUnpacked(const FloatXUnpacked &other)
    //{
    //     // #pragma HLS INLINE
    //
    //     bits_ = other.bits_;
    // }

    template <int oebits, int ofbits>
    FloatXUnpacked(const FloatXUnpacked<oebits, ofbits> &other)
    {
        // #pragma HLS INLINE

        ap_int<oebits + 1> oexp = (ap_int<oebits + 1>)(ap_uint<1>(0), other.exp_) - (ap_int<oebits + 1>)fbias<oebits>::value;
        ap_int<ebits + 1> exp = oexp + (ap_int<ebits + 1>)fbias<ebits>::value;

        zero_ = other.zero_;
        inf_ = other.inf_;
        sign_ = other.sign_;
        exp_ = exp(ebits - 1, 0);
        if (fbits >= ofbits)
        {
            // no rounding needed
            mant_(fbits - 1, fbits - ofbits) = other.mant_(ofbits - 1, 0);
            mant_(fbits - ofbits - 1, 0) = 0;
        }
        else
        {
            // round as well
            // mant_(fbits - 1, 0) = other.mant_(ofbits - 1, ofbits - fbits);

            ap_uint<fbits + 2> rfrac = (ap_uint<1>(0), other.mant_(ofbits - 1, ofbits - fbits - 1));
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
        // #pragma HLS INLINE

        if (bits == 0)
            zero_ = 1;
        else
            zero_ = 0;

        sign_ = bits[nbits - 1];
        exp_ = bits(nbits - 2, fbits);
        mant_ = bits(fbits - 1, 0);

        if (exp_ == ap_uint<ebits>(-1) && mant_ == 0)
            inf_ = 1;
        else
            inf_ = 0;
    }

    ap_uint<nbits> encode() const
    {
        // #pragma HLS INLINE

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
        FloatXUnpacked out;

        // #pragma HLS INLINE
        if (zero_ == 1)
            return rhs;
        if (rhs.zero_ == 1)
            return *this;

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

        bool sign;
        bool zero;
        ap_uint<ebits> exp;

        ap_int<ebits + 1> diff_texp = (ap_int<ebits + 1>)(ap_uint<1>(0), exp_) - (ap_int<ebits + 1>)(ap_uint<1>(0), rhs.exp_);

        ap_int<fbits + 3> frac1 = (ap_uint<2>(0b01), mant_, ap_uint<1>(0));
        ap_int<fbits + 3> frac2 = (ap_uint<2>(0b01), rhs.mant_, ap_uint<1>(0));

        bool lhs_ge_rhs =
            (diff_texp > 0) ||
            (diff_texp == 0 && frac1 >= frac2);

        if (lhs_ge_rhs)
        {
            sign = sign_;
            exp = exp_;
            frac2 = frac2 >> diff_texp;
            if (sign_ != rhs.sign_)
                frac2 = -frac2;
        }
        else
        {
            sign = rhs.sign_;
            exp = rhs.exp_;
            frac1 = frac1 >> -diff_texp;
            if (sign_ != rhs.sign_)
                frac1 = -frac1;
        }

        // do addition (result is sure to be positive)
        ap_int<fbits + 4> frac = frac1 + frac2;
        // #pragma HLS BIND_OP variable = frac op = add impl = dsp latency = -1

        if (frac == 0)
        {
            zero = 1;
        }
        else
        {
            zero = 0;
        }

        // normalize
        int shift = count_leading_zeros(frac) - 2;
        if (shift > 0)
        {
            frac = frac << shift;
            exp = exp - shift;
        }
        if (shift < 0)
        {
            frac = frac >> -shift;
            exp = exp + -shift;
        }

        // round
        /*
        if (frac[0] == 1)
            frac += 1;

        if (frac[fbits + 2] == 1)
        {
            frac = frac >> 1;
            exp++;
        }
        */

        out.zero_ = zero;
        out.sign_ = sign;
        out.exp_ = exp;
        out.mant_ = frac(fbits, 1);

        return out;
    }

    FloatXUnpacked
    operator-(const FloatXUnpacked &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked in2;
        in2.sign_ = !rhs.sign_;
        in2.exp_ = rhs.exp_;
        in2.mant_ = rhs.mant_;

        FloatXUnpacked result = (*this) + in2;
        return result;
    }

    FloatXUnpacked operator*(const FloatXUnpacked &rhs) const
    {
        // #pragma HLS INLINE

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
        // #pragma HLS INLINE

        FloatXUnpacked out;

        out.zero_ = zero_;
        out.inf_ = rhs.zero_;
        out.sign_ = sign_ ^ rhs.sign_;

        ap_uint<ebits + 1> exp = exp_ - rhs.exp_;
        ap_uint<fbits * 2 + 1> frac = mant_ / rhs.mant_;

        // normalize
        if (frac < 1)
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

        out.exp_ = exp;
        out.mant_ = frac;

        return out;
    }

    FloatXUnpacked operator-() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked result;
        result.sign_ = !sign_;
        result.exp_ = exp_;
        result.mant_ = mant_;

        return result;
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
        // #pragma HLS INLINE

        bits_ = other.bits_;
    }

    FloatX &operator=(const FloatX &other)
    {
        // #pragma HLS INLINE

        // if (this != &other)
        {
            bits_ = other.bits_;
        }
        return *this;
    }

    FloatX(float c)
    {
        // #pragma HLS INLINE

        // ap_uint<32> bits = *reinterpret_cast<ap_uint<32> *>(&c);
        ap_uint<32> bits = bitcast_u32(c);
        FloatXUnpacked<8, 23> float_unpacked;
        float_unpacked.decode(bits);
        FloatXUnpacked<ebits, fbits> floatx_unpacked(float_unpacked);
        bits_ = floatx_unpacked.encode();
    }

    FloatX(double c)
    {
        // #pragma HLS INLINE

        // ap_uint<64> bits = *reinterpret_cast<ap_uint<64> *>(&c);
        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<11, 52> double_unpacked;
        double_unpacked.decode(bits);
        FloatXUnpacked<ebits, fbits> floatx_unpacked(double_unpacked);
        bits_ = floatx_unpacked.encode();
    }

    FloatX(const FloatXUnpacked<ebits, fbits> &c)
    {
        // #pragma HLS INLINE

        bits_ = c.encode();
    }

    operator float() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> floatx_unpacked;
        floatx_unpacked.decode(bits_);
        FloatXUnpacked<8, 23> float_unpacked(floatx_unpacked);
        ap_uint<32> bits = float_unpacked.encode();

        // float fresult = *reinterpret_cast<float *>(&bits);
        // return fresult;
        return bitcast_f32(bits);
    }

    operator double() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> floatx_unpacked;
        floatx_unpacked.decode(bits_);
        FloatXUnpacked<11, 52> double_unpacked(floatx_unpacked);
        ap_uint<64> bits = double_unpacked.encode();

        // double fresult = *reinterpret_cast<double *>(&bits);
        // return fresult;
        return bitcast_f64(bits);
    }

    operator FloatXUnpacked<ebits, fbits>() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> unpacked;
        unpacked.decode(bits_);
        return unpacked;
    }

    FloatXUnpacked<ebits, fbits> operator+(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.decode(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 + rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator-(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.decode(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 - rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator*(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.decode(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 * rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator/(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.decode(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 / rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator-() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> res;
        res.decode(bits_);
        return -res;
    }

private:
    ap_uint<nbits> bits_;
};