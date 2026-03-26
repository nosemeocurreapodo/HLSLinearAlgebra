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
#pragma HLS INLINE off

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
#pragma HLS INLINE off

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

        if (zero_ == 1)
            return rhs;
        if (rhs.zero_ == 1)
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
        int shift = count_leading_zeros(frac) - 1;

        if (shift > 0)
        {
            frac = frac << shift;
        }
        if (shift < 0)
        {
            frac = frac >> -shift;
        }

        in1.exp_ = in1.exp_ - shift;

        // round

        // if (frac[0] == 1)
        //     frac += 1;

        // if (frac[fbits + 2] == 1)
        //{
        //     frac = frac >> 1;
        //     exp++;
        // }

        in1.mant_ = frac(fbits, 1);

        return in1;
    }

    /*
    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE off

        FloatXUnpacked out;

        if (zero_)
            return rhs;
        if (rhs.zero_)
            return *this;

        if (inf_ && rhs.inf_)
        {
            if (sign_ != rhs.sign_)
            {
                out.zero_ = 1;
                out.inf_ = 0;
                out.sign_ = 0;
                out.exp_ = 0;
                out.mant_ = 0;
                return out;
            }
            return *this;
        }
        if (inf_)
            return *this;
        if (rhs.inf_)
            return rhs;

        // ------------------------------------------------------------
        // Compare magnitudes once
        // ------------------------------------------------------------
        ap_int<ebits + 1> diff_texp =
            (ap_int<ebits + 1>)(ap_uint<1>(0), exp_) -
            (ap_int<ebits + 1>)(ap_uint<1>(0), rhs.exp_);

        bool lhs_ge_rhs =
            (diff_texp > 0) ||
            ((diff_texp == 0) && (mant_ >= rhs.mant_));

        ap_uint<ebits> exp_big = lhs_ge_rhs ? exp_ : rhs.exp_;
        ap_uint<ebits> exp_small = lhs_ge_rhs ? rhs.exp_ : exp_;

        ap_uint<fbits> mant_big = lhs_ge_rhs ? mant_ : rhs.mant_;
        ap_uint<fbits> mant_small = lhs_ge_rhs ? rhs.mant_ : mant_;

        bool sign_big = lhs_ge_rhs ? sign_ : rhs.sign_;
        bool sign_small = lhs_ge_rhs ? rhs.sign_ : sign_;

        // ------------------------------------------------------------
        // Rebuild significands as UNSIGNED
        // format: 1.mant with one guard LSB => [1][mant][0]
        // ------------------------------------------------------------
        ap_uint<fbits + 3> frac_big = (ap_uint<2>(0b01), mant_big, ap_uint<1>(0));
        ap_uint<fbits + 3> frac_small = (ap_uint<2>(0b01), mant_small, ap_uint<1>(0));

        // ------------------------------------------------------------
        // Align only the smaller operand
        // Clamp shift to reduce shifter size
        // ------------------------------------------------------------
        ap_uint<ebits + 1> diff_abs = lhs_ge_rhs ? ap_uint<ebits + 1>(diff_texp)
                                                 : ap_uint<ebits + 1>(-diff_texp);

        const int MAX_SHIFT = fbits + 3;
        ap_uint<ebits + 1> sh = (diff_abs > MAX_SHIFT) ? ap_uint<ebits + 1>(MAX_SHIFT)
                                                       : diff_abs;

        frac_small = frac_small >> sh;

        // ------------------------------------------------------------
        // Same sign -> add
        // Different sign -> subtract
        // Since big >= small, subtraction stays nonnegative
        // ------------------------------------------------------------
        ap_uint<fbits + 4> frac;
        ap_uint<ebits + 1> exp = exp_big;
        bool sign = sign_big;

        bool same_sign = (sign_big == sign_small);

        if (same_sign)
        {
            frac = (ap_uint<fbits + 4>)frac_big + (ap_uint<fbits + 4>)frac_small;

            // same-sign addition only needs possible 1-bit right normalization
            if (frac[fbits + 3])
            {
                frac = frac >> 1;
                exp = exp + 1;
            }
        }
        else
        {
            frac = (ap_uint<fbits + 4>)frac_big - (ap_uint<fbits + 4>)frac_small;

            if (frac == 0)
            {
                out.zero_ = 1;
                out.inf_ = 0;
                out.sign_ = 0;
                out.exp_ = 0;
                out.mant_ = 0;
                return out;
            }

            // subtraction may need left normalization
            int lz = 0;
            bool found = false;
            for (int i = fbits + 2; i >= 0; --i)
            {
#pragma HLS UNROLL
                if (!found && frac[i])
                {
                    lz = (fbits + 2) - i;
                    found = true;
                }
            }

            if (lz > 0)
            {
                frac = frac << lz;
                exp = exp - lz;
            }
        }

        // ------------------------------------------------------------
        // Optional: cheap rounding using guard bit frac[0]
        // ------------------------------------------------------------

        // if (frac[0])
        //     frac = frac + 2; // bump LSB of stored mantissa region

        // if (frac[fbits + 3])
        //{
        //     frac >>= 1;
        //     exp += 1;
        // }

        // ------------------------------------------------------------
        // Handle underflow / overflow if desired
        // no subnormals here
        // ------------------------------------------------------------
        if (exp == 0)
        {
            out.zero_ = 1;
            out.inf_ = 0;
            out.sign_ = 0;
            out.exp_ = 0;
            out.mant_ = 0;
            return out;
        }

        if (exp == ap_uint<ebits>(-1))
        {
            out.zero_ = 0;
            out.inf_ = 1;
            out.sign_ = sign;
            out.exp_ = ap_uint<ebits>(-1);
            out.mant_ = 0;
            return out;
        }

        out.zero_ = 0;
        out.inf_ = 0;
        out.sign_ = sign;
        out.exp_ = exp;
        out.mant_ = frac(fbits, 1);

        return out;
    }
    */

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
#pragma HLS INLINE off

        // ap_uint<32> bits = *reinterpret_cast<ap_uint<32> *>(&c);
        ap_uint<32> bits = bitcast_u32(c);
        FloatXUnpacked<8, 23> float_unpacked;
        float_unpacked.decode(bits);
        FloatXUnpacked<ebits, fbits> floatx_unpacked(float_unpacked);
        bits_ = floatx_unpacked.encode();
    }

    FloatX(double c)
    {
#pragma HLS INLINE off

        // ap_uint<64> bits = *reinterpret_cast<ap_uint<64> *>(&c);
        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<11, 52> double_unpacked;
        double_unpacked.decode(bits);
        FloatXUnpacked<ebits, fbits> floatx_unpacked(double_unpacked);
        bits_ = floatx_unpacked.encode();
    }

    FloatX(const FloatXUnpacked<ebits, fbits> &c)
    {
#pragma HLS INLINE off

        bits_ = c.encode();
    }

    operator float() const
    {
#pragma HLS INLINE off

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
#pragma HLS INLINE off

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
#pragma HLS INLINE off

        FloatXUnpacked<ebits, fbits> unpacked;
        unpacked.decode(bits_);
        return unpacked;
    }

    FloatXUnpacked<ebits, fbits> operator+(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
#pragma HLS INLINE off

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