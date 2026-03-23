#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

#include "common.h"

template <int ebits, int fbits>
class FloatXUnpacked
{
public:
    template <int in_nbits, int in_ebits>
    void decode(const ap_uint<in_nbits> &bits)
    {
        // #pragma HLS INLINE

        if (bits == 0)
        {
            sign_ = 0;
            exp_ = 0;
            frac_ = 0;
            return;
        }

        sign_ = bits[in_nbits - 1];

        // exponent bits
        ap_uint<in_ebits> exp = bits(in_nbits - 2, in_nbits - 1 - in_ebits);
        exp_ = exp - (ap_int<in_ebits + 1>)fbias<in_ebits>::value;

        ap_ufixed<in_nbits - in_ebits, 1> frac;
        frac[in_nbits - in_ebits - 1] = 1;
        frac(in_nbits - in_ebits - 2, 0) = bits(in_nbits - 2 - in_ebits, 0);

        ap_ufixed<in_nbits - in_ebits, 1> rfrac;
        // if (in_nbits - in_ebits - 1 > fbits)
        //     rfrac = round_to(frac, fbits - 1);
        // else
        rfrac = frac;

        // fraction bits
        // add leading 1
        frac_ = rfrac;
    }

    template <int in_nbits, int in_ebits>
    ap_uint<in_nbits> encode() const
    {
        // #pragma HLS INLINE

        if (frac_ == 0)
            return 0;

        ap_uint<in_nbits> bits;

        ap_int<in_ebits + 1> exp;
        // if (frac_ == 0)
        //     exp = 0;
        // else
        //  exp = unpacked.exp + hls::pow(2, es - 1) - 1;
        exp = exp_ + (ap_int<in_ebits + 1>)fbias<in_ebits>::value;

        ap_ufixed<fbits + 1, 1> rfrac;
        if (fbits > in_nbits - in_ebits - 1)
            rfrac = round_to(frac_, in_nbits - in_ebits - 2);
        else
            rfrac = frac_;

        ap_ufixed<in_nbits - in_ebits, 1> frac;
        frac = rfrac;

        bits[in_nbits - 1] = sign_;
        bits(in_nbits - 2, in_nbits - 1 - in_ebits) = exp(in_ebits - 1, 0);
        bits(in_nbits - 2 - in_ebits, 0) = frac(in_nbits - 2 - in_ebits, 0);

        return bits;
    }

    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
        // #pragma HLS INLINE
        if (frac_ == 0)
            return rhs;
        if (rhs.frac_ == 0)
            return *this;

        // set biggest posit to be in1
        ap_int<ebits + 1> diff_texp = exp_ - rhs.exp_;

        ap_fixed<fbits + 2, 2> frac1 = frac_;
        ap_fixed<fbits + 2, 2> frac2 = rhs.frac_;

        ap_uint<ebits + 1> exp;
        bool sign;

        if (diff_texp >= 0)
        {
            exp = exp_;
            frac2 = frac2 >> diff_texp;

            // sign of output is sign of largest number
            sign = sign_;
            // check if we have to add or substract
            if (sign_ != rhs.sign_)
                frac2 = -frac2;
        }
        else
        {
            exp = rhs.exp_;
            frac1 = frac1 >> -diff_texp;

            // sign of output is sign of largest number
            sign = rhs.sign_;
            // check if we have to add or substract
            if (sign_ != rhs.sign_)
                frac1 = -frac1;
        }

        // do addition (result is sure to be positive)
        ap_ufixed<fbits + 3, 3> frac = frac1 + frac2;
        // #pragma HLS BIND_OP variable = frac op = add impl = dsp latency = -1

        // normalize
        if (frac == 0)
        {
            sign = 0;
            exp = 0;
        }
        else
        {
            int shift = count_leading_zeros(frac) - 2;
            if (shift > 0)
            {
                frac = frac << shift;
                exp = exp - shift;
            }
            else if (shift < 0)
            {
                frac = frac >> -shift;
                exp = exp + -shift;
            }
        }

        ap_ufixed<fbits + 3, 3> rfrac = round_to(frac, fbits - 1);

        if (rfrac >= 2)
        {
            rfrac = rfrac >> 1;
            exp++;
        }

        FloatXUnpacked out;
        out.sign_ = sign;
        out.exp_ = exp;
        out.frac_ = rfrac;

        return out;
    }

    FloatXUnpacked operator-(const FloatXUnpacked &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked in2;
        in2.sign_ = !rhs.sign_;
        in2.exp_ = rhs.exp_;
        in2.frac_ = rhs.frac_;

        FloatXUnpacked result = (*this) + in2;
        return result;
    }

    FloatXUnpacked operator*(const FloatXUnpacked &rhs) const
    {
        // #pragma HLS INLINE

        bool sign = sign_ ^ rhs.sign_;
        ap_int<ebits + 2> exp = exp_ + rhs.exp_;
        ap_ufixed<fbits * 2 + 2, 2> frac = frac_ * rhs.frac_;

        if (frac == 0)
        {
            sign = 0;
            exp = 0;
        }

        // normalize
        if (frac >= 2)
        {
            frac = frac >> 1;
            exp++;
        }

        // round
        ap_ufixed<fbits * 2 + 2, 2> rfrac = round_to(frac, fbits - 1);

        // normalize (again)
        if (rfrac >= 2)
        {
            rfrac = rfrac >> 1;
            exp++;
        }
        /*
        if (rfrac == 0)
        {
            exp = 0;
            rfrac = 0;
        }
        */

        FloatXUnpacked out;
        out.sign_ = sign;
        out.exp_ = exp;
        out.frac_ = rfrac;

        return out;
    }

    FloatXUnpacked operator/(const FloatXUnpacked &rhs) const
    {
        // #pragma HLS INLINE

        bool sign = sign_ ^ rhs.sign_;
        ap_int<ebits + 1> exp = exp_ - rhs.exp_;
        ap_ufixed<fbits * 2 + 1, 1> frac = 0;

        if (rhs.frac_ != 0)
            frac = frac_ / rhs.frac_;

        if (frac == 0)
        {
            sign = 0;
            exp = 0;
        }

        // normalize
        if (frac < 1)
        {
            frac = frac << 1;
            exp--;
        }

        ap_ufixed<fbits * 2 + 1, 1> rfrac = round_to(frac, fbits - 1);

        // normalize
        if (rfrac < 1)
        {
            rfrac = rfrac << 1;
            exp--;
        }

        if (rfrac == 0)
        {
            exp = 0;
            frac = 0;
        }

        FloatXUnpacked out;
        out.sign_ = sign;
        out.exp_ = exp;
        out.frac_ = rfrac;

        return out;
    }

    FloatXUnpacked operator-() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked result;
        result.sign_ = !sign_;
        result.exp_ = exp_;
        result.frac_ = frac_;

        return result;
    }

    bool sign_;
    ap_int<ebits + 1> exp_;
    ap_ufixed<fbits + 1, 1> frac_;
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
        FloatXUnpacked<8, 23> unpacked;
        unpacked.template decode<32, 8>(bits);
        bits_ = unpacked.template encode<nbits, ebits>();
    }

    FloatX(double c)
    {
        // #pragma HLS INLINE

        // ap_uint<64> bits = *reinterpret_cast<ap_uint<64> *>(&c);
        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<11, 52> unpacked;
        unpacked.template decode<64, 11>(bits);
        bits_ = unpacked.template encode<nbits, ebits>();
    }

    FloatX(const FloatXUnpacked<ebits, fbits> &c)
    {
        // #pragma HLS INLINE

        bits_ = c.template encode<nbits, ebits>();
    }

    operator float() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<8, 23> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);
        ap_uint<32> bits = unpacked.template encode<32, 8>();

        // float fresult = *reinterpret_cast<float *>(&bits);
        // return fresult;
        return bitcast_f32(bits);
    }

    operator double() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<11, 52> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);
        ap_uint<64> bits = unpacked.template encode<64, 11>();

        // double fresult = *reinterpret_cast<double *>(&bits);
        // return fresult;
        return bitcast_f64(bits);
    }

    operator FloatXUnpacked<ebits, fbits>() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);
        return unpacked;
    }

    FloatXUnpacked<ebits, fbits> operator+(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 + rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator-(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 - rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator*(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 * rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator/(const FloatXUnpacked<ebits, fbits> &rhs) const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        FloatXUnpacked<ebits, fbits> res = in1 / rhs;
        return res;
    }

    FloatXUnpacked<ebits, fbits> operator-() const
    {
        // #pragma HLS INLINE

        FloatXUnpacked<ebits, fbits> res;
        res.template decode<nbits, ebits>(bits_);
        return -res;
    }

private:
    ap_uint<nbits> bits_;
};