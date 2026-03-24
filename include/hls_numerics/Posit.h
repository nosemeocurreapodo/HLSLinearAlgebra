#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

#include "common.h"
#include "FloatX.h"

namespace detail
{
    // 2^k for *integer* k
    constexpr float pow2(int k)
    {
        // #pragma HLS INLINE
        return k >= 0 ? (1u << k) : 1.0f / (1u << (-k));
    }

    // Compile‑time ceil(log2(N))
    // Usage: clog2<17>::value == 5 (because 2^4 < 17 ≤ 2^5)
    // Works for N≥1.
    template <int N, bool done = (N <= 1)>
    struct clog2_helper
    {
        static constexpr int value = 1 + clog2_helper<(N >> 1)>::value;
    };
    template <int N>
    struct clog2_helper<N, true>
    {
        static constexpr int value = 0;
    };

    template <int N>
    struct clog2
    {
        static constexpr int value = clog2_helper<N>::value;
    };

} // namespace detail

template <int nbits, int ebits>
class posit_unpacked
{
public:
    static const int m_kbits = detail::clog2<nbits>::value + 1;
    static const int m_fbits = nbits - ebits - 3;
    static constexpr int large_k = m_kbits + ebits + m_fbits;

    // posit_unpacked()
    //{
    //  #pragma HLS allocation function instances = decode < kbits, ebits, fbits> limit = 1
    //}

    ap_uint<nbits> encode(const posit_unpacked &unpacked)
    {
        // #pragma HLS INLINE off      // <- do NOT inline this hardware
        // #pragma HLS PIPELINE II = 1 // pipeline so a single instance can accept 1/cycle

        // #pragma HLS INLINE
        //     #pragma HLS PIPELINE off
        ap_uint<nbits> bits;

        bool simbol;
        int reg_len;

        if (unpacked.frac_ == 0)
        {
            simbol = 0;
            reg_len = nbits - 1;
        }
        else
        {
            simbol = unpacked.k_ >= 0 ? 0 : 1;
            reg_len = unpacked.k_ >= 0 ? int(unpacked.k_ + 1) : int(-unpacked.k_);
        }

        bits[nbits - 1] = unpacked.sign_;
        bits[nbits - 2] = simbol;

        enum states
        {
            K,
            E,
            F
        } state = K;

        ap_uint<m_kbits> last_state_bit = nbits - 2;
    Posit_encode_for:
        for (int bit = nbits - 3; bit >= 0; bit--)
        {
            // #pragma HLS UNROLL
            int dist = last_state_bit - bit;
            if (state == K)
            {
                // ap_int<kbits> lenght = last_state_bit - bit;

                if (dist == reg_len)
                {
                    bits[bit] = !simbol;
                    last_state_bit = bit;
                    state = E;
                }
                else
                {
                    bits[bit] = simbol;
                }
                continue;
            }

            if (state == E)
            {
                // unpacked.exp[es - 1 - counter] = bits[bit];
                int pos = ebits - dist;
                bits[bit] = unpacked.exp_[pos];

                if (pos == 0)
                {
                    last_state_bit = bit;
                    state = F;
                }
                continue;
            }

            if (state == F)
            {
                int pos = m_fbits - dist;
                bits[bit] = unpacked.frac_[pos];
                continue;
            }
        }

        return bits;
    }

    posit_unpacked decode(const ap_uint<nbits> &bits)
    {
        // #pragma HLS INLINE off      // <- do NOT inline this hardware
        // #pragma HLS PIPELINE II = 1 // pipeline so a single instance can accept 1/cycle

        posit_unpacked unpacked;

        unpacked.sign_ = bits[nbits - 1];
        bool simbol = bits[nbits - 2];

        // ap_int<kbits> k;
        // ap_uint<ebits> exp;
        // ap_ufixed<fbits + 1, 1> frac;

        // start as zero or inf depending on first bit
        if (bits[nbits - 1])
        {
            // for inf, start k as a large number
            unpacked.k_ = unpacked.large_k;
            unpacked.exp_ = 0;
            unpacked.frac_ = 1.0;
        }
        else
        {
            unpacked.k_ = 0;
            unpacked.exp_ = 0;
            unpacked.frac_ = 0.0;
        }

        enum states
        {
            K,
            E,
            F
        } state = K;

        ap_uint<m_kbits> last_state_bit = nbits - 2;
    Posit_decode_for:
        for (int bit = nbits - 3; bit >= 0; bit--)
        {
            // #pragma HLS UNROLL

            int dist = last_state_bit - bit;
            if (state == K)
            {
                if (bits[bit] != simbol)
                {
                    // ap_int<kbits> lenght = last_state_bit - bit;
                    //  counter = nbits - 3 - bit;

                    if (simbol == 0)
                        unpacked.k_ = dist - 1;
                    else
                        unpacked.k_ = -dist;

                    last_state_bit = bit;

                    state = E;
                }
                continue;
            }

            if (state == E)
            {
                // unpacked.exp[es - 1 - counter] = bits[bit];
                int pos = ebits - dist;
                unpacked.exp_[pos] = bits[bit];

                if (pos == 0)
                {
                    last_state_bit = bit;
                    unpacked.frac_[m_fbits] = 1;
                    state = F;
                }
                continue;
            }

            if (state == F)
            {
                int pos = m_fbits - dist;
                unpacked.frac_[pos] = bits[bit];
                continue;
            }
        }

        return unpacked;
    }

    int getTotalExp() const
    {
        // #pragma HLS INLINE

        // #pragma HLS INLINE
        // get exponent from k and e
        return k_ * (1 << ebits) + exp_;
    }

    void setKEFromTotalExp(int in_exp)
    {
        // #pragma HLS INLINE

        // get k and e from floating point exponent
        // How many times each exponent over - or under - flowed the valid interval
        int k = in_exp / (1 << ebits); // works for negatives too

        // New exponent in the allowed range (0, max_exp_val - 1)
        // exp_ = in_exp % (1 << ebits);
        ap_int<ebits + 1> exp = in_exp - k * (1 << ebits);

        // If `a` and `b` have opposite signs and the remainder is non-zero,
        // the truncated result is too large; subtract one to get the floor.
        // if (exp_ < 0)
        //{
        //    --k_;
        // exp = 0;
        //}

        if (exp < 0)
        {
            --k;
            exp += (1 << ebits);
        }

        k_ = k;
        exp_ = exp;
    }

    posit_unpacked operator+(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE

        // int exp1 = getTotalExp();
        // int exp2 = rhs.getTotalExp();

        // set biggest posit to be in1
        // int diff_texp = exp1 - exp2;

        ap_int<m_kbits + ebits> diff_texp = (k_ - rhs.k_) * (1 << ebits) + exp_ - rhs.exp_;

        ap_fixed<m_fbits + 2, 2> frac1 = frac_;
        ap_fixed<m_fbits + 2, 2> frac2 = rhs.frac_;

        ap_int<ebits + 2> exp;
        ap_int<m_kbits> k;
        bool sign;

        if (diff_texp >= 0)
        {
            sign = sign_;
            exp = exp_;
            k = k_;

            frac2 = frac2 >> diff_texp;

            if (sign_ != rhs.sign_)
                frac2 = -frac2;
        }
        else
        {
            sign = rhs.sign_;
            exp = rhs.exp_;
            k = rhs.k_;

            frac1 = frac1 >> -diff_texp;

            if (sign_ != rhs.sign_)
                frac1 = -frac1;
        }

        // do addition (always positive)
        ap_ufixed<m_fbits + 3, 3> frac = frac1 + frac2;

        // normalize
        if (frac == 0)
        {
            sign = 0;
            k = 0;
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

        // normalize exponent
        if (exp >= (1 << ebits))
        {
            exp -= (1 << ebits);
            k++;
        }

        if (exp < 0)
        {
            exp += (1 << ebits);
            k--;
        }

        ap_ufixed<m_fbits + 3, 3> rfrac = round_to(frac, m_fbits - 1);

        // normalize fraction (again)
        if (rfrac >= 2)
        {
            rfrac = rfrac >> 1;
            exp++;
        }

        // normalize exponent
        if (exp >= (1 << ebits))
        {
            exp -= (1 << ebits);
            k++;
        }

        if (exp < 0)
        {
            exp += (1 << ebits);
            k--;
        }

        posit_unpacked out;

        // out.setKEFromTotalExp(exp);
        out.sign_ = sign;
        out.k_ = k;
        out.exp_ = exp;
        out.frac_ = rfrac;

        return out;
    }

    posit_unpacked operator-(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked in2;
        in2.sign_ = !rhs.sign_;
        in2.k_ = rhs.k_;
        in2.exp_ = rhs.exp_;
        in2.frac_ = rhs.frac_;

        posit_unpacked out = (*this) + in2;

        return out;
    }

    posit_unpacked operator/(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE

        bool sign;
        ap_int<m_kbits> k;
        ap_int<ebits + 1> exp;
        ap_ufixed<m_fbits * 2, 2> frac;

        // set inf
        if (rhs.frac_ == 0)
        {
            sign = 0;
            k = large_k;
            exp = 0;
            frac = 0;
        }
        else
        {
            sign = sign_ ^ rhs.sign_;
            k = k_ - rhs.k_;
            exp = exp_ - rhs.exp_;
            frac = frac_ / rhs.frac_;
        }

        // normalize fraction
        if (frac < 1.0)
        {
            frac = frac << 1;
            exp--;
        }

        // normalize exponent
        if (exp < 0)
        {
            exp += (1 << ebits);
            k--;
        }

        /*
        if (exp < 0)
        {
            exp += hls::pow(2, es);
            k--;
        }
        */

        ap_ufixed<m_fbits + 1, 2> rfrac = round_to(frac, m_fbits - 1);

        /*
        // round to nearest
        ap_ufixed<fbits + 1, 2> rfrac = frac;
        if (frac[fbits * 2 - 3 - (fbits - 1)] == 1)
        {
            ap_ufixed<fbits, 1> one = 0;
            one[0] = 1;
            // if(pmantissa[0] == 0)
            rfrac += one;
            // pmantissa[0] = 1;
        }
        */

        // normalize fraction (again)
        if (rfrac >= 2)
        {
            rfrac = rfrac >> 1;
            exp++;
        }

        // normalize exponent (again)
        if (exp >= (1 << ebits))
        {
            exp -= (1 << ebits);
            k++;
        }

        if (rfrac == 0)
        {
            sign = 0;
            k = 0;
            exp = 0;
        }

        posit_unpacked out;
        out.sign_ = sign;
        out.k_ = k;
        out.exp_ = exp;
        out.frac_ = rfrac;

        return out;
    }

    posit_unpacked operator-() const
    {
        // #pragma HLS INLINE

        posit_unpacked result;
        result.sign_ = !sign_;
        result.exp_ = exp_;
        result.k_ = k_;
        result.frac_ = frac_;

        return result;
    }

    bool sign_;
    // the max amount of bits for r is nbits-1 bits, nbits-2 bits beeing 0 (or 1), and the last beeing 1 (or 0)
    // k is the amount of counted bits
    // which can be stored in log2(nbits - 2) bits
    ap_int<m_kbits> k_;
    ap_uint<ebits> exp_;
    // the max amount of bits for frac is nbits - 1 (sign) - 2 (min bits for k) - es;
    ap_ufixed<m_fbits + 1, 1> frac_;
};

template <int nbits, int ebits>
bool posit_equal(const posit_unpacked<nbits, ebits> &in1, const posit_unpacked<nbits, ebits> &in2)
{
    if (in1.sign != in2.sign)
    {
        return false;
    }

    if (in1.k != in2.k)
    {
        return false;
    }

    if (in1.exp != in2.exp)
    {
        return false;
    }

    if (in1.frac != in2.frac)
    {
        return false;
    }

    return true;
}

template <int nbits, int ebits>
bool posit_lessthan(const posit_unpacked<nbits, ebits> &in1, const posit_unpacked<nbits, ebits> &in2)
{
    if (in1.sign != in2.sign)
    {
        return in1.sign;
    }

    if (in1.frac == 0 || in2.frac == 0)
    {
        return in1.sign != (in1.frac < in2.frac);
    }

    if (in1.k != in2.k)
    {
        return in1.sign != (in1.k < in2.k);
    }

    if (in1.exp != in2.exp)
    {
        return in1.sign != (in1.exp < in2.exp);
    }

    if (in1.frac != in2.frac)
    {
        return in1.sign != (in1.frac < in2.frac);
    }

    return false;
}

template <int nbits, int ebits>
bool posit_lesseqthan(const posit_unpacked<nbits, ebits> &in1, const posit_unpacked<nbits, ebits> &in2)
{
    return posit_equal(in1, in2) || posit_lessthan(in1, in2);
}

template <int nbits, int ebits>
bool posit_morethan(const posit_unpacked<nbits, ebits> &in1, const posit_unpacked<nbits, ebits> &in2)
{
    return posit_lessthan(in2, in1);
}

template <int nbits, int ebits>
bool posit_moreeqthan(const posit_unpacked<nbits, ebits> &in1, const posit_unpacked<nbits, ebits> &in2)
{
    return posit_equal(in1, in2) || posit_morethan(in1, in2);
}

template <int nbits, int ebits>
posit_unpacked<nbits, ebits> posit_fabs(const posit_unpacked<nbits, ebits> &in1)
{
    posit_unpacked<nbits, ebits> result = in1;

    result.sign = 0; // set sign to 0

    return result;
}

template <int nbits, int ebits>
posit_unpacked<nbits, ebits> posit_floor(const posit_unpacked<nbits, ebits> &in1)
{
    int exp = in1.getTotalExp();

    auto frac = in1.frac;

    frac = frac << exp;
    /*
    if (in1.sign && frac(fbits - 1, 0) != 0)
    {
        frac(fbits - 1, 0) = 0;
        frac = frac + 1;
    }
    else
    {
        frac(fbits - 1, 0) = 0;
    }
    */

    frac = frac >> exp;

    while (frac >= 2)
    {
        frac = frac >> 1;
        exp = exp + 1;
    }

    posit_unpacked<nbits, ebits> result;

    result.sign = in1.sign;
    result.frac = frac;
    result.setKEFromTotalExp(exp);
    return result;
}

template <int nbits, int ebits>
posit_unpacked<nbits, ebits> posit_round(const posit_unpacked<nbits, ebits> &in1)
{
    int exp = in1.getTotalExp();

    auto frac = in1.frac;

    frac = frac << exp;
    /*
    if (frac[fbits - 1] != 0)
    {
        frac(fbits - 1, 0) = 0;
        frac = frac + 1;
    }
    else
    {
        frac(fbits - 1, 0) = 0;
    }
    */

    frac = frac >> exp;

    while (frac >= 2)
    {
        frac = frac >> 1;
        exp = exp + 1;
    }

    posit_unpacked<nbits, ebits> result;

    result.sign = in1.sign;
    result.frac = frac;
    result.setKEFromTotalExp(exp);
    return result;
}

template <int nbits, int ebits>
posit_unpacked<nbits, ebits> posit_ceil(const posit_unpacked<nbits, ebits> &in1)
{
    int exp = in1.getTotalExp();

    auto frac = in1.frac;

    frac = frac << exp;
    /*
    if (!in1.sign && frac(fbits - 1, 0) != 0)
    {
        frac(fbits - 1, 0) = 0;
        frac = frac + 1;
    }
    else
    {
        frac(fbits - 1, 0) = 0;
    }
    */

    frac = frac >> exp;

    while (frac >= 2)
    {
        frac = frac >> 1;
        exp = exp + 1;
    }

    posit_unpacked<nbits, ebits> result;

    result.sign = in1.sign;
    result.frac = frac;
    result.setKEFromTotalExp(exp);
    return result;
}

template <int nbits, int ebits>
class Posit
{
public:
    // static constexpr int max_k_size = nbits / 2;
    // static constexpr int k_bit_size = detail::clog2<nbits - 1>::value;
    // static constexpr int fbits = nbits - 3 - ebits + 1;
    // static constexpr int kbits = detail::clog2<nbits>::value + 1;
    // static constexpr int max_count_size = nbits / 2;

    Posit()
    {
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1
    }

    Posit(const Posit &other)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        bits_ = other.bits_;
    }

    Posit &operator=(const Posit &other)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        bits_ = other.bits_;
        /*
        if (this != &other)
        {
            bits_ = other.bits_;
        }
        */
        return *this;
    }

    Posit(int c)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        bool psign = c < 0;

        // mantissa is just zeros
        ap_ufixed<fbits * 2, fbits> pmantissa;
        int fexponent;

        if (c == 0)
        {
            pmantissa = 0.0;
            fexponent = 0;
        }
        else
        {
            // pmantissa[1] = 1.0;
            fexponent = 0;
            pmantissa = hls::abs(c);
            while (pmantissa >= 2.0)
            {
                pmantissa = pmantissa >> 1;
                fexponent++;
            }
        }

        posit_unpacked<kbits, ebits, fbits> unpacked;

        unpacked.setKEFromTotalExp(fexponent);

        unpacked.sign = psign;
        unpacked.frac = pmantissa;

        bits_ = unpacked.template encode<nbits, ebits>();
    }

    Posit(unsigned int c)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        bool psign = c < 0;

        // mantissa is just zeros
        ap_ufixed<fbits * 2, fbits> pmantissa;
        int fexponent;

        if (c == 0)
        {
            pmantissa = 0.0;
            fexponent = 0;
        }
        else
        {
            // pmantissa[1] = 1.0;
            fexponent = 0;
            pmantissa = hls::abs(c);
            while (pmantissa >= 2.0)
            {
                pmantissa = pmantissa >> 1;
                fexponent++;
            }
        }

        posit_unpacked<kbits, ebits, fbits> unpacked;

        unpacked.setKEFromTotalExp(fexponent);

        unpacked.sign = psign;
        unpacked.frac = pmantissa;

        bits_ = unpacked.template encode<nbits, ebits>();
    }

    Posit(float c)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        // ap_uint<32> bits = *reinterpret_cast<ap_uint<32> *>(&c);
        ap_uint<32> bits = bitcast_u32(c);

        FloatXUnpacked<8, 23> float_unpacked;
        float_unpacked.decode(bits);
        FloatXUnpacked<ebits, fbits> floatx_unpacked(float_unpacked);

        posit_unpacked<nbits, ebits> unpacked;

        unpacked.setKEFromTotalExp(floatx_unpacked.exp_);
        unpacked.sign_ = floatx_unpacked.sign_;
        unpacked.frac_ = floatx_unpacked.mant_;

        bits_ = unpacked.template encode<nbits, ebits>();
    }

    Posit(double c)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        // ap_uint<64> bits = *reinterpret_cast<ap_uint<64> *>(&c);
        ap_uint<64> bits = bitcast_u64(c);

        FloatXUnpacked<11, 52> double_unpacked;
        double_unpacked.decode(bits);
        FloatXUnpacked<ebits, fbits> floatx_unpacked(double_unpacked);

        posit_unpacked<kbits, ebits, fbits> unpacked;

        unpacked.setKEFromTotalExp(floatx_unpacked.exp_);
        unpacked.sign_ = floatx_unpacked.sign_;
        unpacked.frac_ = floatx_unpacked.mant_;

        bits_ = encode<nbits, ebits, kbits, ebits, fbits>(unpacked);
    }

    Posit(const posit_unpacked<nbits, ebits> &c)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        bits_ = encode<nbits, ebits, kbits, ebits, fbits>(c);
    }

    template <int fnbits, int fibits>
    Posit(ap_fixed<fnbits, fibits> c)
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = encode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = decode < nbits, ebits, kbits, ebits, fbits> limit = 1
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        bool fsign = c >= 0 ? 0 : 1;
        int fexponent = hls::log2(c(fnbits - 1, fnbits - fibits));

        ap_ufixed<fnbits - fibits + 1, 1> fmantissa;

        if (c == 0.0)
            fmantissa[fnbits - fibits] = 0;
        else
            fmantissa[fnbits - fibits] = 1;

        fmantissa(fnbits - fibits - 1, 0) = c(fnbits - fibits - 1, 0);

        // get sign from float sign
        bool psign = fsign;

        ap_ufixed<fbits + 1, 2> pmantissa = fmantissa;

        posit_unpacked<kbits, ebits, fbits> unpacked;

        unpacked.setKEFromTotalExp(fexponent);

        unpacked.sign = psign;
        unpacked.frac = pmantissa;

        bits_ = unpacked.encode();
    }

    operator int() const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);

        int exp = unpacked.getTotalExp();
        int res = ap_ufixed<fbits * 2, fbits>(unpacked.frac) << exp;
        if (unpacked.sign)
        {
            res = -res;
        }
        return res;
    }

    operator unsigned int() const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);

        int exp = unpacked.getTotalExp();
        int res = ap_fixed<fbits * 2, fbits>(unpacked.frac) << exp;
        return res;
    }

    operator float() const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);

        FloatXUnpacked<ebits, fbits> floatx_unpacked;
        floatx_unpacked.sign_ = unpacked.sign_;
        floatx_unpacked.exp_ = unpacked.getTotalExp();
        floatx_unpacked.mant_ = unpacked.frac_;

        FloatXUnpacked<8, 23> float_unpacked(floatx_unpacked);

        ap_uint<32> bits = floatx_unpacked.encode();

        return bitcast_f32(bits);
    }

    operator double() const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> unpacked;
        unpacked.template decode<nbits, ebits>(bits_);

        FloatXUnpacked<ebits, fbits> floatx_unpacked;
        floatx_unpacked.sign_ = unpacked.sign_;
        floatx_unpacked.exp_ = unpacked.getTotalExp();
        floatx_unpacked.mant_ = unpacked.frac_;

        FloatXUnpacked<11, 52> float_unpacked(floatx_unpacked);

        ap_uint<64> bits = floatx_unpacked.encode();

        return bitcast_f64(bits);
    }

    operator posit_unpacked<nbits, ebits>() const
    {
        // #pragma HLS INLINE

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked;
    }

    template <int fnbits, int fibits>
    operator ap_fixed<fnbits, fibits>() const
    {
        // #pragma HLS INLINE

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);

        if (unpacked.frac == 0.0)
        {
            return 0;
        }
        else
        {
            int exp = unpacked.getTotalExp();
            int res = (unpacked.frac << exp);
            if (unpacked.sign)
            {
                res = -res;
            }
            return res;
        }
    }
    /*
    Posit operator-() const
    {
        posit_unpacked<kbits, ebits, fbits> in1;

        in1.template decode<nbits, ebits>(bits_);

        in1.sign_ = !in1.sign_;

        Posit result;
        result.bits_ = in1.template encode<nbits, ebits>();

        return result;
    }

    Posit &operator+=(const Posit &rhs)
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;

        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 + in2;

        bits_ = out.template encode<nbits, ebits>();

        return *this;
    }

    Posit &operator-=(const Posit &rhs)
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        in2.sign = !in2.sign;

        posit_unpacked<kbits, ebits, fbits> out = posit_adder(in1, in2);

        bits_ = out.template encode<nbits, ebits>();

        return *this;
    }

    Posit &operator*=(const Posit &rhs)
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 * in2;

        bits_ = out.template encode<nbits, ebits>();

        return *this;
    }

    Posit &operator/=(const Posit &rhs)
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 / in2;

        bits_ = out.template encode<nbits, ebits>();

        return *this;
    }

    bool operator<(const Posit &rhs) const
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        return posit_lessthan(in1, in2);
    }

    bool operator>(const Posit &rhs) const
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        return posit_morethan(in1, in2);
    }

    bool operator<=(const Posit &rhs) const
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        return posit_lesseqthan(in1, in2);
    }

    bool operator>=(const Posit &rhs) const
    {
        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        return posit_moreeqthan(in1, in2);
    }

    bool operator==(const Posit &rhs) const
    {
        if (bits_ == rhs.bits_)
            return true;
        else
            return false;
    }

    bool operator!=(const Posit &rhs) const
    {
        if (bits_ != rhs.bits_)
            return true;
        else
            return false;
    }
    */

    posit_unpacked<nbits, ebits> operator+(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 + rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator-(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 - rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator*(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE
//#pragma HLS allocation function instances = posit_mult < kbits, ebits, fbits> limit = 1

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 * rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator/(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 / rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator-() const
    {
        // #pragma HLS INLINE

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);

        return -in1;
    }

    /*
    Posit operator+(const Posit &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 + in2;

        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();

        return result;
    }

    Posit operator-(const Posit &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 - in2;

        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();

        return result;
    }

    Posit operator*(const Posit &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 * in2;

        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();

        return result;
    }

    Posit operator/(const Posit &rhs) const
    {
        // #pragma HLS INLINE

        posit_unpacked<kbits, ebits, fbits> in1, in2;
        in1.template decode<nbits, ebits>(bits_);
        in2.template decode<nbits, ebits>(rhs.bits_);

        posit_unpacked<kbits, ebits, fbits> out = in1 / in2;

        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();

        return result;
    }

    Posit fabs() const
    {
        posit_unpacked<kbits, ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        posit_unpacked<kbits, ebits, fbits> out = posit_fabs(in1);
        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();
        return result;
    }

    Posit floor() const
    {
        posit_unpacked<kbits, ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        posit_unpacked<kbits, ebits, fbits> out = posit_floor(in1);
        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();
        return result;
    }

    Posit round() const
    {
        posit_unpacked<kbits, ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        posit_unpacked<kbits, ebits, fbits> out = posit_round(in1);
        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();
        return result;
    }

    Posit ceil() const
    {
        posit_unpacked<kbits, ebits, fbits> in1;
        in1.template decode<nbits, ebits>(bits_);
        posit_unpacked<kbits, ebits, fbits> out = posit_ceil(in1);
        Posit result;
        result.bits_ = out.template encode<nbits, ebits>();
        return result;
    }
    */

private:
    ap_uint<nbits> bits_;
};

template <int nbits, int ebits>
Posit<nbits, ebits> fabs(const Posit<nbits, ebits> &p)
{
    return p.fabs();
}

template <int nbits, int ebits>
Posit<nbits, ebits> floor(const Posit<nbits, ebits> &p)
{
    return p.floor();
}

template <int nbits, int ebits>
Posit<nbits, ebits> round(const Posit<nbits, ebits> &p)
{
    return p.round();
}

template <int nbits, int ebits>
Posit<nbits, ebits> ceil(const Posit<nbits, ebits> &p)
{
    return p.ceil();
}