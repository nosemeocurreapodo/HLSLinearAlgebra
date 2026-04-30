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

    posit_unpacked()
    {
        //  #pragma HLS allocation function instances = decode < kbits, ebits, fbits> limit = 1
    }

    template <int onbits, int oebits>
    posit_unpacked(const FloatXUnpacked<onbits, oebits> &other)
    {
        // #pragma HLS INLINE off

        ap_int<oebits + 1> oexp = (ap_int<oebits + 1>)(ap_uint<1>(0), other.exp_) - (ap_int<oebits + 1>)fbias<oebits>::value;
        // ap_int<ebits + 1> exp = oexp + (ap_int<ebits + 1>)fbias<ebits>::value;

        zero_ = other.zero_;
        inf_ = other.inf_;
        sign_ = other.sign_;
        // exp_ = exp(ebits - 1, 0);
        setKEFromTotalExp(oexp);
        if (m_fbits >= other.fbits)
        {
            // no rounding needed
            mant_(m_fbits - 1, m_fbits - other.fbits) = other.mant_(other.fbits - 1, 0);
            mant_(m_fbits - other.fbits - 1, 0) = 0;
        }
        else
        {
            // round as well
            mant_(m_fbits - 1, 0) = other.mant_(other.fbits - 1, other.fbits - m_fbits);

            /*
            ap_uint<m_fbits + 2> rfrac = (ap_uint<1>(0), other.mant_(other.fbits - 1, other.fbits - m_fbits - 1));
            if (rfrac[0] == 1)
                rfrac += 1;
            if (rfrac[m_fbits + 1] == 1)
            {
                rfrac = rfrac >> 1;
                exp_++;
            }
            mant_(m_fbits - 1, 0) = rfrac(m_fbits, 1);
            */
        }
    }

    FloatXUnpacked<nbits, ebits + m_kbits> tofloatxunpacked() const
    {
        // #pragma HLS INLINE off

        FloatXUnpacked<nbits, ebits + m_kbits> floatx_unpacked;
        floatx_unpacked.zero_ = zero_;
        floatx_unpacked.inf_ = inf_;
        floatx_unpacked.sign_ = sign_;
        floatx_unpacked.exp_ = getTotalExp() + fbias<ebits + m_kbits>::value;
        if (floatx_unpacked.fbits > m_fbits)
        {
            floatx_unpacked.mant_(floatx_unpacked.fbits - 1, floatx_unpacked.fbits - m_fbits) = mant_;
            floatx_unpacked.mant_(floatx_unpacked.fbits - m_fbits - 1, 0) = 0;
        }
        else
        {
            floatx_unpacked.mant_ = mant_(m_fbits - 1, m_fbits - floatx_unpacked.fbits);
        }

        return floatx_unpacked;
    }

    ap_uint<nbits> encode_old() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_uint<nbits> bits;

        bool simbol;
        int reg_len;

        simbol = k_ >= 0 ? 0 : 1;
        reg_len = k_ >= 0 ? int(k_ + 1) : int(-k_);

        bits[nbits - 1] = sign_;
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
                bits[bit] = exp_[pos];

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
                bits[bit] = mant_[pos];
                continue;
            }
        }

        return bits;
    }

    ap_uint<nbits> encode1() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_uint<nbits> bits = 0;

        const bool reg_bit = (k_ >= 0) ? 0 : 1;
        const int reg_len = (k_ >= 0) ? int(k_ + 1) : int(-k_);

        // available bits after sign
        const int payload_bits = nbits - 1;

        // regime consumes reg_len identical bits + terminating opposite bit
        const int regime_total = reg_len + 1;

        // saturate if regime alone fills everything
        if (regime_total >= payload_bits)
        {
            bits[nbits - 1] = sign_;

            // fill all remaining bits with regime run
            for (int i = 0; i < nbits - 1; i++)
            {
#pragma HLS UNROLL
                bits[i] = reg_bit;
            }

            return bits;
        }

        bits[nbits - 1] = sign_;

        // ------------------------------------------------------------------
        // Build payload = regime || exponent || mantissa
        // payload is packed right-aligned first, then shifted into position.
        // ------------------------------------------------------------------
        ap_uint<nbits - 1> payload = 0;
        int wr_pos = payload_bits - 1;

        // regime run
        for (int i = 0; i < reg_len; i++)
        {
#pragma HLS UNROLL
            payload[wr_pos - i] = reg_bit;
        }
        wr_pos -= reg_len;

        // terminating regime bit
        payload[wr_pos] = !reg_bit;
        wr_pos--;

        // exponent bits, MSB first
        for (int i = ebits - 1; i >= 0; i--)
        {
#pragma HLS UNROLL
            if (wr_pos >= 0)
            {
                payload[wr_pos] = exp_[i];
                wr_pos--;
            }
        }

        // mantissa bits, MSB first
        for (int i = m_fbits - 1; i >= 0; i--)
        {
#pragma HLS UNROLL
            if (wr_pos >= 0)
            {
                payload[wr_pos] = mant_[i];
                wr_pos--;
            }
        }

        bits(nbits - 2, 0) = payload;
        return bits;
    }

    ap_uint<nbits> encode() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        const bool reg_bit = (k_ < 0);
        const ap_uint<m_kbits + 1> reg_len = hls::min((k_ >= 0) ? ap_uint<m_kbits + 1>(k_ + 1) : ap_uint<m_kbits + 1>(-k_), ap_uint<m_kbits + 1>(nbits - 1));
        // ap_uint<m_kbits + 1> reg_len = ap_uint<m_kbits + 1>(k_ + 1);
        // if (k_ < 0)
        //     reg_len = ap_uint<m_kbits + 1>(-k_);

        // ap_uint<nbits> ones = ~ap_uint<nbits>(0);
        //  ap_uint<nbits> zeros = 0;

        const int end_exp_map_bits = 1 + ebits + m_fbits;
        // const int end_bit = nbits - 1 - reg_len;

        const ap_uint<end_exp_map_bits> end_exp_mant = (!reg_bit, exp_, mant_);

        ap_uint<nbits> bits = 0;

        bits[nbits - 1] = sign_;

        if (reg_bit)
            bits(nbits - 2, nbits - 1 - reg_len) = -1; // ones(nbits - 2, nbits - 1 - reg_len);
        // else
        //     bits(nbits - 2, nbits - 1 - reg_len) = 0;

        if (nbits - 1 - reg_len > 0)
            bits(nbits - 2 - reg_len, 0) = end_exp_mant(end_exp_map_bits - 1, end_exp_map_bits - (nbits - 1 - reg_len));

        return bits;
    }

    ap_uint<nbits> encode3() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_uint<nbits> bits = 0;
        bits[nbits - 1] = sign_;

        static constexpr int payload_bits = nbits - 1;
        static constexpr int ef_bits = ebits + m_fbits;

        bool reg_s = (k_ < 0);
        ap_uint<m_kbits + 1> reg_len = (k_ >= 0) ? ap_uint<m_kbits + 1>(k_ + 1)
                                                 : ap_uint<m_kbits + 1>(-k_);
        ap_uint<m_kbits + 2> reg_total = reg_len + 1;

        // Saturation: regime fills all payload
        if (reg_total >= payload_bits)
        {
            bits[nbits - 2] = reg_s;
            if (reg_s)
            {
                // 000...0 pattern after sign, except final structure is all zeros
                bits(nbits - 2, 0) = 0;
            }
            else
            {
                // 111...1 pattern after sign
                bits(nbits - 2, 0) = ~ap_uint<payload_bits>(0);
            }
            return bits;
        }

        ap_uint<payload_bits> regime = 0;

        if (!reg_s)
        {
            // k >= 0 : regime = 111...110
            ap_uint<payload_bits> ones = (((ap_uint<payload_bits>)1 << reg_len) - 1);
            regime = ones << (payload_bits - reg_len);
        }
        else
        {
            // k < 0 : regime = 000...001
            regime = (ap_uint<payload_bits>)1 << (payload_bits - reg_total);
        }

        // exponent+fraction payload, MSB aligned just after regime
        ap_uint<ef_bits> ef = (exp_, mant_);

        ap_uint<payload_bits> ef_field = 0;
        int frac_shift = payload_bits - reg_total - ef_bits;

        if (frac_shift >= 0)
        {
            ef_field = (ap_uint<payload_bits>)ef << frac_shift;
        }
        else
        {
            // truncate low bits if not enough room
            ef_field = (ap_uint<payload_bits>)(ef >> (-frac_shift));
        }

        bits(nbits - 2, 0) = regime | ef_field;
        return bits;
    }

    void decode_old(const ap_uint<nbits> &bits)
    {
        // #pragma HLS INLINE off

        zero_ = bits == 0;
        sign_ = bits[nbits - 1];
        bool simbol = bits[nbits - 2];

        // ap_int<kbits> k;
        // ap_uint<ebits> exp;
        // ap_ufixed<fbits + 1, 1> frac;

        // start as zero or inf depending on first bit
        if (bits[nbits - 1])
        {
            // for inf, start k as a large number
            k_ = large_k;
            exp_ = 0;
            mant_ = 0;
        }
        else
        {
            k_ = 0;
            exp_ = 0;
            mant_ = 0;
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
                        k_ = dist - 1;
                    else
                        k_ = -dist;

                    last_state_bit = bit;

                    state = E;
                }
                continue;
            }

            if (state == E)
            {
                // unpacked.exp[es - 1 - counter] = bits[bit];
                int pos = ebits - dist;
                exp_[pos] = bits[bit];

                if (pos == 0)
                {
                    last_state_bit = bit;
                    // mant_[m_fbits] = 1;
                    state = F;
                }
                continue;
            }

            if (state == F)
            {
                int pos = m_fbits - dist;
                mant_[pos] = bits[bit];
                continue;
            }
        }
    }

    void decode(const ap_uint<nbits> &bits)
    {
        // #pragma HLS INLINE off

        zero_ = (bits == 0);
        sign_ = bits[nbits - 1];
        k_ = 0;
        exp_ = 0;
        mant_ = 0;

        if (zero_)
            return;

        // Regime starts just below sign
        const bool reg_bit = bits[nbits - 2];

        // Count leading run of reg_bit in payload
        // ap_uint<nbits - 1> payload = bits(nbits - 2, 0);
        // ap_uint<clog2<nbits - 1>::value> reg_len = count_leading_symbol(payload, reg_bit);
        const ap_uint<clog2<nbits - 1>::value> reg_len = count_leading_symbol(ap_uint<nbits - 1>(bits(nbits - 2, 0)), reg_bit);

        /*
        int reg_len = 0;
    Decode_regime_loop:
        for (int i = nbits - 2; i >= 0; --i)
        {
            // #pragma HLS UNROLL
            if (bits[i] == reg_bit)
                reg_len++;
            else
                break;
        }
        */

        // Recover k_ according to your encode convention:
        // reg_bit = (k_ < 0)
        if (reg_bit)
            k_ = -reg_len;
        else
            k_ = reg_len - 1;

        // Remaining bits after regime
        const ap_uint<clog2<nbits>::value> rem_bits = nbits - 1 - reg_len;

        // If nothing remains, exponent and mantissa stay zero
        if (rem_bits <= 0)
            return;

        const int end_exp_mant_bits = 1 + ebits + m_fbits;
        ap_uint<end_exp_mant_bits> exp_mant = 0;
        exp_mant(end_exp_mant_bits - 1, end_exp_mant_bits - rem_bits) = bits(rem_bits - 1, 0);

        exp_(ebits - 1, 0) = exp_mant(end_exp_mant_bits - 2, m_fbits);
        mant_(m_fbits - 1, 0) = exp_mant(m_fbits - 1, 0);

        /*
        // Shift payload so the "end_exp_mant" field is aligned at the top
        ap_uint<nbits - 1> payload = bits(nbits - 2, 0);
        ap_uint<nbits - 1> shifted = payload << reg_len;

        // shifted now begins with:
        // [ !reg_bit ][ exp_ ][ mant_ ] ...
        //
        // Skip the first bit (the terminator / first bit of end_exp_mant)
        if (rem_bits >= 2)
        {
            // exponent bits available after skipping terminator
            const int avail_after_term = rem_bits - 1;
            const int exp_bits_to_copy = (avail_after_term >= ebits) ? ebits : avail_after_term;

        Decode_exp_loop:
            for (int i = 0; i < exp_bits_to_copy; ++i)
            {
                // #pragma HLS UNROLL
                exp_[ebits - 1 - i] = shifted[nbits - 3 - i];
            }

            // mantissa bits available after terminator + exponent
            const int mant_bits_avail = avail_after_term - exp_bits_to_copy;
            const int mant_bits_to_copy = (mant_bits_avail >= m_fbits) ? m_fbits : mant_bits_avail;

        Decode_mant_loop:
            for (int i = 0; i < mant_bits_to_copy; ++i)
            {
                // #pragma HLS UNROLL
                mant_[m_fbits - 1 - i] = shifted[nbits - 3 - exp_bits_to_copy - i];
            }
        }
        */
    }

    ap_int<m_kbits + ebits> getTotalExp() const
    {
        // #pragma HLS INLINE off

        ap_int<m_kbits + ebits> total = k_;
        total <<= ebits;
        total += exp_;
        return total;
    }

    template <int in_ebits>
    void setKEFromTotalExp(ap_int<in_ebits> in_exp)
    {
        // #pragma HLS INLINE off

        k_ = in_exp >> ebits;
        exp_ = in_exp.range(ebits - 1, 0);
    }

    template <int in_nbits>
    posit_unpacked(ap_int<in_nbits> c)
    {
        // #pragma HLS INLINE off

        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            k_ = 0;
            exp_ = 0;
            mant_ = 0;
            return;
        }

        zero_ = 0;
        inf_ = 0;

        bool psign = c < 0;
        ap_uint<in_nbits> i = hls::abs(c);
        int lz = count_leading_zeros(i);
        i = i << (lz + 1);

        sign_ = psign;
        ap_int<in_nbits> exp = in_nbits - 1 - lz;

        setKEFromTotalExp(exp);

        if constexpr (m_fbits <= in_nbits)
            mant_ = i(in_nbits - 1, in_nbits - m_fbits);
        else
            mant_(m_fbits - 1, m_fbits - in_nbits) = i;
    }

    template <int in_nbits>
    posit_unpacked(ap_uint<in_nbits> c)
    {
        // #pragma HLS INLINE off

        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            k_ = 0;
            exp_ = 0;
            mant_ = 0;
            return;
        }

        zero_ = 0;
        inf_ = 0;

        bool psign = c < 0;
        ap_uint<in_nbits> i = hls::abs(c);
        int lz = count_leading_zeros(i);
        i = i << (lz + 1);

        sign_ = psign;
        ap_int<in_nbits> exp = in_nbits - 1 - lz;

        setKEFromTotalExp(exp);

        if constexpr (m_fbits <= in_nbits)
            mant_ = i(in_nbits - 1, in_nbits - m_fbits);
        else
            mant_(m_fbits - 1, m_fbits - in_nbits) = i;
    }

    posit_unpacked(int c)
    {
        // #pragma HLS INLINE off

        if (c == 0)
        {
            zero_ = 1;
            inf_ = 0;
            sign_ = 0;
            k_ = 0;
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
        ap_int<32> exp = 31 - lz;

        setKEFromTotalExp(exp);

        if constexpr (m_fbits <= 32)
            mant_ = i(31, 32 - m_fbits);
        else
            mant_(m_fbits - 1, m_fbits - 32) = i;
    }

    template <int onbits>
    operator ap_int<onbits>() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_int<ebits + m_kbits> exp = getTotalExp();
        ap_int<onbits> res = (ap_uint<2>(0b01), mant_) >> (m_fbits - exp);
        if (sign_)
            res = -res;
        return res;
    }

    operator ap_int<16>() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_int<ebits + m_kbits> exp = getTotalExp();
        ap_int<16> res = (ap_uint<2>(0b01), mant_) >> (m_fbits - exp);
        if (sign_)
            res = -res;
        return res;
    }

    operator int() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_int<ebits + m_kbits> exp = getTotalExp();
        int res = (ap_uint<2>(0b01), mant_) >> (m_fbits - exp);
        if (sign_)
            res = -res;
        return res;
    }

    operator unsigned int() const
    {
        // #pragma HLS INLINE off

        if (zero_)
            return 0;

        ap_int<ebits + m_kbits> exp = getTotalExp();
        int res = (ap_uint<2>(0b01), mant_) >> (m_fbits - exp);
        return res;
    }

    posit_unpacked operator+(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE off

        // int exp1 = getTotalExp();
        // int exp2 = rhs.getTotalExp();

        // set biggest posit to be in1
        // int diff_texp = exp1 - exp2;

        if (zero_)
            return rhs;
        if (rhs.zero_)
            return *this;

        bool sign;
        bool zero;
        bool inf = 0;
        ap_int<ebits + 2> exp;
        ap_int<m_kbits + 1> k;

        ap_int<m_kbits + ebits> nk = (ap_int<m_kbits + ebits>)(k_ - rhs.k_) << ebits;
        ap_int<m_kbits + ebits> diff_texp = nk + (ap_int<m_kbits + ebits>)exp_ - (ap_int<m_kbits + ebits>)rhs.exp_;

        ap_int<m_fbits + 3> frac1 = (ap_uint<2>(0b01), mant_, ap_uint<1>(0));
        ap_int<m_fbits + 3> frac2 = (ap_uint<2>(0b01), rhs.mant_, ap_uint<1>(0));

        bool lhs_ge_rhs =
            (diff_texp > 0) ||
            (diff_texp == 0 && frac1 >= frac2);

        if (lhs_ge_rhs)
        {
            sign = sign_;
            k = k_;
            exp = exp_;
            frac2 = frac2 >> diff_texp;
            if (sign_ != rhs.sign_)
                frac2 = -frac2;
        }
        else
        {
            sign = rhs.sign_;
            k = rhs.k_;
            exp = rhs.exp_;
            frac1 = frac1 >> -diff_texp;
            if (sign_ != rhs.sign_)
                frac1 = -frac1;
        }

        // do addition (always positive)
        ap_uint<m_fbits + 4> frac = frac1 + frac2;

        if (frac == 0)
        {
            zero = 1;
        }
        else
        {
            zero = 0;
        }

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

        // round
        /*
        if (frac[0] == 1)
            frac += 1;

        if (frac[fbits + 2] == 1)
        {
            frac = frac >> 1;
            exp++;
        }

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
        */

        posit_unpacked out;

        // out.setKEFromTotalExp(exp);
        out.zero_ = zero;
        out.inf_ = inf;
        out.sign_ = sign;
        out.k_ = k;
        out.exp_ = exp;
        out.mant_ = frac(m_fbits, 1);

        return out;
    }

    posit_unpacked operator-(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked in2;
        in2.zero_ = rhs.zero_;
        in2.inf_ = rhs.inf_;
        in2.sign_ = !rhs.sign_;
        in2.k_ = rhs.k_;
        in2.exp_ = rhs.exp_;
        in2.mant_ = rhs.mant_;

        posit_unpacked out = (*this) + in2;

        return out;
    }

    posit_unpacked operator*(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE off

        bool zero = zero_ | rhs.zero_;
        bool inf = inf_ | rhs.inf_;
        bool sign = sign_ ^ rhs.sign_;

        ap_int<ebits + 2> exp = exp_ + rhs.exp_;
        ap_int<m_kbits + 1> k = k_ + rhs.k_;
        // result goes from [1.0 to 4.0)
        ap_uint<m_fbits * 2 + 2> mant = (ap_uint<1>(1), mant_) * (ap_uint<1>(1), rhs.mant_);

        // normalize
        if (mant[m_fbits * 2 + 1] == 1)
        {
            mant = mant >> 1;
            exp++;
        }

        // normalize exponent
        if (exp >= (1 << ebits))
        {
            exp -= (1 << ebits);
            k++;
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
        */

        posit_unpacked out;

        // out.setKEFromTotalExp(exp);
        out.zero_ = zero;
        out.inf_ = inf;
        out.sign_ = sign;
        out.k_ = k;
        out.exp_ = exp;
        out.mant_ = mant(m_fbits * 2 - 1, m_fbits);

        return out;
    }

    posit_unpacked operator/(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE off

        // bool sign = sign_ | rhs.sign_;
        // bool zero = zero_ | rhs.zero_;
        // bool inf = 0;
        // ap_uint<ebits> exp = exp_ + rhs.exp_;
        // ap_uint<m_kbits> k = k_ + rhs.k_;
        //  result goes from [1.0 to 4.0)
        // ap_uint<m_fbits * 2 + 2> mant = (ap_uint<1>(1), mant_) / (ap_uint<1>(1), rhs.mant_);

        bool zero = zero_;
        bool inf = rhs.zero_;
        bool sign = sign_ ^ rhs.sign_;

        ap_uint<m_fbits + 1> num = (ap_uint<1>(1), mant_);
        ap_uint<m_fbits + 1> den = (ap_uint<1>(1), rhs.mant_);

        ap_int<ebits + 1> exp = (ap_int<ebits + 1>)exp_ - (ap_int<ebits + 1>)rhs.exp_;
        ap_int<m_kbits + 1> k = k_ - rhs.k_;

        ap_uint<m_fbits * 2 + 2> num1 = (ap_uint<m_fbits * 2 + 2>)num << (m_fbits + 1);
        // goes from (0.5, 2.000)
        ap_uint<m_fbits * 2 + 2> mant = num1 / den;

        if (mant[m_fbits + 1] == 0)
        {
            mant = mant << 1;
            exp--;
        }

        if (exp < 0)
        {
            exp += (1 << ebits);
            k--;
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
        */

        posit_unpacked out;

        // out.setKEFromTotalExp(exp);
        out.zero_ = zero;
        out.inf_ = inf;
        out.sign_ = sign;
        out.k_ = k;
        out.exp_ = exp;
        out.mant_ = mant(m_fbits, 1);

        return out;
    }

    posit_unpacked operator-() const
    {
        // #pragma HLS INLINE off

        posit_unpacked result;
        result.zero_ = zero_;
        result.inf_ = inf_;
        result.sign_ = !sign_;
        result.exp_ = exp_;
        result.k_ = k_;
        result.mant_ = mant_;

        return result;
    }

    bool operator==(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE off

        if (zero_)
            if (rhs.zero_)
                return true;
            else
                return false;

        if (rhs.zero_)
            return false;

        if (sign_ != rhs.sign_)
            return false;

        if (k_ != rhs.k_)
            return false;

        if (exp_ != rhs.exp_)
            return false;

        if (mant_ != rhs.mant_)
            return false;

        return true;
    }

    bool operator<(const posit_unpacked &rhs) const
    {
        // #pragma HLS INLINE off

        if (zero_)
            if (rhs.zero_)
                return false;
            else
                return !rhs.sign_;

        if (rhs.zero_)
            return sign_;

        if (sign_ != rhs.sign_)
            return !rhs.sign_;

        if (k_ != rhs.k_)
            return sign_ != (k_ < rhs.k_);

        if (exp_ != rhs.exp_)
            return sign_ != (exp_ < rhs.exp_);

        if (mant_ != rhs.mant_)
            return sign_ != (mant_ < rhs.mant_);

        return false;
    }

    bool zero_;
    bool inf_;
    bool sign_;
    // the max amount of bits for r is nbits-1 bits, nbits-2 bits beeing 0 (or 1), and the last beeing 1 (or 0)
    // k is the amount of counted bits
    // which can be stored in log2(nbits - 2) bits
    ap_int<m_kbits> k_;
    ap_uint<ebits> exp_;
    // the max amount of bits for frac is nbits - 1 (sign) - 2 (min bits for k) - es;
    ap_uint<m_fbits> mant_;
};

template <int nbits, int ebits>
class Posit
{
public:
    Posit()
    {
    }

    Posit(const Posit &other)
    {
        bits_ = other.bits_;
    }

    Posit &operator=(const Posit &other)
    {
        bits_ = other.bits_;
        /*
        if (this != &other)
        {
            bits_ = other.bits_;
        }
        */
        return *this;
    }

    Posit(const posit_unpacked<nbits, ebits> &c)
    {
        // #pragma HLS INLINE off

        bits_ = c.encode();
    }

    operator posit_unpacked<nbits, ebits>() const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked;
    }

    template <int inbits>
    Posit(ap_int<inbits> c)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked(c);
        bits_ = unpacked.encode();
    }

    template <int inbits>
    Posit(ap_uint<inbits> c)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked(c);
        bits_ = unpacked.encode();
    }

    Posit(int c)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked(c);
        bits_ = unpacked.encode();
    }

    Posit(unsigned int c)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked(c);
        bits_ = unpacked.encode();
    }

    Posit(float c)
    {
        // #pragma HLS INLINE off

        ap_uint<32> bits = bitcast_u32(c);
        FloatXUnpacked<32, 8> float_unpacked;
        float_unpacked.decode(bits);
        posit_unpacked<nbits, ebits> unpacked(float_unpacked);
        bits_ = unpacked.encode();
    }

    Posit(double c)
    {
        // #pragma HLS INLINE off

        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<64, 11> float_unpacked;
        float_unpacked.decode(bits);
        posit_unpacked<nbits, ebits> unpacked(float_unpacked);
        bits_ = unpacked.encode();
    }

    template <int onbits>
    operator ap_int<onbits>() const
    {
        // #pragma HLS INLINE off

        if (bits_ == 0)
            return 0;

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return unpacked;
    }

    operator int() const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return int(unpacked);
    }

    operator unsigned int() const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);
        return (unsigned int)(unpacked);
    }

    operator float() const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);

        FloatXUnpacked<32, 8> floatx_unpacked(unpacked.tofloatxunpacked());

        ap_uint<32> bits = floatx_unpacked.encode();

        return bitcast_f32(bits);
    }

    operator double() const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> unpacked;
        unpacked.decode(bits_);

        FloatXUnpacked<64, 11> floatx_unpacked(unpacked.tofloatxunpacked());

        ap_uint<64> bits = floatx_unpacked.encode();

        return bitcast_f64(bits);
    }

    /*
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
    */

    posit_unpacked<nbits, ebits> operator+(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 + rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator-(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 - rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator*(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 * rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator/(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);
        posit_unpacked<nbits, ebits> out = in1 / rhs;

        return out;
    }

    posit_unpacked<nbits, ebits> operator-() const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> in1;
        in1.decode(bits_);

        return -in1;
    }

    Posit &operator+=(const posit_unpacked<nbits, ebits> &rhs)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        aux = aux + rhs;
        bits_ = aux.encode();
        return *this;
    }

    Posit &operator-=(const posit_unpacked<nbits, ebits> &rhs)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        aux = aux - rhs;
        bits_ = aux.encode();
        return *this;
    }

    Posit &operator*=(const posit_unpacked<nbits, ebits> &rhs)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        aux = aux * rhs;
        bits_ = aux.encode();
        return *this;
    }

    Posit &operator/=(const posit_unpacked<nbits, ebits> &rhs)
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        aux = aux / rhs;
        bits_ = aux.encode();
        return *this;
    }

    bool operator<(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        return aux < rhs;
    }

    bool operator>(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        return rhs < aux;
    }

    bool operator<=(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        return (aux < rhs | aux == rhs);
    }

    bool operator>=(const posit_unpacked<nbits, ebits> &rhs) const
    {
        // #pragma HLS INLINE off

        posit_unpacked<nbits, ebits> aux;
        aux.decode(bits_);
        return (rhs < aux | aux == rhs);
    }

    bool operator==(const Posit &rhs) const
    {
        // #pragma HLS INLINE off

        if (bits_ == rhs.bits_)
            return true;
        else
            return false;
    }

    bool operator!=(const Posit &rhs) const
    {
        // #pragma HLS INLINE off

        if (bits_ != rhs.bits_)
            return true;
        else
            return false;
    }

    // private:
    ap_uint<nbits> bits_;
};

template <int nbits, int ebits>
inline Posit<nbits, ebits> fabs(const Posit<nbits, ebits> &a)
{
    Posit<nbits, ebits> b = a;
    b.bits_[nbits - 1] = 0;
    return b;
}

template <int nbits, int ebits>
inline Posit<nbits, ebits> floor(const Posit<nbits, ebits> &a)
{
    Posit<nbits, ebits> f_int = int(a);
    return f_int;
}

template <int nbits, int ebits>
inline Posit<nbits, ebits> ceil(const posit_unpacked<nbits, ebits> &a)
{
    Posit<nbits, ebits> f_int = int(a);
    posit_unpacked<nbits, ebits> diff = a - f_int;
    if (posit_unpacked<nbits, ebits>(0) < diff)
        f_int += posit_unpacked<nbits, ebits>(1);
    return f_int;
}

template <int nbits, int ebits>
inline Posit<nbits, ebits> ceil(const Posit<nbits, ebits> &a)
{
    Posit<nbits, ebits> f_int = int(a);
    Posit<nbits, ebits> diff = a - f_int;
    if (Posit<nbits, ebits>(0) < diff)
        f_int += Posit<nbits, ebits>(1);
    return f_int;
}

template <int nbits, int ebits>
inline Posit<nbits, ebits> round(const Posit<nbits, ebits> &a)
{
    Posit<nbits, ebits> f_int = int(a);
    Posit<nbits, ebits> diff = a - f_int;
    if (Posit<nbits, ebits>(0.5) < diff)
        f_int += Posit<nbits, ebits>(1);
    return f_int;
}

template <int nbits, int ebits>
inline Posit<nbits, ebits> fmod(const Posit<nbits, ebits> &a, const Posit<nbits, ebits> &b)
{
    Posit<nbits, ebits> c = a / b;
    Posit<nbits, ebits> d = a - floor(c) * b;
    return d;
}

template <int nbits, int ebits>
inline Posit<nbits, ebits> exp(const Posit<nbits, ebits> &a)
{
    return Posit<nbits, ebits>(21);
}