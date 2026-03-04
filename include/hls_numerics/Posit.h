#pragma once

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

// Your project headers (kept)
// - common.h is assumed to provide round_to(...) and maybe count_leading_simbol(...)
// - FloatX.h is assumed to provide FloatXUnpacked<...>
#include "common.h"
#include "FloatX.h"

// ============================================================
// Unpacked posit container (sign, regime k, exponent, fraction)
// ============================================================
template <int kbits, int ebits, int fbits>
class posit_unpacked
{
public:
    bool sign_;
    ap_int<kbits> k_;
    ap_uint<ebits> exp_;
    // frac_ includes hidden bit at position fbits (so width is fbits+1, integer bits = 1)
    ap_ufixed<fbits + 1, 1> frac_;

    static constexpr int large_k = kbits + ebits + fbits;

    posit_unpacked()
    {
#pragma HLS INLINE
        sign_ = 0;
        k_ = 0;
        exp_ = 0;
        frac_ = 0;
    }

    // Total exponent = k*2^ebits + exp
    ap_int<32> getTotalExp() const
    {
#pragma HLS INLINE
        return (ap_int<32>)k_ * (ap_int<32>)(1 << ebits) + (ap_int<32>)exp_;
    }

    // Set k and exp from total exponent using floor division (fast, no divider)
    void setKEFromTotalExp(ap_int<32> in_exp)
    {
#pragma HLS INLINE
        ap_int<32> k = floor_div_pow2<ebits>(in_exp);
        ap_int<32> e = in_exp - (k << ebits);

        // e should be in [0, 2^ebits-1] after floor division; keep safety
        if (e < 0)
        {
            e += (1 << ebits);
            k -= 1;
        }

        k_ = (ap_int<kbits>)k;
        exp_ = (ap_uint<ebits>)e;
    }

    // ------------------------------------------------------------
    // Fast encode/decode (no state machine; minimal variable muxing)
    // ------------------------------------------------------------
    template <int out_nbits, int out_ebits>
    ap_uint<out_nbits> encode() const
    {
#pragma HLS INLINE
        ap_uint<out_nbits> out = 0;

        // Zero
        if (frac_ == 0)
            return 0;

        out[out_nbits - 1] = sign_;

        // Regime encoding
        const bool reg_bit = (k_ < 0); // negative k => 1s run, positive k => 0s run
        ap_uint<out_nbits> run = (k_ >= 0) ? (ap_uint<out_nbits>)(k_ + 1)
                                           : (ap_uint<out_nbits>)(-k_);

        // Saturate if regime would eat everything (convention: all regime bits)
        if (run >= (out_nbits - 1))
        {
            out(out_nbits - 2, 0) = reg_bit ? ap_uint<out_nbits - 1>(-1) : ap_uint<out_nbits - 1>(0);
            return out;
        }

        int idx = out_nbits - 2;

    ENCODE_REGIME:
        for (int i = 0; i < out_nbits - 1; i++)
        {
#pragma HLS UNROLL
            if (i < (int)run)
                out[idx - i] = reg_bit;
        }
        idx -= (int)run;

        // terminating bit
        if (idx >= 0)
        {
            out[idx] = !reg_bit;
            idx--;
        }

        // exponent bits (truncate if not enough space)
        const int avail1 = (idx >= 0) ? (idx + 1) : 0;
        const int expbits = (avail1 < out_ebits) ? avail1 : out_ebits;

        if (expbits > 0)
        {
            out(idx, idx - expbits + 1) = exp_(out_ebits - 1, out_ebits - expbits);
            idx -= expbits;
        }

        // fraction bits (top bits below hidden bit)
        const int fracbits = (idx >= 0) ? (idx + 1) : 0;
        if (fracbits > 0)
        {
            // frac_ layout: [fbits:0] with hidden bit at [fbits]
            out(idx, 0) = frac_(fbits - 1, fbits - fracbits);
        }

        return out;
    }

    template <int in_nbits, int in_ebits>
    void decode(const ap_uint<in_nbits> &bits)
    {
#pragma HLS INLINE
        sign_ = bits[in_nbits - 1];

        // Zero
        if (bits == 0)
        {
            k_ = 0;
            exp_ = 0;
            frac_ = 0;
            return;
        }

        const bool regbit = bits[in_nbits - 2];

        // Payload below sign+regbit
        constexpr int PAY_W = in_nbits - 2;
        ap_uint<PAY_W> payload = bits(PAY_W - 1, 0);

        // Count leading run of regbit starting at MSB of payload
        ap_uint<clog2<PAY_W + 1>::value> run =
            count_leading_symbol<PAY_W>(payload, regbit);

        // Saturated / special if it consumes all payload
        if (run >= PAY_W)
        {
            k_ = (ap_int<kbits>)large_k;
            exp_ = 0;
            frac_ = 1.0;
            return;
        }

        // Compute k from run
        // regbit==0 => k = run-1
        // regbit==1 => k = -run
        k_ = (regbit == 0) ? (ap_int<kbits>)((ap_int<32>)run - 1)
                           : (ap_int<kbits>)(-(ap_int<32>)run);

        // Drop (run + 1 terminating bit)
        ap_uint<PAY_W> rest = payload << (run + 1);

        int avail = PAY_W - (int)(run + 1);

        // Exponent
        exp_ = 0;
        int expbits = (avail > 0) ? ((avail < in_ebits) ? avail : in_ebits) : 0;

        if (expbits > 0)
        {
            exp_ = rest(PAY_W - 1, PAY_W - expbits);
            rest <<= expbits;
            avail -= expbits;
        }

        // Fraction: hidden bit + remaining bits
        frac_ = 0;
        frac_[fbits] = 1;

        int fracbits = (avail > 0) ? ((avail < fbits) ? avail : fbits) : 0;
        if (fracbits > 0)
        {
            frac_(fbits - 1, fbits - fracbits) = rest(PAY_W - 1, PAY_W - fracbits);
        }
    }

    // -------------------------
    // Arithmetic (kept, cleaned)
    // -------------------------
    posit_unpacked operator-() const
    {
#pragma HLS INLINE
        posit_unpacked r = *this;
        r.sign_ = !sign_;
        return r;
    }

    posit_unpacked operator+(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        // total exp diff: (k1-k2)*2^ebits + (e1-e2)
        ap_int<32> diff_texp =
            ((ap_int<32>)k_ - (ap_int<32>)rhs.k_) * (ap_int<32>)(1 << ebits) +
            (ap_int<32>)exp_ - (ap_int<32>)rhs.exp_;

        // widen a bit to keep headroom
        ap_fixed<fbits + 2, 2> frac1 = frac_;
        ap_fixed<fbits + 2, 2> frac2 = rhs.frac_;

        ap_int<32> k = 0;
        ap_int<32> e = 0;
        bool sign = 0;

        if (diff_texp >= 0)
        {
            sign = sign_;
            k = (ap_int<32>)k_;
            e = (ap_int<32>)exp_;

            // align rhs
            frac2 = frac2 >> (int)diff_texp;
            if (sign_ != rhs.sign_)
                frac2 = -frac2;
        }
        else
        {
            sign = rhs.sign_;
            k = (ap_int<32>)rhs.k_;
            e = (ap_int<32>)rhs.exp_;

            frac1 = frac1 >> (int)(-diff_texp);
            if (sign_ != rhs.sign_)
                frac1 = -frac1;
        }

        ap_ufixed<fbits + 3, 3> frac = (ap_ufixed<fbits + 3, 3>)(frac1 + frac2);

        // normalize
        if (frac == 0)
        {
            posit_unpacked out;
            out.sign_ = 0;
            out.k_ = 0;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }

        // NOTE: assumes you have a fast count_leading_simbol(frac) in common.h.
        // If not, replace with a small unrolled leading-one detector.
        int shift = count_leading_simbol(frac) - 2;
        if (shift > 0)
        {
            frac <<= shift;
            e -= shift;
        }
        else if (shift < 0)
        {
            frac >>= (-shift);
            e += (-shift);
        }

        // normalize exponent into [0, 2^ebits-1], adjusting k
        if (e >= (1 << ebits))
        {
            e -= (1 << ebits);
            k += 1;
        }
        if (e < 0)
        {
            e += (1 << ebits);
            k -= 1;
        }

        ap_ufixed<fbits + 3, 3> rfrac = round_to(frac, fbits - 1);

        // normalize fraction again if rounding overflowed
        if (rfrac >= 2)
        {
            rfrac >>= 1;
            e += 1;
            if (e >= (1 << ebits))
            {
                e -= (1 << ebits);
                k += 1;
            }
        }

        posit_unpacked out;
        out.sign_ = sign;
        out.k_ = (ap_int<kbits>)k;
        out.exp_ = (ap_uint<ebits>)e;
        out.frac_ = (ap_ufixed<fbits + 1, 1>)rfrac;
        return out;
    }

    posit_unpacked operator-(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        posit_unpacked neg = rhs;
        neg.sign_ = !rhs.sign_;
        return (*this) + neg;
    }

    posit_unpacked operator/(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        posit_unpacked out;

        // div by zero => saturated/special (keep your previous convention)
        if (rhs.frac_ == 0)
        {
            out.sign_ = 0;
            out.k_ = (ap_int<kbits>)large_k;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }

        bool sign = sign_ ^ rhs.sign_;
        ap_int<32> k = (ap_int<32>)k_ - (ap_int<32>)rhs.k_;
        ap_int<32> e = (ap_int<32>)exp_ - (ap_int<32>)rhs.exp_;

        ap_ufixed<fbits * 2, 2> frac = (ap_ufixed<fbits * 2, 2>)(frac_ / rhs.frac_);

        // normalize fraction into [1,2)
        if (frac < 1.0)
        {
            frac <<= 1;
            e -= 1;
        }

        // wrap exponent
        if (e < 0)
        {
            e += (1 << ebits);
            k -= 1;
        }

        ap_ufixed<fbits + 1, 2> rfrac = round_to(frac, fbits - 1);

        if (rfrac >= 2)
        {
            rfrac >>= 1;
            e += 1;
        }

        if (e >= (1 << ebits))
        {
            e -= (1 << ebits);
            k += 1;
        }

        if (rfrac == 0)
        {
            sign = 0;
            k = 0;
            e = 0;
        }

        out.sign_ = sign;
        out.k_ = (ap_int<kbits>)k;
        out.exp_ = (ap_uint<ebits>)e;
        out.frac_ = (ap_ufixed<fbits + 1, 1>)rfrac;
        return out;
    }
};

// Multiply (kept as free function, cleaned)
template <int kbits, int ebits, int fbits>
static inline posit_unpacked<kbits, ebits, fbits>
posit_mult(const posit_unpacked<kbits, ebits, fbits> &lhs,
           const posit_unpacked<kbits, ebits, fbits> &rhs)
{
#pragma HLS INLINE
    posit_unpacked<kbits, ebits, fbits> out;

    bool sign = lhs.sign_ ^ rhs.sign_;
    ap_int<32> k = (ap_int<32>)lhs.k_ + (ap_int<32>)rhs.k_;
    ap_int<32> e = (ap_int<32>)lhs.exp_ + (ap_int<32>)rhs.exp_;

    ap_ufixed<fbits * 2 + 2, 2> frac = (ap_ufixed<fbits * 2 + 2, 2>)(lhs.frac_ * rhs.frac_);

    if (frac == 0)
    {
        out.sign_ = 0;
        out.k_ = 0;
        out.exp_ = 0;
        out.frac_ = 0;
        return out;
    }

    if (frac >= 2)
    {
        frac >>= 1;
        e += 1;
    }

    if (e >= (1 << ebits))
    {
        e -= (1 << ebits);
        k += 1;
    }

    ap_ufixed<fbits * 2, 2> rfrac = round_to(frac, fbits - 1);

    if (rfrac >= 2)
    {
        rfrac >>= 1;
        e += 1;
    }
    if (e >= (1 << ebits))
    {
        e -= (1 << ebits);
        k += 1;
    }

    out.sign_ = sign;
    out.k_ = (ap_int<kbits>)k;
    out.exp_ = (ap_uint<ebits>)e;
    out.frac_ = (ap_ufixed<fbits + 1, 1>)rfrac;
    return out;
}

// ============================================================
// Packed Posit class (nbits, ebits) using fast encode/decode
// ============================================================
template <int nbits, int ebits>
class Posit
{
public:
    static_assert(nbits >= 3, "Posit: nbits too small");
    static_assert(ebits >= 0, "Posit: ebits must be >= 0");

    // Fraction bits available excluding hidden bit
    // Minimal regime consumes 2 bits (run + terminating)
    static constexpr int fbits = nbits - ebits - 3;
    static_assert(fbits >= 0, "Posit: nbits too small for given ebits");

    // Enough bits to store k for worst-case regime length (bounded)
    static constexpr int kbits = clog2<nbits>::value + 1;

    using unpacked_t = posit_unpacked<kbits, ebits, fbits>;

    Posit()
    {
#pragma HLS INLINE
        bits_ = 0;
    }

    Posit(const Posit &other)
    {
#pragma HLS INLINE
        bits_ = other.bits_;
    }

    Posit &operator=(const Posit &other)
    {
#pragma HLS INLINE
        bits_ = other.bits_;
        return *this;
    }

    explicit Posit(const unpacked_t &u)
    {
#pragma HLS INLINE
        bits_ = u.template encode<nbits, ebits>();
    }

    // ----------------------------
    // Constructors from primitives
    // ----------------------------
    explicit Posit(int c)
    {
#pragma HLS INLINE
        from_int((ap_int<32>)c);
    }

    explicit Posit(unsigned int c)
    {
#pragma HLS INLINE
        from_uint((ap_uint<32>)c);
    }

    explicit Posit(float c)
    {
#pragma HLS INLINE
        ap_uint<32> b = bitcast_u32(c);

        FloatXUnpacked<8, 23> fx;
        fx.template decode<32, 8>(b);

        unpacked_t u;
        u.setKEFromTotalExp((ap_int<32>)fx.exp_);
        u.sign_ = fx.sign_;

        if (23 > fbits)
            u.frac_ = round_to(fx.frac_, fbits - 1);
        else
            u.frac_ = fx.frac_;

        bits_ = u.template encode<nbits, ebits>();
    }

    explicit Posit(double c)
    {
#pragma HLS INLINE
        ap_uint<64> b = bitcast_u64(c);

        FloatXUnpacked<11, 52> fx;
        fx.template decode<64, 11>(b);

        unpacked_t u;
        u.setKEFromTotalExp((ap_int<32>)fx.exp_);
        u.sign_ = fx.sign_;

        if (52 > fbits)
            u.frac_ = round_to(fx.frac_, fbits - 1);
        else
            u.frac_ = fx.frac_;

        bits_ = u.template encode<nbits, ebits>();
    }

    template <int fnbits, int fibits>
    explicit Posit(ap_fixed<fnbits, fibits> c)
    {
#pragma HLS INLINE
        // (Simple conversion: normalize to [1,2) and set exponent)
        unpacked_t u;

        if (c == 0)
        {
            bits_ = 0;
            return;
        }

        bool sign = (c < 0);
        ap_ufixed<fnbits, fibits> mag = sign ? (ap_ufixed<fnbits, fibits>)(-c) : (ap_ufixed<fnbits, fibits>)c;

        // find exponent by shifting down until <2
        ap_int<32> exp = 0;
        ap_ufixed<fnbits, fibits> m = mag;

        // NOTE: this loop is bounded (fnbits). Unroll if fnbits is small and fixed.
    NORM_LOOP:
        for (int i = 0; i < fnbits; i++)
        {
#pragma HLS UNROLL
            if (m >= 2)
            {
                m >>= 1;
                exp++;
            }
        }

        u.setKEFromTotalExp(exp);
        u.sign_ = sign;

        // Build frac_ with hidden bit + fraction bits
        // m in [1,2). Map to frac_.
        ap_ufixed<fbits + 1, 1> pf = m; // trunc/resize ok
        u.frac_ = pf;

        bits_ = u.template encode<nbits, ebits>();
    }

    // ----------------------------
    // Conversions back out
    // ----------------------------
    operator unpacked_t() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);
        return u;
    }

    operator int() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);

        if (u.frac_ == 0)
            return 0;

        ap_int<32> exp = u.getTotalExp();
        ap_fixed<fbits * 2 + 4, fbits + 2> v = (ap_fixed<fbits * 2 + 4, fbits + 2>)u.frac_;
        v = v << (int)exp;

        int r = (int)v;
        return u.sign_ ? -r : r;
    }

    operator unsigned int() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);

        if (u.frac_ == 0)
            return 0;

        ap_int<32> exp = u.getTotalExp();
        ap_ufixed<fbits * 2 + 4, fbits + 2> v = (ap_ufixed<fbits * 2 + 4, fbits + 2>)u.frac_;
        v = v << (int)exp;

        return (unsigned int)v;
    }

    operator float() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);

        FloatXUnpacked<8, 23> fx;
        fx.sign_ = u.sign_;
        fx.exp_ = u.getTotalExp();

        if (fbits > 23)
            fx.frac_ = round_to(u.frac_, 22);
        else
            fx.frac_ = u.frac_;

        ap_uint<32> b = fx.template encode<32, 8>();
        return bitcast_f32(b);
    }

    operator double() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);

        FloatXUnpacked<11, 52> fx;
        fx.sign_ = u.sign_;
        fx.exp_ = u.getTotalExp();

        if (fbits > 52)
            fx.frac_ = round_to(u.frac_, 51);
        else
            fx.frac_ = u.frac_;

        ap_uint<64> b = fx.template encode<64, 11>();
        return bitcast_f64(b);
    }

    template <int fnbits, int fibits>
    operator ap_fixed<fnbits, fibits>() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);

        if (u.frac_ == 0)
            return 0;

        ap_int<32> exp = u.getTotalExp();
        ap_fixed<fnbits + 8, fibits + 4> v = (ap_fixed<fnbits + 8, fibits + 4>)u.frac_;
        v = v << (int)exp;

        ap_fixed<fnbits, fibits> r = (ap_fixed<fnbits, fibits>)v;
        return u.sign_ ? (ap_fixed<fnbits, fibits>)(-r) : r;
    }

    // ----------------------------
    // Basic ops (packed -> unpacked)
    // ----------------------------
    unpacked_t unpack() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);
        return u;
    }

    void pack(const unpacked_t &u)
    {
#pragma HLS INLINE
        bits_ = u.template encode<nbits, ebits>();
    }

    // returning unpacked results matches your current pattern
    unpacked_t operator+(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        return unpack() + rhs;
    }
    unpacked_t operator-(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        return unpack() - rhs;
    }
    unpacked_t operator*(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        return posit_mult(unpack(), rhs);
    }
    unpacked_t operator/(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        return unpack() / rhs;
    }
    unpacked_t operator-() const
    {
#pragma HLS INLINE
        return -unpack();
    }

    // ----------------------------
    // “Mathy” helpers (packed->packed)
    // (You had free functions calling methods that were commented out.)
    // ----------------------------
    Posit fabs() const
    {
#pragma HLS INLINE
        unpacked_t u = unpack();
        u.sign_ = 0;
        return Posit(u);
    }

    Posit floor() const
    {
#pragma HLS INLINE
        // crude but synthesizable: shift frac by exp, clear fractional bits, renormalize
        unpacked_t u = unpack();
        if (u.frac_ == 0)
            return Posit(u);

        ap_int<32> texp = u.getTotalExp();

        ap_fixed<fbits * 2 + 8, fbits + 2> v = (ap_fixed<fbits * 2 + 8, fbits + 2>)u.frac_;
        v = v << (int)texp;

        // clear fractional part
        const int frac_mask_bits = fbits + 2;
        if (u.sign_ && (v(frac_mask_bits - 1, 0) != 0))
        {
            v(frac_mask_bits - 1, 0) = 0;
            v += 1;
        }
        else
        {
            v(frac_mask_bits - 1, 0) = 0;
        }

        // renormalize back to [1,2)
        ap_int<32> exp = texp;
        ap_ufixed<fbits + 1, 1> m = (ap_ufixed<fbits + 1, 1>)(v >> (int)exp);

        // ensure m in [1,2)
        if (m >= 2)
        {
            m >>= 1;
            exp += 1;
        }
        if (m < 1)
        {
            m <<= 1;
            exp -= 1;
        }

        unpacked_t r;
        r.sign_ = u.sign_;
        r.frac_ = m;
        r.setKEFromTotalExp(exp);
        return Posit(r);
    }

    Posit round() const
    {
#pragma HLS INLINE
        // round-to-nearest: v += 0.5 then floor (cheap-ish)
        unpacked_t u = unpack();
        if (u.frac_ == 0)
            return Posit(u);

        ap_int<32> texp = u.getTotalExp();
        ap_fixed<fbits * 2 + 8, fbits + 2> v = (ap_fixed<fbits * 2 + 8, fbits + 2>)u.frac_;
        v = v << (int)texp;

        // add 0.5 in integer domain
        v += 0.5;

        // clear fractional bits
        const int frac_mask_bits = fbits + 2;
        v(frac_mask_bits - 1, 0) = 0;

        // renormalize
        ap_int<32> exp = texp;
        ap_ufixed<fbits + 1, 1> m = (ap_ufixed<fbits + 1, 1>)(v >> (int)exp);
        if (m >= 2)
        {
            m >>= 1;
            exp += 1;
        }
        if (m < 1)
        {
            m <<= 1;
            exp -= 1;
        }

        unpacked_t r;
        r.sign_ = u.sign_;
        r.frac_ = m;
        r.setKEFromTotalExp(exp);
        return Posit(r);
    }

    Posit ceil() const
    {
#pragma HLS INLINE
        unpacked_t u = unpack();
        if (u.frac_ == 0)
            return Posit(u);

        ap_int<32> texp = u.getTotalExp();

        ap_fixed<fbits * 2 + 8, fbits + 2> v = (ap_fixed<fbits * 2 + 8, fbits + 2>)u.frac_;
        v = v << (int)texp;

        const int frac_mask_bits = fbits + 2;

        if (!u.sign_ && (v(frac_mask_bits - 1, 0) != 0))
        {
            v(frac_mask_bits - 1, 0) = 0;
            v += 1;
        }
        else
        {
            v(frac_mask_bits - 1, 0) = 0;
        }

        // renormalize
        ap_int<32> exp = texp;
        ap_ufixed<fbits + 1, 1> m = (ap_ufixed<fbits + 1, 1>)(v >> (int)exp);
        if (m >= 2)
        {
            m >>= 1;
            exp += 1;
        }
        if (m < 1)
        {
            m <<= 1;
            exp -= 1;
        }

        unpacked_t r;
        r.sign_ = u.sign_;
        r.frac_ = m;
        r.setKEFromTotalExp(exp);
        return Posit(r);
    }

private:
    ap_uint<nbits> bits_;

    void from_int(ap_int<32> c)
    {
#pragma HLS INLINE
        if (c == 0)
        {
            bits_ = 0;
            return;
        }

        bool sign = (c < 0);
        ap_uint<32> mag = sign ? (ap_uint<32>)(-c) : (ap_uint<32>)c;

        ap_int<32> exp = 0;
        ap_ufixed<fbits + 4, 4> m = (ap_ufixed<fbits + 4, 4>)mag;

        // normalize to [1,2)
    INT_NORM:
        for (int i = 0; i < 32; i++)
        {
#pragma HLS UNROLL
            if (m >= 2.0)
            {
                m >>= 1;
                exp++;
            }
        }

        unpacked_t u;
        u.setKEFromTotalExp(exp);
        u.sign_ = sign;
        u.frac_ = (ap_ufixed<fbits + 1, 1>)m;

        bits_ = u.template encode<nbits, ebits>();
    }

    void from_uint(ap_uint<32> c)
    {
#pragma HLS INLINE
        if (c == 0)
        {
            bits_ = 0;
            return;
        }

        ap_int<32> exp = 0;
        ap_ufixed<fbits + 4, 4> m = (ap_ufixed<fbits + 4, 4>)c;

    UINT_NORM:
        for (int i = 0; i < 32; i++)
        {
#pragma HLS UNROLL
            if (m >= 2.0)
            {
                m >>= 1;
                exp++;
            }
        }

        unpacked_t u;
        u.setKEFromTotalExp(exp);
        u.sign_ = 0;
        u.frac_ = (ap_ufixed<fbits + 1, 1>)m;

        bits_ = u.template encode<nbits, ebits>();
    }
};

// Free-function wrappers (now these exist and compile)
template <int nbits, int ebits>
static inline Posit<nbits, ebits> fabs(const Posit<nbits, ebits> &p)
{
#pragma HLS INLINE
    return p.fabs();
}
template <int nbits, int ebits>
static inline Posit<nbits, ebits> floor(const Posit<nbits, ebits> &p)
{
#pragma HLS INLINE
    return p.floor();
}
template <int nbits, int ebits>
static inline Posit<nbits, ebits> round(const Posit<nbits, ebits> &p)
{
#pragma HLS INLINE
    return p.round();
}
template <int nbits, int ebits>
static inline Posit<nbits, ebits> ceil(const Posit<nbits, ebits> &p)
{
#pragma HLS INLINE
    return p.ceil();
}
