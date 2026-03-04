#pragma once

#include "ap_int.h"
#include "ap_fixed.h"
#include "common.h"

// ============================================================
// FloatX: simplified float-like format for HLS
// Layout: [sign][exp(out_ebits)][frac(OUT_FBITS)]
// - Zero is all bits == 0
// - No NaN/Inf/denormals (you can add later if needed)
// - Unpacked representation uses:
//     sign_  : bool
//     exp_   : unbiased exponent (signed)
//     frac_  : mantissa in [1,2) with hidden bit included (ap_ufixed<fbits+1,1>)
// ============================================================
//
// Key fixes / optimizations vs your current version:
// 1) Addition normalization now counts leading zeros on *m's own bit-width*,
//    not on some wider "sum" slice. This fixes large errors like your -50.25+25.125 case.
// 2) All variable shifts are clamped to avoid "shift by >= width" corner behaviors.
// 3) Rounding overflow is handled consistently (m -> 2.0 => shift right, exp++).
// 4) No reinterpret_cast bit-punning; relies on bitcast helpers expected in common.h
//    (bitcast_u32/u64 and bitcast_f32/f64). If your common.h doesn't have them,
//    add the union versions there.
// ============================================================

template <int ebits, int fbits>
class FloatXUnpacked
{
public:
    bool sign_;
    ap_int<ebits + 3> exp_;          // unbiased exponent with headroom
    ap_ufixed<fbits + 1, 1> frac_;   // mantissa in [1,2), hidden bit included when nonzero

    FloatXUnpacked()
    {
#pragma HLS INLINE
        sign_ = 0;
        exp_  = 0;
        frac_ = 0;
    }

    // -----------------------------------------
    // Decode packed float-like into unpacked
    // -----------------------------------------
    template <int in_nbits, int in_ebits>
    void decode(const ap_uint<in_nbits> &bits)
    {
#pragma HLS INLINE
        static_assert(in_nbits >= 1 + in_ebits, "FloatXUnpacked::decode: bad sizes");
        constexpr int IN_FBITS = in_nbits - 1 - in_ebits;

        if (bits == 0)
        {
            sign_ = 0;
            exp_  = 0;
            frac_ = 0;
            return;
        }

        sign_ = bits[in_nbits - 1];

        ap_uint<in_ebits> bexp = bits(in_nbits - 2, in_nbits - 1 - in_ebits);

        // unbiased exponent = bexp - bias
        exp_ = (ap_int<ebits + 3>)((ap_int<in_ebits + 1>)bexp - (ap_int<in_ebits + 1>)fbias<in_ebits>::value);

        // build mantissa in [1,2): implicit 1 + IN_FBITS frac bits
        ap_ufixed<IN_FBITS + 1, 1> m;
        m[IN_FBITS] = 1;
        if (IN_FBITS > 0)
            m(IN_FBITS - 1, 0) = bits(IN_FBITS - 1, 0);

        // resize/round to local fbits
        if (IN_FBITS > fbits)
        {
            ap_ufixed<IN_FBITS + 1, 1> mr = round_to(m, fbits - 1);
            frac_ = (ap_ufixed<fbits + 1, 1>)mr;
        }
        else
        {
            frac_ = (ap_ufixed<fbits + 1, 1>)m;
        }

        // rounding overflow => renormalize
        if (frac_ >= 2)
        {
            frac_ >>= 1;
            exp_ += 1;
        }
    }

    // -----------------------------------------
    // Encode unpacked into packed float-like
    // -----------------------------------------
    template <int out_nbits, int out_ebits>
    ap_uint<out_nbits> encode() const
    {
#pragma HLS INLINE
        static_assert(out_nbits >= 1 + out_ebits, "FloatXUnpacked::encode: bad sizes");
        constexpr int OUT_FBITS = out_nbits - 1 - out_ebits;

        if (frac_ == 0)
            return 0;

        // Defensive normalize to [1,2)
        ap_int<ebits + 3> e = exp_;
        ap_ufixed<fbits + 1, 1> m = frac_;

        if (m >= 2)
        {
            m >>= 1;
            e += 1;
        }
        else if (m < 1)
        {
            m <<= 1;
            e -= 1;
        }

        // bias exponent and clamp to representable range (no Inf/NaN)
        const int max_bexp = (1 << out_ebits) - 1;
        int bexp_i = (int)((ap_int<out_ebits + 2>)e + (ap_int<out_ebits + 2>)fbias<out_ebits>::value);
        bexp_i = clamp_int(bexp_i, 0, max_bexp);
        ap_uint<out_ebits> bexp_u = (ap_uint<out_ebits>)bexp_i;

        // Mantissa resize
        ap_ufixed<OUT_FBITS + 1, 1> mout;
        if (fbits > OUT_FBITS)
        {
            // keep OUT_FBITS fractional bits => pass OUT_FBITS-1
            ap_ufixed<fbits + 1, 1> mr = round_to(m, OUT_FBITS - 1);
            mout = (ap_ufixed<OUT_FBITS + 1, 1>)mr;
        }
        else
        {
            mout = (ap_ufixed<OUT_FBITS + 1, 1>)m;
        }

        // rounding overflow on mantissa
        if (mout >= 2)
        {
            mout >>= 1;
            int bi = bexp_i + 1;
            if (bi > max_bexp) bi = max_bexp;
            bexp_u = (ap_uint<out_ebits>)bi;
        }

        ap_uint<out_nbits> out = 0;
        out[out_nbits - 1] = sign_;
        out(out_nbits - 2, out_nbits - 1 - out_ebits) = bexp_u;
        if (OUT_FBITS > 0)
            out(OUT_FBITS - 1, 0) = mout(OUT_FBITS - 1, 0); // drop hidden bit
        return out;
    }

    // -----------------------------------------
    // Unary negate
    // -----------------------------------------
    FloatXUnpacked operator-() const
    {
#pragma HLS INLINE
        FloatXUnpacked r = *this;
        r.sign_ = !sign_;
        return r;
    }

    // -----------------------------------------
    // Add
    // -----------------------------------------
    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        // quick zeros
        if (frac_ == 0) return rhs;
        if (rhs.frac_ == 0) return *this;

        // Compare exponents
        ap_int<ebits + 4> diff = (ap_int<ebits + 4>)(exp_ - rhs.exp_);

        // Use signed fixed for aligned mantissas
        ap_fixed<fbits + 4, 4> a = (ap_fixed<fbits + 4, 4>)frac_;
        ap_fixed<fbits + 4, 4> b = (ap_fixed<fbits + 4, 4>)rhs.frac_;

        ap_int<ebits + 3> e;
        bool s;

        // align smaller mantissa to larger exponent (clamp shifts)
        if (diff >= 0)
        {
            e = exp_;
            s = sign_;

            int sh = (int)diff;
            if (sh > (fbits + 4)) sh = (fbits + 4);
            b = b >> sh;

            if (sign_ != rhs.sign_) b = -b;
        }
        else
        {
            e = rhs.exp_;
            s = rhs.sign_;

            int sh = (int)(-diff);
            if (sh > (fbits + 4)) sh = (fbits + 4);
            a = a >> sh;

            if (sign_ != rhs.sign_) a = -a;
        }

        ap_fixed<fbits + 5, 5> sum = a + b;

        // If sum is negative, flip sign and take abs
        if (sum < 0)
        {
            s = !s;
            sum = -sum;
        }

        if (sum == 0)
        {
            FloatXUnpacked out;
            out.sign_ = 0;
            out.exp_  = 0;
            out.frac_ = 0;
            return out;
        }

        // Convert to unsigned mantissa domain
        ap_ufixed<fbits + 2, 2> m = (ap_ufixed<fbits + 2, 2>)sum;

        // ---------------------------
        // NORMALIZATION (FIXED)
        // Count leading zeros on *m* itself (fbits+2 width),
        // then shift-left by (lz - 1) when m < 1.
        // ---------------------------
        if (m >= 2)
        {
            m >>= 1;
            e += 1;
        }
        else if (m < 1)
        {
            // mbits width is exactly fbits+2 => MSB index = fbits+1
            ap_uint<fbits + 2> mbits = m.range(fbits + 1, 0);
            int lz = (int)count_leading_symbol(mbits, 0);

            int sh = lz - 1; // <-- key fix
            if (sh < 0) sh = 0;
            if (sh > (fbits + 1)) sh = (fbits + 1);

            m <<= sh;
            e -= sh;

            // safety: ensure in [1,2)
            if (m < 1)
            {
                m <<= 1;
                e -= 1;
            }
        }

        // Round to fbits precision (keep fbits fractional bits => pass fbits-1)
        ap_ufixed<fbits + 2, 2> mr = round_to(m, fbits - 1);

        // rounding overflow
        if (mr >= 2)
        {
            mr >>= 1;
            e += 1;
        }

        FloatXUnpacked out;
        out.sign_ = s;
        out.exp_  = e;
        out.frac_ = (ap_ufixed<fbits + 1, 1>)mr;
        return out;
    }

    FloatXUnpacked operator-(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        FloatXUnpacked neg = rhs;
        neg.sign_ = !rhs.sign_;
        return (*this) + neg;
    }

    FloatXUnpacked operator*(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        FloatXUnpacked out;

        if (frac_ == 0 || rhs.frac_ == 0)
            return out;

        out.sign_ = sign_ ^ rhs.sign_;
        out.exp_  = (ap_int<ebits + 3>)(exp_ + rhs.exp_);

        ap_ufixed<2 * fbits + 2, 2> m = (ap_ufixed<2 * fbits + 2, 2>)(frac_ * rhs.frac_);

        if (m >= 2)
        {
            m >>= 1;
            out.exp_ += 1;
        }

        ap_ufixed<2 * fbits + 2, 2> mr = round_to(m, fbits - 1);

        if (mr >= 2)
        {
            mr >>= 1;
            out.exp_ += 1;
        }

        out.frac_ = (ap_ufixed<fbits + 1, 1>)mr;
        return out;
    }

    FloatXUnpacked operator/(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        FloatXUnpacked out;

        if (frac_ == 0)
            return out;

        if (rhs.frac_ == 0)
        {
            // simplified (no Inf/NaN)
            return out;
        }

        out.sign_ = sign_ ^ rhs.sign_;
        out.exp_  = (ap_int<ebits + 3>)(exp_ - rhs.exp_);

        ap_ufixed<2 * fbits + 2, 2> m = (ap_ufixed<2 * fbits + 2, 2>)(frac_ / rhs.frac_);

        if (m < 1)
        {
            m <<= 1;
            out.exp_ -= 1;
        }
        else if (m >= 2)
        {
            m >>= 1;
            out.exp_ += 1;
        }

        ap_ufixed<2 * fbits + 2, 2> mr = round_to(m, fbits - 1);

        if (mr >= 2)
        {
            mr >>= 1;
            out.exp_ += 1;
        }
        if (mr < 1)
        {
            mr <<= 1;
            out.exp_ -= 1;
        }

        out.frac_ = (ap_ufixed<fbits + 1, 1>)mr;
        return out;
    }
};

// ============================================================
// Packed FloatX: custom (nbits, ebits) float-like container
// ============================================================
template <int nbits, int ebits>
class FloatX
{
public:
    static_assert(nbits >= 1 + ebits + 1, "FloatX: not enough bits");
    static constexpr int fbits = nbits - ebits - 1;

    using unpacked_t = FloatXUnpacked<ebits, fbits>;

    FloatX()
    {
#pragma HLS INLINE
        bits_ = 0;
    }

    FloatX(const FloatX &other)
    {
#pragma HLS INLINE
        bits_ = other.bits_;
    }

    FloatX &operator=(const FloatX &other)
    {
#pragma HLS INLINE
        bits_ = other.bits_;
        return *this;
    }

    FloatX(const unpacked_t &u)
    {
#pragma HLS INLINE
        bits_ = u.template encode<nbits, ebits>();
    }

    FloatX(float c)
    {
#pragma HLS INLINE
        ap_uint<32> b = bitcast_u32(c);
        FloatXUnpacked<8, 23> u;
        u.template decode<32, 8>(b);
        bits_ = u.template encode<nbits, ebits>();
    }

    FloatX(double c)
    {
#pragma HLS INLINE
        ap_uint<64> b = bitcast_u64(c);
        FloatXUnpacked<11, 52> u;
        u.template decode<64, 11>(b);
        bits_ = u.template encode<nbits, ebits>();
    }

    operator float() const
    {
#pragma HLS INLINE
        FloatXUnpacked<8, 23> u;
        u.template decode<nbits, ebits>(bits_);
        ap_uint<32> b = u.template encode<32, 8>();
        return bitcast_f32(b);
    }

    operator double() const
    {
#pragma HLS INLINE
        FloatXUnpacked<11, 52> u;
        u.template decode<nbits, ebits>(bits_);
        ap_uint<64> b = u.template encode<64, 11>();
        return bitcast_f64(b);
    }

    operator unpacked_t() const
    {
#pragma HLS INLINE
        unpacked_t u;
        u.template decode<nbits, ebits>(bits_);
        return u;
    }

    // Packed -> unpacked ops
    unpacked_t operator+(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a;
        a.template decode<nbits, ebits>(bits_);
        return a + rhs;
    }

    unpacked_t operator-(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a;
        a.template decode<nbits, ebits>(bits_);
        return a - rhs;
    }

    unpacked_t operator*(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a;
        a.template decode<nbits, ebits>(bits_);
        return a * rhs;
    }

    unpacked_t operator/(const unpacked_t &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a;
        a.template decode<nbits, ebits>(bits_);
        return a / rhs;
    }

    unpacked_t operator-() const
    {
#pragma HLS INLINE
        unpacked_t a;
        a.template decode<nbits, ebits>(bits_);
        return -a;
    }

private:
    ap_uint<nbits> bits_;
};