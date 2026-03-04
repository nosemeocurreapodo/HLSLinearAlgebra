#pragma once

#include "ap_int.h"
#include "ap_fixed.h"

// Use your optimized common.h (count_leading_symbol / round_to)
#include "common.h"

// ============================================================
// Optimized FloatX for Vitis HLS
// - Fixes correctness issues (zero handling, out-of-range rounding indices)
// - Removes reinterpret_cast bit-punning (uses unions)
// - Avoids wide "int" where possible (uses bounded ap_int/ap_uint)
// - Keeps decode/encode mostly as fixed slices + minimal muxing
// - Normalization uses your count_leading_simbol(...) (assumed unrolled)
// - Preserves your "no denormals / no NaN/Inf" simplified model:
//   * bits==0 => zero
//   * any nonzero => implicit leading 1 in frac_
// ============================================================

// ============================================================
// Unpacked FloatX: sign + unbiased exponent + [1,2) fraction
// ============================================================
template <int ebits, int fbits>
class FloatXUnpacked
{
public:
    bool sign_;
    ap_int<ebits + 2> exp_;        // unbiased exponent (a bit of headroom)
    ap_ufixed<fbits + 1, 1> frac_; // includes implicit 1 at bit fbits when nonzero

    FloatXUnpacked()
    {
#pragma HLS INLINE
        sign_ = 0;
        exp_ = 0;
        frac_ = 0;
    }

    // ----------------------------
    // Decode from packed float-like
    // ----------------------------
    template <int in_nbits, int in_ebits>
    void decode(const ap_uint<in_nbits> &bits)
    {
#pragma HLS INLINE
        static_assert(in_nbits == 1 + in_ebits + (in_nbits - in_ebits - 1), "bad layout");
        constexpr int IN_FBITS = in_nbits - 1 - in_ebits; // stored fraction bits (no implicit 1)

        // zero
        if (bits == 0)
        {
            sign_ = 0;
            exp_ = 0;
            frac_ = 0;
            return;
        }

        sign_ = bits[in_nbits - 1];

        ap_uint<in_ebits> bexp = bits(in_nbits - 2, in_nbits - 1 - in_ebits);

        // Build full mantissa in [1,2): implicit 1 + IN_FBITS fraction bits
        ap_ufixed<IN_FBITS + 1, 1> m;
        m[IN_FBITS] = 1;
        if (IN_FBITS > 0)
            m(IN_FBITS - 1, 0) = bits(IN_FBITS - 1, 0);

        // Convert biased exponent to unbiased:
        // unbiased = bexp - bias
        // Your previous code used: bexp - 2^(E-1) + 1 == bexp - bias
        exp_ = (ap_int<ebits + 2>)((ap_int<in_ebits + 1>)bexp - (ap_int<in_ebits + 1>)fbias<in_ebits>::value);

        // Resize / round mantissa to this FloatXUnpacked fbits
        if ((IN_FBITS) > (fbits))
        {
            // need to round from IN_FBITS to fbits
            // keep fbits fractional bits => pass frac_bit = fbits-1 (your convention)
            ap_ufixed<IN_FBITS + 1, 1> mr = round_to(m, fbits - 1);
            frac_ = (ap_ufixed<fbits + 1, 1>)mr;
        }
        else
        {
            frac_ = (ap_ufixed<fbits + 1, 1>)m;
        }

        // Handle rounding overflow: if frac_ became 2.0, renormalize and bump exponent
        if (frac_ >= 2)
        {
            frac_ >>= 1;
            exp_ += 1;
        }
    }

    // ----------------------------
    // Encode to packed float-like
    // ----------------------------
    template <int out_nbits, int out_ebits>
    ap_uint<out_nbits> encode() const
    {
#pragma HLS INLINE
        constexpr int OUT_FBITS = out_nbits - 1 - out_ebits;

        ap_uint<out_nbits> bits = 0;

        // zero
        if (frac_ == 0)
            return 0;

        // Normalize (defensive): ensure frac_ in [1,2)
        ap_int<ebits + 2> e = exp_;
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

        // Bias exponent
        ap_int<out_ebits + 2> bexp = (ap_int<out_ebits + 2>)e + (ap_int<out_ebits + 2>)fbias<out_ebits>::value;

        // (No NaN/Inf/denorm handling here; clamp exponent to range to avoid OOB)
        // Range of biased exponent: [0 .. 2^out_ebits - 1]
        const int max_bexp = (1 << out_ebits) - 1;
        int bexp_i = (int)bexp;
        bexp_i = clamp_int(bexp_i, 0, max_bexp);
        ap_uint<out_ebits> bexp_u = (ap_uint<out_ebits>)bexp_i;

        // Mantissa: need OUT_FBITS fraction bits stored (implicit 1 not stored)
        // m is [1,2). Its implicit bit is at position OUT_FBITS in type ap_ufixed<OUT_FBITS+1,1>.
        ap_ufixed<OUT_FBITS + 1, 1> mout;

        if (fbits > OUT_FBITS)
        {
            // reduce precision, round
            // keep OUT_FBITS fractional bits => frac_bit = OUT_FBITS-1
            ap_ufixed<fbits + 1, 1> mr = round_to(m, OUT_FBITS - 1);
            mout = (ap_ufixed<OUT_FBITS + 1, 1>)mr;
        }
        else
        {
            mout = (ap_ufixed<OUT_FBITS + 1, 1>)m;
        }

        // If rounding caused overflow to 2.0, renormalize and increment exponent (with clamp)
        if (mout >= 2)
        {
            mout >>= 1;
            int bi = bexp_i + 1;
            if (bi > max_bexp)
                bi = max_bexp;
            bexp_u = (ap_uint<out_ebits>)bi;
        }

        bits[out_nbits - 1] = sign_;
        bits(out_nbits - 2, out_nbits - 1 - out_ebits) = bexp_u;

        if (OUT_FBITS > 0)
            bits(OUT_FBITS - 1, 0) = mout(OUT_FBITS - 1, 0);

        return bits;
    }

    // =========================================================
    // Arithmetic (cleaned + reduced width where possible)
    // =========================================================
    FloatXUnpacked operator-() const
    {
#pragma HLS INLINE
        FloatXUnpacked r = *this;
        r.sign_ = !sign_;
        return r;
    }

    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        FloatXUnpacked out;

        // handle zeros quickly
        if (frac_ == 0)
            return rhs;
        if (rhs.frac_ == 0)
            return *this;

        ap_int<ebits + 3> diff = (ap_int<ebits + 3>)(exp_ - rhs.exp_);

        ap_fixed<fbits + 3, 3> a = (ap_fixed<fbits + 3, 3>)frac_;
        ap_fixed<fbits + 3, 3> b = (ap_fixed<fbits + 3, 3>)rhs.frac_;

        ap_int<ebits + 2> e;
        bool s;

        // Align smaller to bigger
        if (diff >= 0)
        {
            e = exp_;
            s = sign_;
            b = b >> (int)diff;
            if (sign_ != rhs.sign_)
                b = -b;
        }
        else
        {
            e = rhs.exp_;
            s = rhs.sign_;
            a = a >> (int)(-diff);
            if (sign_ != rhs.sign_)
                a = -a;
        }

        ap_fixed<fbits + 4, 4> sum = a + b;

        // if subtraction led to negative, flip sign and abs
        if (sum < 0)
        {
            s = !s;
            sum = -sum;
        }

        if (sum == 0)
        {
            out.sign_ = 0;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }

        // Normalize sum into [1,2)
        // sum is ap_fixed; reinterpret as unsigned for leading-zero count:
        ap_uint<fbits + 4> usum = sum.range(fbits + 3, 0);

        // count leading zeros in the magnitude (MSB side), then compute shift
        int lz = (int)count_leading_symbol(usum, 0);
        // For a (fbits+4)-bit unsigned magnitude, MSB index = fbits+3
        // We want leading 1 to land at bit position (fbits+2) for [1,2) in ap_ufixed<fbits+2,1>
        // A simpler approach:
        // - If sum >= 2 => shift right 1 and e++
        // - Else while sum < 1 => shift left and e--
        // Use single-step normalization plus an LZ-based multi-shift:
        ap_ufixed<fbits + 2, 2> m = (ap_ufixed<fbits + 2, 2>)sum;

        if (m >= 2)
        {
            m >>= 1;
            e += 1;
        }
        else if (m < 1)
        {
            // shift left by as much as possible in one go
            // compute how far from MSB the first 1 is.
            // For m width (fbits+2), integer bits=2; leading one for [1,2) is at bit (fbits)
            // We'll use a loop-free approximate: use lz from usum and shift accordingly.
            // usum width is fbits+4, first '1' target at bit (fbits+2) (0-based)
            int first_one_pos = (fbits + 3) - lz; // 0..fbits+3
            int target_pos = (fbits + 2);         // want it here
            int sh = target_pos - first_one_pos;  // positive => shift left
            if (sh < 0)
                sh = 0;
            if (sh > (fbits + 2))
                sh = (fbits + 2);

            m <<= sh;
            e -= sh;

            // one more safety step
            if (m < 1)
            {
                m <<= 1;
                e -= 1;
            }
        }

        // Round to fbits (keep fbits fractional bits -> pass fbits-1)
        ap_ufixed<fbits + 2, 2> mr = round_to(m, fbits - 1);

        // Rounding overflow
        if (mr >= 2)
        {
            mr >>= 1;
            e += 1;
        }

        out.sign_ = s;
        out.exp_ = e;
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
        {
            out.sign_ = 0;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }

        bool s = sign_ ^ rhs.sign_;
        ap_int<ebits + 3> e = (ap_int<ebits + 3>)(exp_ + rhs.exp_);
        ap_ufixed<2 * fbits + 2, 2> m = (ap_ufixed<2 * fbits + 2, 2>)(frac_ * rhs.frac_);

        if (m >= 2)
        {
            m >>= 1;
            e += 1;
        }

        // Round product down to target precision
        ap_ufixed<2 * fbits + 2, 2> mr = round_to(m, fbits - 1);

        if (mr >= 2)
        {
            mr >>= 1;
            e += 1;
        }

        out.sign_ = s;
        out.exp_ = e;
        out.frac_ = (ap_ufixed<fbits + 1, 1>)mr;
        return out;
    }

    FloatXUnpacked operator/(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        FloatXUnpacked out;

        if (frac_ == 0)
            return out; // zero

        if (rhs.frac_ == 0)
        {
            // simplified: "inf-ish" not supported -> return zero
            // (if you want Inf/NaN, we can add it properly)
            out.sign_ = 0;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }

        bool s = sign_ ^ rhs.sign_;
        ap_int<ebits + 3> e = (ap_int<ebits + 3>)(exp_ - rhs.exp_);

        ap_ufixed<2 * fbits + 2, 2> m = (ap_ufixed<2 * fbits + 2, 2>)(frac_ / rhs.frac_);

        // Normalize into [1,2)
        if (m < 1)
        {
            m <<= 1;
            e -= 1;
        }
        else if (m >= 2)
        {
            m >>= 1;
            e += 1;
        }

        ap_ufixed<2 * fbits + 2, 2> mr = round_to(m, fbits - 1);

        if (mr >= 2)
        {
            mr >>= 1;
            e += 1;
        }
        if (mr < 1)
        {
            mr <<= 1;
            e -= 1;
        }

        out.sign_ = s;
        out.exp_ = e;
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

    // Packed->unpacked ops (same style as your posit class)
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
