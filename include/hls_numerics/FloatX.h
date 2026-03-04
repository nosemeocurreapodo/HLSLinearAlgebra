#pragma once
#include "ap_int.h"
#include "ap_fixed.h"
#include "common.h" // bitcast_u32/u64, bitcast_f32/f64, clamp_int, fbias<>, count_leading_symbol

// ============================================================
// Bitfield RNE rounding for mantissa in [1,2)
// Mantissa bitfield convention (integer bits only):
//   - width = 1 + FRAC
//   - bit[FRAC] is the hidden 1 (value 1.0)
//   - bit[FRAC-1] is 2^-1
//   ...
//   - bit[0] is 2^-FRAC
// ============================================================

// Round mantissa with EXTRA low bits down to OUT_FRAC fractional bits (RNE).
// Input: m has width = 1 + OUT_FRAC + EXTRA, with hidden 1 at bit (OUT_FRAC+EXTRA).
// Output: rounded mantissa width = 1 + OUT_FRAC, hidden 1 at bit OUT_FRAC.
// Returns: rounded mantissa and a carry_out meaning "overflow to 2.0" (needs renorm).
template <int OUT_FRAC, int EXTRA>
static inline ap_uint<1 + OUT_FRAC> rne_round_mantissa(ap_uint<1 + OUT_FRAC + EXTRA> m, bool &carry_out)
{
#pragma HLS INLINE
    // Keep top (1+OUT_FRAC) bits
    ap_uint<1 + OUT_FRAC> keep = m.range(OUT_FRAC + EXTRA, EXTRA);

    // Guard bit is the next bit below kept LSB
    bool guard = (EXTRA >= 1) ? (bool)m[EXTRA - 1] : false;

    // Sticky is OR of all bits below guard
    bool sticky = false;
    if constexpr (EXTRA >= 2)
    {
#pragma HLS INLINE
        ap_uint<EXTRA - 1> low = m.range(EXTRA - 2, 0);
        sticky = (low != 0);
    }

    // LSB of kept part (for ties-to-even)
    bool lsb = (bool)keep[0];

    // RNE increment condition
    bool inc = guard && (sticky || lsb);

    ap_uint<1 + OUT_FRAC> rounded = keep + (ap_uint<1 + OUT_FRAC>)inc;

    // If rounding overflowed (e.g. 1.111.. + 1 ulp => 10.000..)
    carry_out = (rounded[OUT_FRAC] == 0); // hidden bit dropped -> overflow
    if (carry_out)
    {
        // After overflow, value is exactly 2.0 in this fixed field; renorm is handled outside.
        // rounded currently wrapped; set it to 1.0 (i.e., shift right by 1 will be done outside).
        // Easiest: restore as 1.0 with zero fraction
        rounded = 0;
        rounded[OUT_FRAC] = 1;
    }

    return rounded;
}

// Convert mantissa bitfield to ap_ufixed<FRAC+1,1> safely
template <int FRAC>
static inline ap_ufixed<FRAC + 1, 1> mant_bits_to_ufixed(const ap_uint<FRAC + 1> &mb)
{
#pragma HLS INLINE
    ap_ufixed<FRAC + 1, 1> u;
    u.range(FRAC, 0) = mb;
    return u;
}

// Convert ap_ufixed<FRAC+1,1> to mantissa bitfield safely
template <int FRAC>
static inline ap_uint<FRAC + 1> ufixed_to_mant_bits(const ap_ufixed<FRAC + 1, 1> &u)
{
#pragma HLS INLINE
    return u.range(FRAC, 0);
}

// ============================================================
// Unpacked FloatX: sign + unbiased exponent + mantissa [1,2)
// ============================================================
template <int ebits, int fbits>
class FloatXUnpacked
{
public:
    bool sign_;
    ap_int<ebits + 6> exp_;        // unbiased exponent
    ap_ufixed<fbits + 1, 1> frac_; // [1,2) hidden bit included

    FloatXUnpacked()
    {
#pragma HLS INLINE
        sign_ = 0;
        exp_ = 0;
        frac_ = 0;
    }

    // Decode packed -> unpacked (no rounding)
    template <int in_nbits, int in_ebits>
    void decode(const ap_uint<in_nbits> &bits)
    {
#pragma HLS INLINE
        constexpr int IN_FBITS = in_nbits - 1 - in_ebits;

        if (bits == 0)
        {
            sign_ = 0;
            exp_ = 0;
            frac_ = 0;
            return;
        }

        sign_ = bits[in_nbits - 1];
        ap_uint<in_ebits> bexp = bits(in_nbits - 2, in_nbits - 1 - in_ebits);
        exp_ = (ap_int<ebits + 6>)((ap_int<in_ebits + 1>)bexp - (ap_int<in_ebits + 1>)fbias<in_ebits>::value);

        // Mantissa bits: hidden 1 + IN_FBITS fraction bits in LSBs
        ap_uint<IN_FBITS + 1> mant = 0;
        mant[IN_FBITS] = 1;
        if (IN_FBITS > 0)
            mant(IN_FBITS - 1, 0) = bits(IN_FBITS - 1, 0);

        // Resize mantissa to our fbits by trunc/zero-extend (NO rounding)
        ap_uint<fbits + 1> tgt = 0;
        tgt[fbits] = 1;

        constexpr int COPY = (IN_FBITS < fbits) ? IN_FBITS : fbits;
    COPY_LOOP:
        for (int i = 0; i < COPY; ++i)
        {
#pragma HLS UNROLL
            tgt[fbits - 1 - i] = mant[IN_FBITS - 1 - i]; // 2^-(i+1)
        }

        frac_ = mant_bits_to_ufixed<fbits>(tgt);
    }

    // Encode unpacked -> packed (ONE rounding here to OUT_FBITS using RNE)
    template <int out_nbits, int out_ebits>
    ap_uint<out_nbits> encode() const
    {
#pragma HLS INLINE
        constexpr int OUT_FBITS = out_nbits - 1 - out_ebits;

        if (frac_ == 0)
            return 0;

        // normalize defensively
        ap_int<ebits + 6> e = exp_;
        ap_uint<fbits + 1> mant = ufixed_to_mant_bits<fbits>(frac_);

        // If mantissa is not in [1,2) adjust (rare)
        // mant[fbits] should be 1 for normal nonzero
        if (mant[fbits] == 0)
        {
            // shift left until hidden bit becomes 1
            int lz = (int)count_leading_symbol(mant, 0);
            int sh = lz - 1;
            if (sh < 0)
                sh = 0;
            if (sh > fbits)
                sh = fbits;
            mant <<= sh;
            e -= sh;
        }

        // Round mantissa to OUT_FBITS using integer RNE:
        if constexpr (OUT_FBITS == fbits)
        {
            // no mantissa rounding needed
        }
        else if constexpr (OUT_FBITS < fbits)
        {
            constexpr int EXTRA = fbits - OUT_FBITS;
            ap_uint<1 + OUT_FBITS + EXTRA> mwide = mant.range(fbits, EXTRA ? 0 : 0); // take all bits (same width)
            // mwide expects hidden at (OUT_FBITS+EXTRA)
            // Our mant hidden is at fbits, so this matches.
            bool carry = false;
            ap_uint<1 + OUT_FBITS> mround = rne_round_mantissa<OUT_FBITS, EXTRA>(mwide, carry);
            if (carry)
            {
                // overflow -> renorm (value was 2.0), set mant=1.0 and increment exponent
                e += 1;
            }
            // replace mant with rounded-sized version, then expand back for packing
            ap_uint<fbits + 1> newmant = 0;
            newmant[fbits] = 1;
            // copy rounded fraction bits into the top bits of our mant field
            // rounded hidden at OUT_FBITS -> map to fbits
            // easiest: construct pack mant directly later; here we just use mround for packing
            // We'll pack using mround below.
            // (so skip storing back into newmant)
            mant = 0; // not used further in this branch
            // pack directly from mround below
            // compute biased exponent now:
            const int max_bexp = (1 << out_ebits) - 1;
            int bexp_i = (int)((ap_int<out_ebits + 2>)e + (ap_int<out_ebits + 2>)fbias<out_ebits>::value);
            bexp_i = clamp_int(bexp_i, 0, max_bexp);
            ap_uint<out_ebits> bexp_u = (ap_uint<out_ebits>)bexp_i;

            ap_uint<out_nbits> out = 0;
            out[out_nbits - 1] = sign_;
            out(out_nbits - 2, out_nbits - 1 - out_ebits) = bexp_u;
            if (OUT_FBITS > 0)
                out(OUT_FBITS - 1, 0) = mround(OUT_FBITS - 1, 0);
            return out;
        }
        else
        {
            // OUT_FBITS > fbits: widen with zeros, no rounding needed
        }

        // If we get here, either OUT_FBITS==fbits or widening case:
        // Bias exponent and pack using current mant
        const int max_bexp = (1 << out_ebits) - 1;
        int bexp_i = (int)((ap_int<out_ebits + 2>)e + (ap_int<out_ebits + 2>)fbias<out_ebits>::value);
        bexp_i = clamp_int(bexp_i, 0, max_bexp);
        ap_uint<out_ebits> bexp_u = (ap_uint<out_ebits>)bexp_i;

        ap_uint<out_nbits> out = 0;
        out[out_nbits - 1] = sign_;
        out(out_nbits - 2, out_nbits - 1 - out_ebits) = bexp_u;

        if constexpr (OUT_FBITS > 0)
        {
            // take top OUT_FBITS fraction bits from our mant field
            // mant hidden at fbits -> fraction MSB is mant[fbits-1]
            ap_uint<OUT_FBITS> frac_field = 0;
            constexpr int COPY = (OUT_FBITS < fbits) ? OUT_FBITS : fbits;
        PACK_FRAC:
            for (int i = 0; i < COPY; ++i)
            {
#pragma HLS UNROLL
                frac_field[OUT_FBITS - 1 - i] = mant[fbits - 1 - i];
            }
            out(OUT_FBITS - 1, 0) = frac_field;
        }
        return out;
    }

    // Unary minus
    FloatXUnpacked operator-() const
    {
#pragma HLS INLINE
        FloatXUnpacked r = *this;
        r.sign_ = !sign_;
        return r;
    }

    // One-rounding-per-op: do internal math with EXTRA bits, then RNE once to fbits.
    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        if (frac_ == 0)
            return rhs;
        if (rhs.frac_ == 0)
            return *this;

        constexpr int G = 5; // more guard bits for safety
        constexpr int MF = fbits + G;

        ap_int<ebits + 6> ea = exp_;
        ap_int<ebits + 6> eb = rhs.exp_;

        ap_uint<fbits + 1> ma0 = ufixed_to_mant_bits<fbits>(frac_);
        ap_uint<fbits + 1> mb0 = ufixed_to_mant_bits<fbits>(rhs.frac_);

        // expand mantissas with guard zeros at LSB
        ap_uint<MF + 1> ma = ((ap_uint<MF + 1>)ma0) << G;
        ap_uint<MF + 1> mb = ((ap_uint<MF + 1>)mb0) << G;

        // align by exponent
        ap_int<ebits + 7> diff = (ap_int<ebits + 7>)(ea - eb);

        ap_int<ebits + 6> e = ea;
        if (diff >= 0)
        {
            int sh = (int)diff;
            if (sh >= (MF + 1))
                mb = 0;
            else
                mb >>= sh;
            e = ea;
        }
        else
        {
            int sh = (int)(-diff);
            if (sh >= (MF + 1))
                ma = 0;
            else
                ma >>= sh;
            e = eb;
        }

        // signed add in wider signed container
        ap_int<MF + 3> sa = (ap_int<MF + 3>)ma;
        ap_int<MF + 3> sb = (ap_int<MF + 3>)mb;
        if (sign_)
            sa = -sa;
        if (rhs.sign_)
            sb = -sb;

        ap_int<MF + 4> sum = sa + sb;

        FloatXUnpacked out;
        if (sum == 0)
        {
            out.sign_ = 0;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }

        bool s = (sum < 0);
        ap_uint<MF + 3> um = s ? (ap_uint<MF + 3>)(-sum) : (ap_uint<MF + 3>)sum;

        // normalize so hidden bit ends at position MF (i.e., value in [1,2))
        // hidden bit should be um[MF] == 1
        if (um[MF + 1]) // >=2
        {
            um >>= 1;
            e += 1;
        }
        else if (!um[MF]) // <1
        {
            int lz = (int)count_leading_symbol(um, 0);
            // um width is MF+3, MSB index MF+2, want first '1' at MF
            int first_one = (MF + 2) - lz;
            int sh = MF - first_one;
            if (sh < 0)
                sh = 0;
            if (sh > (MF + 1))
                sh = (MF + 1);
            um <<= sh;
            e -= sh;
        }

        // Now round from MF fractional bits down to fbits fractional bits in one RNE step.
        // Current mantissa bitfield width is (MF+1) (hidden at MF).
        // We want output mantissa width (fbits+1) (hidden at fbits).
        constexpr int EXTRA = MF - fbits;
        ap_uint<1 + fbits + EXTRA> mwide = um.range(MF, EXTRA ? 0 : 0); // hidden at (fbits+EXTRA)
        bool carry = false;
        ap_uint<1 + fbits> mround = rne_round_mantissa<fbits, EXTRA>(mwide, carry);
        if (carry)
            e += 1;

        out.sign_ = s;
        out.exp_ = e;
        out.frac_ = mant_bits_to_ufixed<fbits>(mround);
        return out;
    }

    FloatXUnpacked operator-(const FloatXUnpacked &rhs) const
    {
#pragma HLS INLINE
        FloatXUnpacked t = rhs;
        t.sign_ = !t.sign_;
        return (*this) + t;
    }
};

// ============================================================
// Packed FloatX with FloatX + FloatX -> FloatX (matches your test)
// ============================================================
template <int nbits, int ebits>
class FloatX
{
public:
    static constexpr int fbits = nbits - ebits - 1;
    using unpacked_t = FloatXUnpacked<ebits, fbits>;

    FloatX()
    {
#pragma HLS INLINE bits_ = 0;
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

    operator double() const
    {
#pragma HLS INLINE
        FloatXUnpacked<11, 52> u;
        u.template decode<nbits, ebits>(bits_);
        ap_uint<64> b = u.template encode<64, 11>();
        return bitcast_f64(b);
    }

    // FloatX + FloatX -> FloatX (your test)
    FloatX operator+(const FloatX &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a, b;
        a.template decode<nbits, ebits>(bits_);
        b.template decode<nbits, ebits>(rhs.bits_);
        unpacked_t s = a + b;
        FloatX out;
        out.bits_ = s.template encode<nbits, ebits>();
        return out;
    }

private:
    ap_uint<nbits> bits_;
};