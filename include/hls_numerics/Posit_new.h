#pragma once
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

#include "common.h"
#include "FloatX.h"

// ============================================================
// Helpers: integer mantissa bitfields + RNE
// Mantissa bitfield convention (same as FloatX rewrite):
//  - width = 1 + FRAC
//  - bit[FRAC] is hidden 1 (value 1.0)
//  - bit[FRAC-1] is 2^-1
//  - ...
//  - bit[0] is 2^-FRAC
// ============================================================

template <int FRAC>
static inline ap_uint<FRAC + 1> ufixed_to_mant_bits(const ap_ufixed<FRAC + 1, 1> &u)
{
#pragma HLS INLINE
    return u.range(FRAC, 0);
}

template <int FRAC>
static inline ap_ufixed<FRAC + 1, 1> mant_bits_to_ufixed(const ap_uint<FRAC + 1> &mb)
{
#pragma HLS INLINE
    ap_ufixed<FRAC + 1, 1> u;
    u.range(FRAC, 0) = mb;
    return u;
}

// RNE rounding: input mantissa has width (1+OUT_FRAC+EXTRA), hidden bit at (OUT_FRAC+EXTRA).
// output mantissa width (1+OUT_FRAC), hidden at OUT_FRAC.
// carry_out indicates rounding overflow to 2.0 => renormalize and increment exponent.
template <int OUT_FRAC, int EXTRA>
static inline ap_uint<1 + OUT_FRAC>
rne_round_mantissa(ap_uint<1 + OUT_FRAC + EXTRA> m, bool &carry_out)
{
#pragma HLS INLINE
    ap_uint<1 + OUT_FRAC> keep = m.range(OUT_FRAC + EXTRA, EXTRA);

    bool guard = (EXTRA >= 1) ? (bool)m[EXTRA - 1] : false;

    bool sticky = false;
    if constexpr (EXTRA >= 2)
    {
        ap_uint<EXTRA - 1> low = m.range(EXTRA - 2, 0);
        sticky = (low != 0);
    }

    bool lsb = (bool)keep[0];
    bool inc = guard && (sticky || lsb);

    ap_uint<1 + OUT_FRAC> rounded = keep + (ap_uint<1 + OUT_FRAC>)inc;

    // overflow if hidden bit flipped to 0 due to carry out (wrap)
    carry_out = (rounded[OUT_FRAC] == 0);
    if (carry_out)
    {
        rounded = 0;
        rounded[OUT_FRAC] = 1; // represent 1.0; renorm handled outside
    }
    return rounded;
}

// clamp shift to avoid "shift >= width" corner behavior
template <typename T>
static inline int clamp_shift(int sh, int maxw)
{
#pragma HLS INLINE
    if (sh < 0) return 0;
    if (sh > maxw) return maxw;
    return sh;
}

// ============================================================
// Unpacked posit container: sign, regime k, exponent, mantissa [1,2)
// ============================================================
template <int kbits, int ebits, int fbits>
class posit_unpacked
{
public:
    bool sign_;
    ap_int<kbits> k_;
    ap_uint<ebits> exp_;
    ap_ufixed<fbits + 1, 1> frac_; // hidden bit included

    static constexpr int large_k = kbits + ebits + fbits;

    posit_unpacked()
    {
#pragma HLS INLINE
        sign_ = 0;
        k_ = 0;
        exp_ = 0;
        frac_ = 0;
    }

    // total exponent: k*2^ebits + exp
    ap_int<32> getTotalExp() const
    {
#pragma HLS INLINE
        return (ap_int<32>)k_ * (ap_int<32>)(1 << ebits) + (ap_int<32>)exp_;
    }

    // floor-divide by 2^ebits (fast) and remainder in [0, 2^ebits-1]
    void setKEFromTotalExp(ap_int<32> in_exp)
    {
#pragma HLS INLINE
        ap_int<32> k = floor_div_pow2<ebits>(in_exp);
        ap_int<32> e = in_exp - (k << ebits);
        if (e < 0)
        {
            e += (1 << ebits);
            k -= 1;
        }
        k_ = (ap_int<kbits>)k;
        exp_ = (ap_uint<ebits>)e;
    }

    // =========================================================
    // Encode / Decode (bitfield-based, no state machine)
    // =========================================================
    template <int out_nbits, int out_ebits>
    ap_uint<out_nbits> encode() const
    {
#pragma HLS INLINE
        ap_uint<out_nbits> out = 0;

        // zero
        if (frac_ == 0) return 0;

        out[out_nbits - 1] = sign_;

        // regime: run length
        bool regbit = (k_ < 0);
        ap_uint<out_nbits> run = (k_ >= 0) ? (ap_uint<out_nbits>)(k_ + 1) : (ap_uint<out_nbits>)(-k_);

        // saturate if regime consumes everything
        if (run >= (out_nbits - 1))
        {
            out(out_nbits - 2, 0) = regbit ? ap_uint<out_nbits - 1>(-1) : ap_uint<out_nbits - 1>(0);
            return out;
        }

        int idx = out_nbits - 2;

    ENCODE_REG:
        for (int i = 0; i < out_nbits - 1; ++i)
        {
#pragma HLS UNROLL
            if (i < (int)run) out[idx - i] = regbit;
        }
        idx -= (int)run;

        // terminating bit
        out[idx] = !regbit;
        idx -= 1;

        // exponent bits
        int expbits = (idx >= 0) ? hls::min((int)out_ebits, idx + 1) : 0;
        if (expbits > 0)
        {
            out(idx, idx - expbits + 1) = exp_(out_ebits - 1, out_ebits - expbits);
            idx -= expbits;
        }

        // fraction bits: use mantissa bitfield, drop hidden 1
        int fracbits = (idx >= 0) ? (idx + 1) : 0;
        if (fracbits > 0)
        {
            ap_uint<fbits + 1> mant = ufixed_to_mant_bits<fbits>(frac_);
            // want top fracbits from mant[fbits-1:0]
            // pack into out[idx:0] MSB-first
        PACK_FRAC:
            for (int i = 0; i < fracbits; ++i)
            {
#pragma HLS UNROLL
                out[idx - i] = mant[fbits - 1 - i];
            }
        }

        return out;
    }

    template <int in_nbits, int in_ebits>
    void decode(const ap_uint<in_nbits> &bits)
    {
#pragma HLS INLINE
        if (bits == 0)
        {
            sign_ = 0; k_ = 0; exp_ = 0; frac_ = 0;
            return;
        }

        sign_ = bits[in_nbits - 1];
        bool regbit = bits[in_nbits - 2];

        constexpr int PAY_W = in_nbits - 2;
        ap_uint<PAY_W> payload = bits(PAY_W - 1, 0);

        // run length of regbit from MSB of payload
        ap_uint<clog2<PAY_W + 1>::value> run = count_leading_symbol<PAY_W>(payload, regbit);

        if (run >= PAY_W)
        {
            // saturated/special
            k_ = (ap_int<kbits>)large_k;
            exp_ = 0;
            frac_ = 1.0;
            return;
        }

        // k
        k_ = (regbit == 0) ? (ap_int<kbits>)((ap_int<32>)run - 1)
                           : (ap_int<kbits>)(-(ap_int<32>)run);

        // drop run + terminating bit
        ap_uint<PAY_W> rest = payload << (run + 1);
        int avail = PAY_W - (int)(run + 1);

        // exponent bits
        exp_ = 0;
        int expbits = (avail > 0) ? hls::min((int)in_ebits, avail) : 0;
        if (expbits > 0)
        {
            exp_ = rest(PAY_W - 1, PAY_W - expbits);
            rest <<= expbits;
            avail -= expbits;
        }

        // fraction bits -> build mantissa bitfield
        ap_uint<fbits + 1> mant = 0;
        mant[fbits] = 1;

        int fracbits = (avail > 0) ? hls::min((int)fbits, avail) : 0;
        if (fracbits > 0)
        {
        UNPACK_FRAC:
            for (int i = 0; i < fracbits; ++i)
            {
#pragma HLS UNROLL
                mant[fbits - 1 - i] = rest[PAY_W - 1 - i];
            }
        }

        frac_ = mant_bits_to_ufixed<fbits>(mant);
    }

    // =========================================================
    // Arithmetic: one rounding per operation (RNE via bitfields)
    // =========================================================

    posit_unpacked operator-() const
    {
#pragma HLS INLINE
        posit_unpacked r = *this;
        r.sign_ = !sign_;
        return r;
    }

    // Add (one rounding at the end)
    posit_unpacked operator+(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        if (frac_ == 0) return rhs;
        if (rhs.frac_ == 0) return *this;

        // Compare total exponents to align
        ap_int<32> texp_a = getTotalExp();
        ap_int<32> texp_b = rhs.getTotalExp();
        ap_int<32> diff = texp_a - texp_b;

        // guard bits for safe alignment + sticky
        constexpr int G = 6;
        constexpr int MF = fbits + G;

        ap_uint<fbits + 1> ma0 = ufixed_to_mant_bits<fbits>(frac_);
        ap_uint<fbits + 1> mb0 = ufixed_to_mant_bits<fbits>(rhs.frac_);

        ap_uint<MF + 1> ma = ((ap_uint<MF + 1>)ma0) << G;
        ap_uint<MF + 1> mb = ((ap_uint<MF + 1>)mb0) << G;

        ap_int<32> base_texp = texp_a;
        bool base_sign = sign_;

        if (diff >= 0)
        {
            int sh = clamp_shift((int)diff, MF + 1);
            // sticky: if shift drops bits, fold into LSB
            if (sh > 0 && sh <= (MF + 1))
            {
                ap_uint<MF + 1> dropped = mb & ((ap_uint<MF + 1>)((1ULL << sh) - 1));
                mb >>= sh;
                if (dropped != 0) mb[0] = 1;
            }
            base_texp = texp_a;
            base_sign = sign_;
        }
        else
        {
            int sh = clamp_shift((int)(-diff), MF + 1);
            if (sh > 0 && sh <= (MF + 1))
            {
                ap_uint<MF + 1> dropped = ma & ((ap_uint<MF + 1>)((1ULL << sh) - 1));
                ma >>= sh;
                if (dropped != 0) ma[0] = 1;
            }
            base_texp = texp_b;
            base_sign = rhs.sign_;
        }

        // signed add
        ap_int<MF + 3> sa = (ap_int<MF + 3>)ma;
        ap_int<MF + 3> sb = (ap_int<MF + 3>)mb;
        if (sign_)     sa = -sa;
        if (rhs.sign_) sb = -sb;

        ap_int<MF + 4> sum = sa + sb;

        posit_unpacked out;
        if (sum == 0)
        {
            out.sign_ = 0; out.k_ = 0; out.exp_ = 0; out.frac_ = 0;
            return out;
        }

        bool s = (sum < 0);
        ap_uint<MF + 3> um = s ? (ap_uint<MF + 3>)(-sum) : (ap_uint<MF + 3>)sum;

        ap_int<32> texp = base_texp;

        // normalize so hidden bit at MF
        if (um[MF + 1]) // >=2
        {
            um >>= 1;
            texp += 1;
        }
        else if (!um[MF]) // <1
        {
            int lz = (int)count_leading_symbol(um, 0);
            int first_one = (MF + 2) - lz;
            int sh = MF - first_one;
            sh = clamp_shift(sh, MF + 1);
            um <<= sh;
            texp -= sh;
        }

        // one RNE rounding from MF -> fbits
        constexpr int EXTRA = MF - fbits;
        ap_uint<1 + fbits + EXTRA> mwide = um.range(MF, 0); // width = 1+MF
        // mwide expects hidden at fbits+EXTRA == MF OK
        bool carry = false;
        ap_uint<1 + fbits> mround = rne_round_mantissa<fbits, EXTRA>(mwide, carry);
        if (carry) texp += 1;

        out.sign_ = s;
        out.setKEFromTotalExp(texp);
        out.frac_ = mant_bits_to_ufixed<fbits>(mround);
        return out;
    }

    posit_unpacked operator-(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        posit_unpacked t = rhs;
        t.sign_ = !t.sign_;
        return (*this) + t;
    }

    // Multiply (one rounding at the end)
    posit_unpacked operator*(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        posit_unpacked out;
        if (frac_ == 0 || rhs.frac_ == 0)
            return out;

        bool s = sign_ ^ rhs.sign_;

        ap_int<32> texp = getTotalExp() + rhs.getTotalExp();

        ap_uint<fbits + 1> ma = ufixed_to_mant_bits<fbits>(frac_);
        ap_uint<fbits + 1> mb = ufixed_to_mant_bits<fbits>(rhs.frac_);

        // Multiply mantissas: widths add
        ap_uint<2 * (fbits + 1)> prod = (ap_uint<2 * (fbits + 1)>)ma * (ap_uint<2 * (fbits + 1)>)mb;

        // prod represents [1,4). hidden position is (2*fbits)
        // Normalize to [1,2): if top bit indicates >=2, shift right and inc exp
        constexpr int PH = 2 * fbits; // target hidden after normalization (we'll pick a consistent slice)
        // Build a convenient window with extra bits for rounding
        // We'll keep (fbits + EXTRA) fractional bits from normalized mantissa.
        constexpr int G = 6;
        constexpr int OUT_MF = fbits + G;

        // Create normalized mantissa bitfield with hidden at OUT_MF
        // Start by aligning prod's hidden at PH (bit index 2*fbits)
        // If prod >= 2 -> prod bit (2*fbits+1) == 1
        ap_uint<2 * (fbits + 1)> p = prod;
        if (p[2 * fbits + 1])
        {
            p >>= 1;
            texp += 1;
        }

        // Now hidden is at bit (2*fbits)
        // We want to extract 1 + OUT_MF bits with hidden at OUT_MF:
        // take bits [2*fbits : 2*fbits-OUT_MF]
        ap_uint<1 + OUT_MF> mnorm = 0;
        mnorm = p.range(2 * fbits, 2 * fbits - OUT_MF);

        // One rounding from OUT_MF -> fbits
        constexpr int EXTRA = OUT_MF - fbits;
        bool carry = false;
        ap_uint<1 + fbits> mround = rne_round_mantissa<fbits, EXTRA>(mnorm, carry);
        if (carry) texp += 1;

        out.sign_ = s;
        out.setKEFromTotalExp(texp);
        out.frac_ = mant_bits_to_ufixed<fbits>(mround);
        return out;
    }

    // Divide (one rounding at the end)
    posit_unpacked operator/(const posit_unpacked &rhs) const
    {
#pragma HLS INLINE
        posit_unpacked out;

        // div by zero => your saturated convention
        if (rhs.frac_ == 0)
        {
            out.sign_ = 0;
            out.k_ = (ap_int<kbits>)large_k;
            out.exp_ = 0;
            out.frac_ = 0;
            return out;
        }
        if (frac_ == 0)
            return out;

        bool s = sign_ ^ rhs.sign_;
        ap_int<32> texp = getTotalExp() - rhs.getTotalExp();

        ap_uint<fbits + 1> na = ufixed_to_mant_bits<fbits>(frac_);
        ap_uint<fbits + 1> nb = ufixed_to_mant_bits<fbits>(rhs.frac_);

        // Fixed-point division with extra bits:
        constexpr int G = 6;
        constexpr int OUT_MF = fbits + G;
        // We want quotient mantissa with hidden at OUT_MF:
        // compute q = (na << OUT_MF) / nb  => q has ~ (1+OUT_MF) bits
        ap_uint<(fbits + 1) + OUT_MF + 2> num = ((ap_uint<(fbits + 1) + OUT_MF + 2>)na) << OUT_MF;
        ap_uint<1 + OUT_MF + 2> q = (ap_uint<1 + OUT_MF + 2>)(num / nb);

        // Normalize q into [1,2): ensure hidden at OUT_MF
        if (q[OUT_MF + 1]) // >=2
        {
            q >>= 1;
            texp += 1;
        }
        else if (!q[OUT_MF]) // <1
        {
            q <<= 1;
            texp -= 1;
        }

        // One rounding OUT_MF -> fbits (q currently has hidden at OUT_MF, plus maybe 1 spare bit)
        ap_uint<1 + OUT_MF> mnorm = q.range(OUT_MF, 0);

        constexpr int EXTRA = OUT_MF - fbits;
        bool carry = false;
        ap_uint<1 + fbits> mround = rne_round_mantissa<fbits, EXTRA>(mnorm, carry);
        if (carry) texp += 1;

        out.sign_ = s;
        out.setKEFromTotalExp(texp);
        out.frac_ = mant_bits_to_ufixed<fbits>(mround);
        return out;
    }
};

// ============================================================
// Packed Posit<nbits,ebits>
// ============================================================
template <int nbits, int ebits>
class Posit
{
public:
    static_assert(nbits >= 3, "Posit: nbits too small");
    static constexpr int fbits = nbits - ebits - 3;
    static_assert(fbits >= 0, "Posit: nbits too small for given ebits");

    static constexpr int kbits = clog2<nbits>::value + 1;
    using unpacked_t = posit_unpacked<kbits, ebits, fbits>;

    Posit() { #pragma HLS INLINE bits_ = 0; }
    Posit(const Posit &o) { #pragma HLS INLINE bits_ = o.bits_; }
    Posit &operator=(const Posit &o) { #pragma HLS INLINE bits_ = o.bits_; return *this; }

    explicit Posit(const unpacked_t &u)
    {
#pragma HLS INLINE
        bits_ = u.template encode<nbits, ebits>();
    }

    explicit Posit(float c)
    {
#pragma HLS INLINE
        ap_uint<32> b = bitcast_u32(c);
        FloatXUnpacked<8, 23> fx;
        fx.template decode<32, 8>(b);

        unpacked_t u;
        u.sign_ = fx.sign_;
        u.setKEFromTotalExp((ap_int<32>)fx.exp_);

        // copy mantissa bits using bitfields then RNE once to posit fbits
        ap_uint<23 + 1> fm = fx.frac_.range(23, 0); // hidden at 23
        if constexpr (23 > fbits)
        {
            constexpr int EXTRA = 23 - fbits;
            bool carry = false;
            ap_uint<1 + fbits> pm = rne_round_mantissa<fbits, EXTRA>(fm, carry);
            if (carry) u.setKEFromTotalExp((ap_int<32>)fx.exp_ + 1);
            u.frac_ = mant_bits_to_ufixed<fbits>(pm);
        }
        else
        {
            ap_uint<fbits + 1> pm = 0;
            pm[fbits] = 1;
            // map 23 fraction bits into posit fraction field
            constexpr int COPY = (23 < fbits) ? 23 : fbits;
        MAPF:
            for (int i = 0; i < COPY; ++i)
            {
#pragma HLS UNROLL
                pm[fbits - 1 - i] = fm[23 - 1 - i];
            }
            u.frac_ = mant_bits_to_ufixed<fbits>(pm);
        }

        bits_ = u.template encode<nbits, ebits>();
    }

    explicit Posit(double c)
    {
#pragma HLS INLINE
        ap_uint<64> b = bitcast_u64(c);
        FloatXUnpacked<11, 52> fx;
        fx.template decode<64, 11>(b);

        unpacked_t u;
        u.sign_ = fx.sign_;
        u.setKEFromTotalExp((ap_int<32>)fx.exp_);

        ap_uint<52 + 1> fm = fx.frac_.range(52, 0); // hidden at 52
        if constexpr (52 > fbits)
        {
            constexpr int EXTRA = 52 - fbits;
            bool carry = false;
            ap_uint<1 + fbits> pm = rne_round_mantissa<fbits, EXTRA>(fm, carry);
            ap_int<32> texp = (ap_int<32>)fx.exp_ + (carry ? 1 : 0);
            u.setKEFromTotalExp(texp);
            u.frac_ = mant_bits_to_ufixed<fbits>(pm);
        }
        else
        {
            ap_uint<fbits + 1> pm = 0;
            pm[fbits] = 1;
            constexpr int COPY = (52 < fbits) ? 52 : fbits;
        MAPD:
            for (int i = 0; i < COPY; ++i)
            {
#pragma HLS UNROLL
                pm[fbits - 1 - i] = fm[52 - 1 - i];
            }
            u.frac_ = mant_bits_to_ufixed<fbits>(pm);
        }

        bits_ = u.template encode<nbits, ebits>();
    }

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

    // Packed arithmetic (return packed, like FloatX)
    Posit operator+(const Posit &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a = unpack();
        unpacked_t b = rhs.unpack();
        unpacked_t s = a + b;
        return Posit(s);
    }

    Posit operator-(const Posit &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a = unpack();
        unpacked_t b = rhs.unpack();
        unpacked_t d = a - b;
        return Posit(d);
    }

    Posit operator*(const Posit &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a = unpack();
        unpacked_t b = rhs.unpack();
        unpacked_t p = a * b;
        return Posit(p);
    }

    Posit operator/(const Posit &rhs) const
    {
#pragma HLS INLINE
        unpacked_t a = unpack();
        unpacked_t b = rhs.unpack();
        unpacked_t q = a / b;
        return Posit(q);
    }

    operator double() const
    {
#pragma HLS INLINE
        unpacked_t u = unpack();

        if (u.frac_ == 0) return 0.0;

        FloatXUnpacked<11, 52> fx;
        fx.sign_ = u.sign_;
        fx.exp_  = u.getTotalExp();

        // map posit mantissa to 52-bit float mantissa via bitfield widen (no rounding needed)
        ap_uint<fbits + 1> pm = u.frac_.range(fbits, 0);
        ap_uint<52 + 1> fm = 0;
        fm[52] = 1;
        constexpr int COPY = (fbits < 52) ? fbits : 52;
    MAPOUT:
        for (int i = 0; i < COPY; ++i)
        {
#pragma HLS UNROLL
            fm[52 - 1 - i] = pm[fbits - 1 - i];
        }
        fx.frac_.range(52, 0) = fm;

        ap_uint<64> b = fx.template encode<64, 11>();
        return bitcast_f64(b);
    }

private:
    ap_uint<nbits> bits_;
};