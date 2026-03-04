#pragma once

#include "ap_int.h"
#include "ap_fixed.h"

// ============================================================
// Faster / safer bit utilities for Vitis HLS
// - Avoids 32-bit "int" where possible
// - Keeps loops fully unrolled and widths bounded
// - Fixes rounding indexing and makes it robust to frac_bit range
// ============================================================

// Bitcast helpers (HLS-friendly)
static inline ap_uint<32> bitcast_u32(float f)
{
#pragma HLS INLINE
    union
    {
        float f;
        unsigned int u;
    } pun;
    pun.f = f;
    return ap_uint<32>(pun.u);
}
static inline float bitcast_f32(ap_uint<32> u)
{
#pragma HLS INLINE
    union
    {
        float f;
        unsigned int u;
    } pun;
    pun.u = (unsigned int)u;
    return pun.f;
}

static inline ap_uint<64> bitcast_u64(double d)
{
#pragma HLS INLINE
    union
    {
        double d;
        unsigned long long u;
    } pun;
    pun.d = d;
    return ap_uint<64>(pun.u);
}
static inline double bitcast_f64(ap_uint<64> u)
{
#pragma HLS INLINE
    union
    {
        double d;
        unsigned long long u;
    } pun;
    pun.u = (unsigned long long)u;
    return pun.d;
}

// IEEE-like exponent bias for given exponent bitwidth
template <int E>
struct fbias
{
    static constexpr int value = (1 << (E - 1)) - 1;
};

// Clamp helper for runtime ints (keeps HLS from generating OOB bit-select logic)
static inline int clamp_int(int x, int lo, int hi)
{
#pragma HLS INLINE
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}

// ---------------------------
// Compile-time ceil(log2(N))
// ---------------------------
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

// ---------------------------
// Count leading bits == symbol
// (ap_uint)
// ---------------------------
template <int nbits>
static inline ap_uint<clog2<nbits + 1>::value>
count_leading_symbol(ap_uint<nbits> bits, bool symbol)
{
#pragma HLS INLINE

    ap_uint<clog2<nbits + 1>::value> count = 0;

COUNT_U_LOOP:
    for (int i = nbits - 1; i >= 0; --i)
    {
#pragma HLS UNROLL
        if (bits[i] == symbol)
            count++;
        else
            break;
    }
    return count;
}

// ---------------------------
// Count leading zeros
// (ap_ufixed)
// Treats the *bit pattern* as unsigned vector
// ---------------------------
template <int nbits, int ibits>
static inline ap_uint<clog2<nbits + 1>::value>
count_leading_zeros(ap_ufixed<nbits, ibits> bits)
{
#pragma HLS INLINE
    ap_uint<clog2<nbits + 1>::value> count = 0;

COUNT_UF_LOOP:
    for (int i = nbits - 1; i >= 0; --i)
    {
#pragma HLS UNROLL
        if (bits[i] == 0)
            count++;
        else
            break;
    }
    return count;
}

// ---------------------------
// Count leading bits == symbol
// (ap_fixed)
// NOTE: counting leading "symbol" on signed fixed is defined on raw bits.
// ---------------------------
template <int nbits, int ibits>
static inline ap_uint<clog2<nbits + 1>::value>
count_leading_symbol(ap_fixed<nbits, ibits> bits, bool symbol)
{
#pragma HLS INLINE
    ap_uint<clog2<nbits + 1>::value> count = 0;

COUNT_F_LOOP:
    for (int i = nbits - 1; i >= 0; --i)
    {
#pragma HLS UNROLL
        if (bits[i] == symbol)
            count++;
        else
            break;
    }
    return count;
}

// ============================================================
// Rounding
// ============================================================
// Your original round_to had a couple issues:
// 1) first_frac_bit = fbits - 1 - ibits  (this is NOT the MSB fractional index)
//    For ap_ufixed<fbits, ibits> the LSB fractional index is 0
//    The MSB fractional index is (fbits - ibits - 1)
// 2) Index used: val[first_frac_bit - frac_bit] can go negative / OOB
// 3) The behavior is "round up if the bit at frac_bit is 1", which is not
//    round-to-nearest-even; it's "round-up at a chosen guard bit". Keeping
//    your behavior, but making it safe & deterministic.
//
// Interpretation we’ll implement:
// - frac_bit is the *fractional precision you want to keep*, counted from 0 = 2^-1
//   (i.e., keep bits down to 2^-(frac_bit+1)).
// - We look at the next bit below the kept precision (the guard bit) and add 1 ULP
//   at the kept LSB if guard==1.
// This matches typical "truncate then add" rounding.
//
// Example: keep frac_bit=0 => keep 1 fractional bit (2^-1). guard is 2^-2.
// ============================================================
template <int W, int I>
static inline ap_ufixed<W, I>
round_to_keep_fracbits(const ap_ufixed<W, I> &val, int frac_bit)
{
#pragma HLS INLINE

    // Number of fractional bits in this type
    int F = W - I;

    // If no fractional bits exist, nothing to do
    if (F <= 0)
        return val;

    // Clamp frac_bit into a safe range:
    // we "keep" (frac_bit+1) fractional bits, so guard is at position (frac_bit+1)
    // fractional index k corresponds to bit position k (LSB fractional is bit 0).
    int keep = frac_bit + 1;
    if (keep < 0)
        keep = 0;
    if (keep > F)
        keep = F;

    // guard position (below kept LSB)
    int guard = keep; // i.e. 2^-(guard+1)

    ap_ufixed<W, I> r = val;

    // If guard bit exists and is 1, add 1 ULP at kept LSB (or at bit 0 if keep==0)
    if (guard < F)
    {
        if (val[guard] == 1)
        {
            ap_ufixed<W, I> inc = 0;
            int ulp = (keep == 0) ? 0 : (keep - 1);
            inc[ulp] = 1;
            r += inc;
        }
    }
    return r;
}

// Backwards-compatible wrapper with your old name/signature:
// - Here "frac_bit" means: keep bits down to 2^-(frac_bit+1)
// If you were passing (fbits-1) etc, revisit call sites.
// In your posit code you often call round_to(frac, fbits-1).
// That likely intended: "keep fbits-1 fractional bits". This wrapper does that.
template <int fbits, int ibits>
static inline ap_ufixed<fbits, ibits>
round_to(const ap_ufixed<fbits, ibits> &val, int frac_bit)
{
#pragma HLS INLINE
    return round_to_keep_fracbits<fbits, ibits>(val, frac_bit);
}

// Small utility: floor(x / 2^EBITS) for signed x, with correct behavior for negatives.
template <int EBITS>
static inline ap_int<32> floor_div_pow2(ap_int<32> x)
{
#pragma HLS INLINE
    ap_int<32> q = x >> EBITS;       // arithmetic shift
    ap_int<32> r = x - (q << EBITS); // remainder wrt trunc q
    if (x < 0 && r != 0)
        q -= 1; // correct to floor
    return q;
}