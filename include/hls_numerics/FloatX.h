#pragma once

#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"

#include "common.h"

// HLS-oriented custom floating-point type.
// Notes / intentional limitations:
// - Normal numbers only: subnormals are flushed to zero.
// - Rounding is simple round-to-nearest using guard bits where practical, not full IEEE-754 RNE everywhere.
// - NaN/Inf are represented and propagated, but exception flags are not implemented.
// - This is designed to be synthesis-friendly rather than perfectly IEEE-compliant.

template <int nbits, int ebits>
class FloatXUnpacked
{
public:
    static constexpr int fbits = nbits - ebits - 1;
    static_assert(nbits >= 4, "FloatX requires at least 4 bits");
    static_assert(ebits >= 2, "FloatX requires at least 2 exponent bits");
    static_assert(fbits >= 1, "FloatX requires at least 1 mantissa bit");

    static constexpr int bias = fbias<ebits>::value;

    FloatXUnpacked()
        : sign_(0), exp_(0), mant_(0), zero_(1), inf_(0), nan_(0)
    {
    }

    static FloatXUnpacked zero(bool sign = false)
    {
        FloatXUnpacked z;
        z.sign_ = sign;
        z.zero_ = 1;
        z.inf_ = 0;
        z.nan_ = 0;
        z.exp_ = 0;
        z.mant_ = 0;
        return z;
    }

    static FloatXUnpacked inf(bool sign = false)
    {
        FloatXUnpacked x;
        x.sign_ = sign;
        x.zero_ = 0;
        x.inf_ = 1;
        x.nan_ = 0;
        x.exp_ = all_exp_ones();
        x.mant_ = 0;
        return x;
    }

    static FloatXUnpacked nan()
    {
        FloatXUnpacked x;
        x.sign_ = 0;
        x.zero_ = 0;
        x.inf_ = 0;
        x.nan_ = 1;
        x.exp_ = all_exp_ones();
        x.mant_ = 1;
        return x;
    }

    template <int onbits, int oebits>
    FloatXUnpacked(const FloatXUnpacked<onbits, oebits> &other)
        : sign_(other.sign_), exp_(0), mant_(0), zero_(other.zero_), inf_(other.inf_), nan_(other.nan_)
    {
        static constexpr int ofbits = onbits - oebits - 1;

        if (other.nan_)
        {
            *this = nan();
            return;
        }
        if (other.inf_)
        {
            *this = inf(other.sign_);
            return;
        }
        if (other.zero_)
        {
            *this = zero(other.sign_);
            return;
        }

        const ap_int<oebits + 2> unbiased =
            (ap_int<oebits + 2>)((ap_uint<1>(0), other.exp_)) -
            (ap_int<oebits + 2>)fbias<oebits>::value;
        ap_int<ebits + 2> new_exp = unbiased + (ap_int<ebits + 2>)bias;

        if (new_exp <= 0)
        {
            *this = zero(other.sign_); // flush subnormals/underflow
            return;
        }
        if (new_exp >= (ap_int<ebits + 2>)all_exp_ones())
        {
            *this = inf(other.sign_);
            return;
        }

        sign_ = other.sign_;
        zero_ = 0;
        inf_ = 0;
        nan_ = 0;
        exp_ = new_exp(ebits - 1, 0);
        mant_ = 0;

        if constexpr (fbits >= ofbits)
        {
            mant_(fbits - 1, fbits - ofbits) = other.mant_(ofbits - 1, 0);
        }
        else
        {
            // Truncate plus simple round-to-nearest using the first discarded bit.
            mant_ = other.mant_(ofbits - 1, ofbits - fbits);
            const bool guard = other.mant_[ofbits - fbits - 1];
            if (guard)
                increment_mantissa_or_exp();
        }
    }

    template <int in_nbits>
    FloatXUnpacked(ap_int<in_nbits> c)
        : sign_(0), exp_(0), mant_(0), zero_(1), inf_(0), nan_(0)
    {
        const bool s = c < 0;
        ap_uint<in_nbits> mag = s ? ap_uint<in_nbits>(-c) : ap_uint<in_nbits>(c);
        from_unsigned_magnitude<in_nbits>(mag, s);
    }

    template <int in_nbits>
    FloatXUnpacked(ap_uint<in_nbits> c)
        : sign_(0), exp_(0), mant_(0), zero_(1), inf_(0), nan_(0)
    {
        from_unsigned_magnitude<in_nbits>(c, false);
    }

    FloatXUnpacked(int c) : FloatXUnpacked(ap_int<32>(c)) {}
    FloatXUnpacked(unsigned int c) : FloatXUnpacked(ap_uint<32>(c)) {}

    template <int in_nbits>
    ap_int<in_nbits> to_ap_int() const
    {
        if (zero_ || nan_)
            return 0;

        if (inf_)
            return sign_ ? min_ap_int<in_nbits>() : max_ap_int<in_nbits>();

        const int e = (int)exp_ - bias;
        if (e < 0)
            return 0;

        if (e >= in_nbits - 1)
            return sign_ ? min_ap_int<in_nbits>() : max_ap_int<in_nbits>();

        ap_uint<fbits + 1> sig = (ap_uint<1>(1), mant_);
        ap_uint<in_nbits> mag = 0;

        if (e >= fbits)
            mag = (ap_uint<in_nbits>)sig << (e - fbits);
        else
            mag = (ap_uint<in_nbits>)(sig >> (fbits - e));

        ap_int<in_nbits> out = (ap_int<in_nbits>)mag;
        return sign_ ? ap_int<in_nbits>(-out) : out;
    }

    operator int() const { return (int)to_ap_int<32>(); }
    operator unsigned int() const { return (unsigned int)((ap_uint<32>)to_ap_int<32>()); }

    void decode(const ap_uint<nbits> &bits)
    {
        sign_ = bits[nbits - 1];
        exp_ = bits(nbits - 2, fbits);
        mant_ = bits(fbits - 1, 0);

        const bool exp_zero = exp_ == 0;
        const bool exp_ones = exp_ == all_exp_ones();
        const bool mant_zero = mant_ == 0;

        zero_ = exp_zero && mant_zero;
        inf_ = exp_ones && mant_zero;
        nan_ = exp_ones && !mant_zero;
    }

    ap_uint<nbits> encode() const
    {
        ap_uint<nbits> bits = 0;
        bits[nbits - 1] = sign_;

        if (nan_)
        {
            bits(nbits - 2, fbits) = all_exp_ones();
            bits(fbits - 1, 0) = 1;
            return bits;
        }

        if (inf_)
        {
            bits(nbits - 2, fbits) = all_exp_ones();
            bits(fbits - 1, 0) = 0;
            return bits;
        }

        if (zero_)
        {
            bits(nbits - 2, 0) = 0;
            return bits;
        }

        bits(nbits - 2, fbits) = exp_;
        bits(fbits - 1, 0) = mant_;
        return bits;
    }

    FloatXUnpacked operator+(const FloatXUnpacked &rhs) const
    {
        if (nan_ || rhs.nan_)
            return nan();

        if (inf_ && rhs.inf_)
            return (sign_ == rhs.sign_) ? inf(sign_) : nan();
        if (inf_)
            return *this;
        if (rhs.inf_)
            return rhs;

        if (zero_)
            return rhs;
        if (rhs.zero_)
            return *this;

        FloatXUnpacked a = *this;
        FloatXUnpacked b = rhs;

        // Sort by magnitude so subtraction is non-negative.
        if (!mag_ge(a, b))
        {
            FloatXUnpacked tmp = a;
            a = b;
            b = tmp;
        }

        const ap_uint<ebits> de = a.exp_ - b.exp_;
        const int shift = (int)de;

        // Extra 3 bits: guard/round/sticky-ish space.
        ap_uint<fbits + 4> fa = (ap_uint<1>(1), a.mant_, ap_uint<3>(0));
        ap_uint<fbits + 4> fb = (ap_uint<1>(1), b.mant_, ap_uint<3>(0));

        if (shift >= fbits + 4)
            fb = 0;
        else
            fb >>= shift;

        ap_uint<fbits + 5> fr;
        if (a.sign_ == b.sign_)
            fr = fa + fb;
        else
            fr = fa - fb;

        if (fr == 0)
            return zero(false);

        ap_int<ebits + 2> e = (ap_int<ebits + 2>)a.exp_;

        // For addition carry, hidden bit moved one position left.
        if (fr[fbits + 4])
        {
            fr >>= 1;
            e++;
        }
        else
        {
            // Normalize after subtraction.
            while (fr[fbits + 3] == 0 && e > 0)
            {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 64
                fr <<= 1;
                e--;
            }
        }

        FloatXUnpacked out;
        out.sign_ = a.sign_;
        out.zero_ = 0;
        out.inf_ = 0;
        out.nan_ = 0;

        if (e <= 0)
            return zero(out.sign_);
        if (e >= (ap_int<ebits + 2>)all_exp_ones())
            return inf(out.sign_);

        out.exp_ = e(ebits - 1, 0);
        out.mant_ = fr(fbits + 2, 3);

        // Simple rounding from guard bit.
        if (fr[2])
            out.increment_mantissa_or_exp();

        return out;
    }

    FloatXUnpacked operator-(const FloatXUnpacked &rhs) const
    {
        FloatXUnpacked neg_rhs = rhs;
        if (!neg_rhs.nan_)
            neg_rhs.sign_ = !rhs.sign_;
        return (*this) + neg_rhs;
    }

    FloatXUnpacked operator*(const FloatXUnpacked &rhs) const
    {
        if (nan_ || rhs.nan_)
            return nan();
        if ((zero_ && rhs.inf_) || (inf_ && rhs.zero_))
            return nan();
        if (inf_ || rhs.inf_)
            return inf(sign_ ^ rhs.sign_);
        if (zero_ || rhs.zero_)
            return zero(sign_ ^ rhs.sign_);

        FloatXUnpacked out;
        out.sign_ = sign_ ^ rhs.sign_;
        out.zero_ = 0;
        out.inf_ = 0;
        out.nan_ = 0;

        ap_int<ebits + 2> e =
            (ap_int<ebits + 2>)exp_ + (ap_int<ebits + 2>)rhs.exp_ - (ap_int<ebits + 2>)bias;

        ap_uint<fbits + 1> a = (ap_uint<1>(1), mant_);
        ap_uint<fbits + 1> b = (ap_uint<1>(1), rhs.mant_);
        ap_uint<2 * fbits + 2> p = a * b; // range [1, 4)

        if (p[2 * fbits + 1])
        {
            p >>= 1;
            e++;
        }

        if (e <= 0)
            return zero(out.sign_);
        if (e >= (ap_int<ebits + 2>)all_exp_ones())
            return inf(out.sign_);

        out.exp_ = e(ebits - 1, 0);
        out.mant_ = p(2 * fbits - 1, fbits);

        // First discarded bit after mantissa.
        if (p[fbits - 1])
            out.increment_mantissa_or_exp();

        return out;
    }

    FloatXUnpacked operator/(const FloatXUnpacked &rhs) const
    {
        if (nan_ || rhs.nan_)
            return nan();
        if ((zero_ && rhs.zero_) || (inf_ && rhs.inf_))
            return nan();
        if (rhs.zero_)
            return inf(sign_ ^ rhs.sign_);
        if (inf_)
            return inf(sign_ ^ rhs.sign_);
        if (zero_)
            return zero(sign_ ^ rhs.sign_);
        if (rhs.inf_)
            return zero(sign_ ^ rhs.sign_);

        FloatXUnpacked out;
        out.sign_ = sign_ ^ rhs.sign_;
        out.zero_ = 0;
        out.inf_ = 0;
        out.nan_ = 0;

        ap_int<ebits + 2> e =
            (ap_int<ebits + 2>)exp_ - (ap_int<ebits + 2>)rhs.exp_ + (ap_int<ebits + 2>)bias;

        ap_uint<fbits + 1> num = (ap_uint<1>(1), mant_);
        ap_uint<fbits + 1> den = (ap_uint<1>(1), rhs.mant_);

        // Keep fbits + 3 fractional bits before rounding.
        ap_uint<2 * fbits + 5> q = ((ap_uint<2 * fbits + 5>)num << (fbits + 4)) / den;

        // q is in [0.5, 2). Normalize to [1, 2).
        if (q[fbits + 4] == 0)
        {
            q <<= 1;
            e--;
        }

        if (e <= 0)
            return zero(out.sign_);
        if (e >= (ap_int<ebits + 2>)all_exp_ones())
            return inf(out.sign_);

        out.exp_ = e(ebits - 1, 0);
        out.mant_ = q(fbits + 3, 4);

        if (q[3])
            out.increment_mantissa_or_exp();

        return out;
    }

    FloatXUnpacked operator-() const
    {
        FloatXUnpacked result = *this;
        if (!result.nan_)
            result.sign_ = !sign_;
        return result;
    }

    bool operator==(const FloatXUnpacked &rhs) const
    {
        if (nan_ || rhs.nan_)
            return false;
        if (zero_ && rhs.zero_)
            return true;
        if (inf_ || rhs.inf_)
            return inf_ && rhs.inf_ && sign_ == rhs.sign_;
        return sign_ == rhs.sign_ && exp_ == rhs.exp_ && mant_ == rhs.mant_;
    }

    bool operator<(const FloatXUnpacked &rhs) const
    {
        if (nan_ || rhs.nan_)
            return false;
        if (*this == rhs)
            return false;

        if (inf_)
            return sign_; // -inf < finite, +inf is not < anything here
        if (rhs.inf_)
            return !rhs.sign_; // finite < +inf

        if (zero_ && rhs.zero_)
            return false;
        if (zero_)
            return !rhs.sign_; // 0 < positive
        if (rhs.zero_)
            return sign_; // negative < 0

        if (sign_ != rhs.sign_)
            return sign_; // negative < positive

        const bool mag_less = (exp_ < rhs.exp_) || (exp_ == rhs.exp_ && mant_ < rhs.mant_);
        return sign_ ? !mag_less : mag_less;
    }

    bool sign_;
    ap_uint<ebits> exp_;
    ap_uint<fbits> mant_;
    bool zero_;
    bool inf_;
    bool nan_;

private:
    static constexpr ap_uint<ebits> all_exp_ones()
    {
        return ap_uint<ebits>(-1);
    }

    template <int in_nbits>
    void from_unsigned_magnitude(ap_uint<in_nbits> mag, bool sign)
    {
        sign_ = sign;
        zero_ = (mag == 0);
        inf_ = 0;
        nan_ = 0;
        exp_ = 0;
        mant_ = 0;

        if (zero_)
            return;

        const int lz = count_leading_zeros(mag);
        const int msb_pos = in_nbits - 1 - lz;
        ap_int<ebits + 2> e = (ap_int<ebits + 2>)msb_pos + (ap_int<ebits + 2>)bias;

        if (e >= (ap_int<ebits + 2>)all_exp_ones())
        {
            *this = inf(sign);
            return;
        }

        exp_ = e(ebits - 1, 0);

        // Shift so the leading 1 is removed and the next fbits become mantissa.
        ap_uint<in_nbits> shifted = mag << lz;
        shifted <<= 1;

        if constexpr (fbits <= in_nbits)
        {
            mant_ = shifted(in_nbits - 1, in_nbits - fbits);
        }
        else
        {
            mant_ = 0;
            mant_(fbits - 1, fbits - in_nbits) = shifted;
        }
    }

    static bool mag_ge(const FloatXUnpacked &a, const FloatXUnpacked &b)
    {
        return (a.exp_ > b.exp_) || (a.exp_ == b.exp_ && a.mant_ >= b.mant_);
    }

    void increment_mantissa_or_exp()
    {
        ap_uint<fbits + 1> m = (ap_uint<1>(0), mant_);
        m++;

        if (m[fbits])
        {
            mant_ = 0;
            ap_int<ebits + 2> e = (ap_int<ebits + 2>)exp_ + 1;
            if (e >= (ap_int<ebits + 2>)all_exp_ones())
            {
                *this = inf(sign_);
            }
            else
            {
                exp_ = e(ebits - 1, 0);
            }
        }
        else
        {
            mant_ = m(fbits - 1, 0);
        }
    }

    template <int W>
    static ap_int<W> max_ap_int()
    {
        ap_int<W> x = 0;
        x[W - 1] = 0;
        for (int i = 0; i < W - 1; ++i)
#pragma HLS UNROLL
            x[i] = 1;
        return x;
    }

    template <int W>
    static ap_int<W> min_ap_int()
    {
        ap_int<W> x = 0;
        x[W - 1] = 1;
        return x;
    }
};

template <int nbits, int ebits>
class FloatX
{
public:
    static constexpr int fbits = nbits - ebits - 1;

    FloatX() : bits_(0) {}
    FloatX(const FloatX &other) : bits_(other.bits_) {}

    FloatX &operator=(const FloatX &other)
    {
        bits_ = other.bits_;
        return *this;
    }

    template <typename T>
    FloatX(T c)
    {
        FloatXUnpacked<nbits, ebits> u(c);
        bits_ = u.encode();
    }

    FloatX(float c)
    {
        ap_uint<32> bits = bitcast_u32(c);
        FloatXUnpacked<32, 8> src;
        src.decode(bits);
        FloatXUnpacked<nbits, ebits> dst(src);
        bits_ = dst.encode();
    }

    FloatX(double c)
    {
        ap_uint<64> bits = bitcast_u64(c);
        FloatXUnpacked<64, 11> src;
        src.decode(bits);
        FloatXUnpacked<nbits, ebits> dst(src);
        bits_ = dst.encode();
    }

    FloatX(const FloatXUnpacked<nbits, ebits> &c) : bits_(c.encode()) {}

    template <int in_nbits>
    operator ap_int<in_nbits>() const
    {
        FloatXUnpacked<nbits, ebits> u;
        u.decode(bits_);
        return u.template to_ap_int<in_nbits>();
    }

    operator int() const
    {
        FloatXUnpacked<nbits, ebits> u;
        u.decode(bits_);
        return int(u);
    }

    operator unsigned int() const
    {
        FloatXUnpacked<nbits, ebits> u;
        u.decode(bits_);
        return (unsigned int)u;
    }

    operator float() const
    {
        FloatXUnpacked<nbits, ebits> src;
        src.decode(bits_);
        FloatXUnpacked<32, 8> dst(src);
        return bitcast_f32(dst.encode());
    }

    operator double() const
    {
        FloatXUnpacked<nbits, ebits> src;
        src.decode(bits_);
        FloatXUnpacked<64, 11> dst(src);
        return bitcast_f64(dst.encode());
    }

    operator FloatXUnpacked<nbits, ebits>() const
    {
        FloatXUnpacked<nbits, ebits> u;
        u.decode(bits_);
        return u;
    }

    FloatXUnpacked<nbits, ebits> unpack() const
    {
        FloatXUnpacked<nbits, ebits> u;
        u.decode(bits_);
        return u;
    }

    FloatXUnpacked<nbits, ebits> operator+(const FloatXUnpacked<nbits, ebits> &rhs) const { return unpack() + rhs; }
    FloatXUnpacked<nbits, ebits> operator-(const FloatXUnpacked<nbits, ebits> &rhs) const { return unpack() - rhs; }
    FloatXUnpacked<nbits, ebits> operator*(const FloatXUnpacked<nbits, ebits> &rhs) const { return unpack() * rhs; }
    FloatXUnpacked<nbits, ebits> operator/(const FloatXUnpacked<nbits, ebits> &rhs) const { return unpack() / rhs; }
    FloatXUnpacked<nbits, ebits> operator-() const { return -unpack(); }

    FloatX &operator+=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        bits_ = (unpack() + rhs).encode();
        return *this;
    }

    FloatX &operator-=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        bits_ = (unpack() - rhs).encode();
        return *this;
    }

    FloatX &operator*=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        bits_ = (unpack() * rhs).encode();
        return *this;
    }

    FloatX &operator/=(const FloatXUnpacked<nbits, ebits> &rhs)
    {
        bits_ = (unpack() / rhs).encode();
        return *this;
    }

    bool operator==(const FloatXUnpacked<nbits, ebits> &rhs) const { return unpack() == rhs; }
    bool operator!=(const FloatXUnpacked<nbits, ebits> &rhs) const { return !(unpack() == rhs); }
    bool operator<(const FloatXUnpacked<nbits, ebits> &rhs) const { return unpack() < rhs; }
    bool operator>(const FloatXUnpacked<nbits, ebits> &rhs) const { return rhs < unpack(); }
    bool operator<=(const FloatXUnpacked<nbits, ebits> &rhs) const { return !(rhs < unpack()); }
    bool operator>=(const FloatXUnpacked<nbits, ebits> &rhs) const { return !(unpack() < rhs); }

    ap_uint<nbits> bits_;
};

// Symmetric operators for FloatX vs FloatX.
template <int nbits, int ebits>
inline FloatXUnpacked<nbits, ebits> operator+(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return a.unpack() + b.unpack();
}

template <int nbits, int ebits>
inline FloatXUnpacked<nbits, ebits> operator-(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return a.unpack() - b.unpack();
}

template <int nbits, int ebits>
inline FloatXUnpacked<nbits, ebits> operator*(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return a.unpack() * b.unpack();
}

template <int nbits, int ebits>
inline FloatXUnpacked<nbits, ebits> operator/(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return a.unpack() / b.unpack();
}

template <int nbits, int ebits>
inline bool operator==(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return a.unpack() == b.unpack();
}

template <int nbits, int ebits>
inline bool operator!=(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return !(a == b);
}

template <int nbits, int ebits>
inline bool operator<(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return a.unpack() < b.unpack();
}

template <int nbits, int ebits>
inline bool operator>(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return b < a;
}

template <int nbits, int ebits>
inline bool operator<=(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return !(b < a);
}

template <int nbits, int ebits>
inline bool operator>=(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    return !(a < b);
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> fabs(const FloatX<nbits, ebits> &a)
{
    FloatX<nbits, ebits> b = a;
    b.bits_[nbits - 1] = 0;
    return b;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> floor(const FloatX<nbits, ebits> &a)
{
    int i = int(a); // truncates toward zero
    FloatX<nbits, ebits> fi(i);
    if (a < fi)
        fi -= FloatX<nbits, ebits>(1);
    return fi;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> floor(const FloatXUnpacked<nbits, ebits> &a)
{
    int i = int(a); // truncates toward zero
    FloatXUnpacked<nbits, ebits> fi(i);
    if (a < fi)
        fi = fi - FloatXUnpacked<nbits, ebits>(1);
    return fi;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> ceil(const FloatX<nbits, ebits> &a)
{
    int i = int(a); // truncates toward zero
    FloatX<nbits, ebits> fi(i);
    if (fi < a)
        fi += FloatX<nbits, ebits>(1);
    return fi;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> ceil(const FloatXUnpacked<nbits, ebits> &a)
{
    int i = int(a); // truncates toward zero
    FloatXUnpacked<nbits, ebits> fi(i);
    if (fi < a)
        fi = fi + FloatXUnpacked<nbits, ebits>(1);
    return fi;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> round(const FloatXUnpacked<nbits, ebits> &a)
{
    // round half away from zero, matching a common embedded approximation.
    if (a < FloatXUnpacked<nbits, ebits>(0))
        return ceil(FloatXUnpacked<nbits, ebits>(a - FloatX<nbits, ebits>(0.5)));
    return floor(FloatXUnpacked<nbits, ebits>(a + FloatX<nbits, ebits>(0.5)));
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> round(const FloatX<nbits, ebits> &a)
{
    // round half away from zero, matching a common embedded approximation.
    if (a < FloatX<nbits, ebits>(0))
        return ceil(FloatX<nbits, ebits>(a - FloatX<nbits, ebits>(0.5)));
    return floor(FloatX<nbits, ebits>(a + FloatX<nbits, ebits>(0.5)));
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> fmod(const FloatX<nbits, ebits> &a, const FloatX<nbits, ebits> &b)
{
    // std::fmod semantics use truncation toward zero, not floor.
    FloatX<nbits, ebits> q = int(a / b);
    return a - q * b;
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> pow2_int(int k)
{
    FloatXUnpacked<nbits, ebits> u;
    const int e = k + FloatXUnpacked<nbits, ebits>::bias;

    if (e <= 0)
        u = FloatXUnpacked<nbits, ebits>::zero(false);
    else if (e >= (int)ap_uint<ebits>(-1))
        u = FloatXUnpacked<nbits, ebits>::inf(false);
    else
    {
        u.sign_ = 0;
        u.zero_ = 0;
        u.inf_ = 0;
        u.nan_ = 0;
        u.exp_ = e;
        u.mant_ = 0;
    }

    return FloatX<nbits, ebits>(u);
}

template <int nbits, int ebits>
inline FloatX<nbits, ebits> exp(const FloatX<nbits, ebits> &x)
{
    // Synthesizable approximation:
    //   exp(x) = 2^k * exp(r), k = round(x / ln(2)), r in about [-ln2/2, ln2/2]
    //   exp(r) approximated with a 5th-order Taylor polynomial.
    // Good enough for tests/basic math; for production, validate error for your nbits/ebits.

    using FX = FloatX<nbits, ebits>;

    const FX zero(0);
    const FX half(0.5);
    const FX one(1);
    const FX inv_ln2(1.4426950408889634);
    const FX ln2(0.6931471805599453);

    if (x < FX(-60.0))
        return FX(0);
    if (x > FX(60.0))
        return FX(FloatXUnpacked<nbits, ebits>::inf(false));

    FX kf = round(x * inv_ln2);
    int k = int(kf);
    FX r = x - FX(k) * ln2;

    FX r2 = r * r;
    FX r3 = r2 * r;
    FX r4 = r2 * r2;
    FX r5 = r4 * r;

    FX er = one + r + r2 * FX(0.5) + r3 * FX(1.0 / 6.0) + r4 * FX(1.0 / 24.0) + r5 * FX(1.0 / 120.0);
    return er * pow2_int<nbits, ebits>(k);
}
