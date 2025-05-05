#pragma once
#include "ap_int.h"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

class Posit16 {
public:
    ap_uint<16> bits;

    static constexpr int nbits = 16;
    static constexpr int es = 1;

    Posit16() : bits(0) {}
    Posit16(uint16_t b) : bits(b) {}

    // Special cases
    bool isZero() const { return bits == 0x0000; }
    bool isNaR() const { return bits == 0x8000; }

    // Decode posit into sign, regime (k), exponent, fraction, fraction length
    void decode(bool& sign, int& k, int& exponent, uint32_t& fraction, int& frac_len) const {
        if (isZero() || isNaR()) {
            sign = false; k = 0; exponent = 0; fraction = 0; frac_len = 0;
            return;
        }

        uint16_t ui = bits;
        sign = (ui >> 15) & 1;
        if (sign) ui = -ui;

        // Decode regime
        bool reg_bit = (ui >> 14) & 1;
        int reg_len = 1;
        while (reg_len < 15 && ((ui >> (14 - reg_len)) & 1) == reg_bit) {
            reg_len++;
        }
        k = reg_bit ? (reg_len - 1) : (-reg_len);

        // Decode exponent bits (es bits after regime)
        int exp_start = 14 - reg_len;
        exponent = 0;
        for (int i = 0; i < es; ++i) {
            int bit_pos = exp_start - i;
            if (bit_pos >= 0) {
                exponent |= ((ui >> bit_pos) & 1) << (es - 1 - i);
            }
        }

        // Fraction bits after regime and exponent
        int frac_start = exp_start - es;
        frac_len = frac_start + 1;
        fraction = 0;
        for (int i = 0; i < frac_len; ++i) {
            int bit_pos = frac_start - i;
            if (bit_pos >= 0) {
                fraction |= ((ui >> bit_pos) & 1) << (frac_len - 1 - i);
            }
        }
    }

    // Encode posit from sign, regime (k), exponent, fraction bits and length
    static Posit16 encode(bool sign, int k, int exponent, uint32_t fraction, int frac_len) {
        // Saturate regime
        if (k > 14) return Posit16(sign ? 0x8001 : 0x7FFF);
        if (k < -14) return Posit16(sign ? 0x0001 : 0x0000);

        uint16_t result = 0;
        int reg_len = std::min(std::abs(k) + 1, 15);

        // Build regime bits
        if (k >= 0) {
            result = (0x7FFF >> (15 - reg_len)) << (15 - reg_len);
        } else {
            result = (0x4000 >> (reg_len - 1)) & 0x7FFF;
        }

        // Insert exponent bits
        int exp_shift = 15 - reg_len - es;
        if (exp_shift >= 0) {
            uint16_t exp_val = (exponent & ((1 << es) - 1)) << exp_shift;
            result |= exp_val;
        }

        // Insert fraction bits
        int frac_bits_avail = 15 - reg_len - es;
        if (frac_bits_avail > 0 && frac_len > 0) {
            uint16_t frac_mask = (1u << std::min(frac_len, frac_bits_avail)) - 1;
            uint16_t frac_val = (fraction >> std::max(0, frac_len - frac_bits_avail)) & frac_mask;
            result |= frac_val;
        }

        if (sign) result = -result;

        return Posit16(result);
    }

    // Multiply two Posit16 values directly
    Posit16 operator*(const Posit16& rhs) const {
        if (isNaR() || rhs.isNaR()) return Posit16(0x8000);
        if (isZero() || rhs.isZero()) return Posit16(0x0000);

        bool signA, signB;
        int kA, kB, expA, expB;
        uint32_t fracA, fracB;
        int fracLenA, fracLenB;
        decode(signA, kA, expA, fracA, fracLenA);
        rhs.decode(signB, kB, expB, fracB, fracLenB);

        bool res_sign = signA ^ signB;
        int res_k = kA + kB;
        int res_exp = expA + expB;

        // Multiply fractions with implicit 1
        uint64_t mantA = (1ull << fracLenA) | fracA;
        uint64_t mantB = (1ull << fracLenB) | fracB;
        uint64_t product = mantA * mantB;

        int product_len = fracLenA + fracLenB + 1;

        // Normalize product and adjust exponent/regime accordingly
        if (product & (1ull << product_len)) {
            product >>= 1;
            res_exp++;
        }

        // Handle exponent overflow into regime
        if (res_exp >= (1 << es)) {
            res_exp -= (1 << es);
            res_k++;
        }

        // Extract fraction bits with rounding
        int frac_bits = std::max(15 - (std::abs(res_k) + 1) - es, 0);
        int shift = product_len - frac_bits;
        uint32_t res_frac = 0;

        if (shift > 0) {
            uint64_t round_mask = (1ull << (shift - 1)) - 1;
            bool lsb = (product >> (shift - 1)) & 1;
            bool guard = (product & round_mask) != 0;
            res_frac = (product >> shift) + (guard && lsb);
        } else {
            res_frac = product << (-shift);
        }

        // Handle rounding overflow
        if (res_frac >= (1u << frac_bits)) {
            res_frac >>= 1;
            res_exp++;
            if (res_exp >= (1 << es)) {
                res_exp = 0;
                res_k++;
            }
        }

        return encode(res_sign, res_k, res_exp, res_frac, frac_bits);
    }

    // Convert posit to float (for testing)
    float toFloat() const {
        if (isZero()) return 0.0f;
        if (isNaR()) return NAN;

        bool sign;
        int k, exponent;
        uint32_t frac;
        int frac_len;
        decode(sign, k, exponent, frac, frac_len);

        float scale = std::ldexp(1.0f, (k << es) + exponent);
        float fraction = 1.0f;
        for (int i = 0; i < frac_len; ++i) {
            fraction += ((frac >> (frac_len - 1 - i)) & 1) * std::ldexp(1.0f, -i - 1);
        }
        return sign ? -scale * fraction : scale * fraction;
    }
};

