#pragma once
#include "ap_int.h"
#include <cmath>
#include <cstdint>
#include <limits>

class Posit8 {
public:
    ap_uint<8> bits;

    static constexpr int nbits = 8;
    static constexpr int es = 0;

    Posit8() : bits(0) {}
    Posit8(uint8_t b) : bits(b) {}

    // --- Decode Function ---
    void decode(bool& sign, int& k, uint32_t& fraction, int& frac_len) const {
        if (bits == 0x00 || bits == 0x80) return; // Handle 0/NaR early

        uint8_t ui = bits;
        sign = ui >> 7;
        if (sign) ui = -ui;

        // Regime calculation
        bool regime_bit = (ui >> 6) & 1;
        int regime_count = 0;
        int bit_pos = 6;
        while (bit_pos >= 0 && ((ui >> bit_pos) & 1) == regime_bit) {
            regime_count++;
            bit_pos--;
        }
        k = regime_bit ? regime_count - 1 : -regime_count;

        // Fraction extraction
        frac_len = nbits - 1 - regime_count - 1; // Exclude sign and regime terminator
        frac_len = std::max(frac_len, 0);
        fraction = 0;
        for (int i = 0; i < frac_len; ++i) {
            if (bit_pos - i - 1 >= 0) {
                fraction |= ((ui >> (bit_pos - i - 1)) & 1) << (frac_len - 1 - i);
            }
        }
    }

    // --- Encode Function ---
    static Posit8 encode(bool sign, int k, uint32_t fraction, int frac_len) {
        if (k > 6) return Posit8(sign ? 0x81 : 0x7F); // Saturate to max
        if (k < -6) return Posit8(sign ? 0x01 : 0x00); // Min pos/zero

        // Build regime
        uint8_t regime = 0;
        int regime_bits = std::abs(k) + 1;
        regime_bits = std::min(regime_bits, 7);
        if (k >= 0) {
            regime = (0x7F >> (7 - regime_bits)) << (7 - regime_bits);
        } else {
            regime = 0x40 >> (regime_bits - 1);
        }

        // Build fraction
        uint8_t frac_part = 0;
        int available_bits = 7 - regime_bits;
        if (available_bits > 0) {
            frac_part = (fraction >> (frac_len - available_bits)) & ((1 << available_bits) - 1);
        }

        // Combine
        uint8_t result = regime | frac_part;
        if (sign) result = -result;

        return Posit8(result);
    }

    // --- Direct Multiplication ---
    Posit8 operator*(const Posit8& rhs) const {
        if (bits == 0x80 || rhs.bits == 0x80) return Posit8(0x80); // NaR
        if (bits == 0x00 || rhs.bits == 0x00) return Posit8(0x00); // Zero

        bool sign1, sign2;
        int k1, k2;
        uint32_t frac1, frac2;
        int frac_len1, frac_len2;
        decode(sign1, k1, frac1, frac_len1);
        decode(sign2, k2, frac2, frac_len2);

        // Calculate components
        bool res_sign = sign1 ^ sign2;
        int res_k = k1 + k2;
        uint64_t product = (1ull << frac_len1 | frac1) * (1ull << frac_len2 | frac2);

        // Normalize product (assumes product is in [1, 4))
        int shift = (product >> (frac_len1 + frac_len2 + 1)) ? 1 : 0;
        res_k += shift;
        product >>= shift;

        // Truncate fraction to available bits
        int total_frac_bits = std::max(7 - (std::abs(res_k) + 1), 0);
        uint32_t res_frac = (product >> (frac_len1 + frac_len2 + 1 - total_frac_bits)) 
                          & ((1 << total_frac_bits) - 1);

        return encode(res_sign, res_k, res_frac, total_frac_bits);
    }

    // --- Float Conversion for Testing ---
    float toFloat() const {
        if (bits == 0x00) return 0.0f;
        if (bits == 0x80) return NAN;

        bool sign;
        int k;
        uint32_t frac;
        int frac_len;
        decode(sign, k, frac, frac_len);

        float value = std::ldexp(1.0f, k);
        float fraction = 1.0f;
        for (int i = 0; i < frac_len; ++i) {
            fraction += ((frac >> (frac_len - 1 - i)) & 1) * std::ldexp(1.0f, -i - 1);
        }

        return sign ? -value * fraction : value * fraction;
    }
};
