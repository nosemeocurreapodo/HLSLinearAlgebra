#ifndef POSIT_H
#define POSIT_H

#include "ap_int.h"

#define POSIT_SIZE 8
#define POSIT_ES 1

class Posit8 {
public:
    ap_uint<POSIT_SIZE> bits;

    Posit8() : bits(0) {}
    Posit8(ap_uint<POSIT_SIZE> b) : bits(b) {}

    struct PositComponents {
        bool sign;
        int k;           // regime
        int exp;         // exponent
        ap_uint<16> frac; // fraction in fixed-point format (1.xxx...)
    };

    // Decode bit-level posit into components
    PositComponents decode_raw() const {
        PositComponents pc = {0, 0, 0, 0};

        if (bits == 0x00) return pc; // zero
        if (bits == 0x80) return pc; // NaR (not handled yet)

        ap_uint<POSIT_SIZE> tmp = bits;
        pc.sign = tmp[POSIT_SIZE - 1];
        if (pc.sign) tmp = (~tmp) + 1; // two's complement for negative numbers

        int i = POSIT_SIZE - 2;
        bool reg_bit = tmp[i--];
        int k = 0;
        while (i >= 0 && tmp[i] == reg_bit) {
            k++;
            i--;
        }
        if (!reg_bit) k = -k - 1;
        pc.k = k;

        int exp = 0;
        for (int j = 0; j < POSIT_ES && i >= 0; j++, i--) {
            exp = (exp << 1) | tmp[i];
        }
        pc.exp = exp;

        // remaining bits are the fraction
        ap_uint<16> frac = 0x8000; // implicit leading 1 in fixed point
        int shift = 15;
        while (i >= 0 && shift > 0) {
            frac |= (tmp[i--] << --shift);
        }
        pc.frac = frac;

        return pc;
    }

    // Encode posit components into a Posit8
    static Posit8 encode_raw(PositComponents pc) {
        ap_uint<POSIT_SIZE> res = 0;

        // handle zero case
        if (pc.frac == 0) return Posit8(0);

        // compute regime
        int reg_len = (pc.k >= 0) ? pc.k + 2 : -pc.k + 1;
        if (reg_len >= POSIT_SIZE) return Posit8(pc.sign ? 0x80 : 0x7F); // overflow to max/min

        int i = POSIT_SIZE - 2;
        for (int j = 0; j < reg_len - 1 && i >= 0; j++, i--) {
            res[i] = (pc.k >= 0) ? 1 : 0;
        }
        if (i >= 0) res[i--] = (pc.k >= 0) ? 0 : 1;

        // insert exponent
        for (int j = POSIT_ES - 1; j >= 0 && i >= 0; j--, i--) {
            res[i] = (pc.exp >> j) & 1;
        }

        // insert fraction
        int frac_pos = 14;
        while (i >= 0 && frac_pos >= 0) {
            res[i--] = (pc.frac >> frac_pos--) & 1;
        }

        if (pc.sign) res = (~res) + 1;
        return Posit8(res);
    }

    // Real posit multiplication using integer math
    Posit8 operator*(const Posit8& rhs) const {
        PositComponents a = this->decode_raw();
        PositComponents b = rhs.decode_raw();

        PositComponents r;
        r.sign = a.sign ^ b.sign;
        r.k = a.k + b.k;
        r.exp = a.exp + b.exp;

        // Multiply fractions
        ap_uint<32> frac = a.frac * b.frac; // result in [2.0, 4.0)

        if (frac[31]) {
            r.k += 1;
            r.frac = frac >> 17; // shift right to normalize to [1.0, 2.0)
        } else {
            r.frac = frac >> 16;
        }

        return encode_raw(r);
    }

    // Debug: convert to float using approximate method for testing
    float to_float() const {
        PositComponents pc = decode_raw();
        float frac = pc.frac / 32768.0f;
        float scale = (1 << POSIT_ES) * pc.k + pc.exp;
        float result = frac * powf(2.0f, scale);
        return pc.sign ? -result : result;
    }
};

#endif // POSIT_H