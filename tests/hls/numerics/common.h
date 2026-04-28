#pragma once

template <typename T>
void test(double in_a, double in_b, double out[64])
{
#pragma HLS INLINE off

    T a = T(in_a);
    T b = T(in_b);

    // Avoid divide-by-zero in the test itself.
    // This assumes T supports comparison with zero.
    if (b == T(0))
        b = T(1);

    int idx = 0;

    // ---------------------------------------------------------------------
    // Basic conversions
    // ---------------------------------------------------------------------

    int i = int(a);
    unsigned int ui = 0; //(unsigned int)(a);
    float f = float(a);
    double d = double(a);

    out[idx++] = double(i);
    out[idx++] = double(ui);
    out[idx++] = double(f);
    out[idx++] = d;

    // ---------------------------------------------------------------------
    // Construction from integer and floating-point literals
    // ---------------------------------------------------------------------

    T z = T(int(in_a));
    T one = 0; // T((unsigned int)(in_a));
    T two = T(float(in_a));
    T half = T(double(in_a));

    out[idx++] = double(z);
    out[idx++] = double(one);
    out[idx++] = double(two);
    out[idx++] = double(half);

    // ---------------------------------------------------------------------
    // Unary operators
    // ---------------------------------------------------------------------

    T pos = 0; //+a;
    T neg = -a;

    out[idx++] = double(pos);
    out[idx++] = double(neg);

    // ---------------------------------------------------------------------
    // Basic binary arithmetic
    // ---------------------------------------------------------------------

    T add = a + b;
    T sub = a - b;
    T mul = a * b;
    T div = a / b;

    out[idx++] = double(add);
    out[idx++] = double(sub);
    out[idx++] = double(mul);
    out[idx++] = double(div);

    // ---------------------------------------------------------------------
    // Arithmetic with literals
    // ---------------------------------------------------------------------

    T add_lit = a + T(1);
    T sub_lit = a - T(1);
    T mul_lit = a * T(2);
    T div_lit = a / T(2);

    out[idx++] = double(add_lit);
    out[idx++] = double(sub_lit);
    out[idx++] = double(mul_lit);
    out[idx++] = double(div_lit);

    // ---------------------------------------------------------------------
    // Compound assignment operators
    // ---------------------------------------------------------------------

    T a_add = a;
    a_add += b;

    T a_sub = a;
    a_sub -= b;

    T a_mul = a;
    a_mul *= b;

    T a_div = a;
    a_div /= b;

    out[idx++] = double(a_add);
    out[idx++] = double(a_sub);
    out[idx++] = double(a_mul);
    out[idx++] = double(a_div);

    // ---------------------------------------------------------------------
    // Mixed expression trees
    // ---------------------------------------------------------------------

    T mac = a * b + a;
    T msub = a * b - a;
    T complex1 = (a + b) * (a - b);
    T complex2 = 0; //(a * b + T(1)) / (b + T(1));

    out[idx++] = double(mac);
    out[idx++] = double(msub);
    out[idx++] = double(complex1);
    out[idx++] = double(complex2);

    // ---------------------------------------------------------------------
    // Comparisons
    // Store as 0.0 or 1.0
    // ---------------------------------------------------------------------

    out[idx++] = double(a == b);
    out[idx++] = double(a != b);
    out[idx++] = double(a < b);
    out[idx++] = double(a <= b);
    out[idx++] = double(a > b);
    out[idx++] = double(a >= b);

    // ---------------------------------------------------------------------
    // Accumulation patterns
    // ---------------------------------------------------------------------

    /*
    T acc_add = T(0);
    T acc_sub = T(0);
    T acc_mul = T(1);
    T acc_div = T(1);

acc_loop:
    for (int j = 0; j < 4; j++)
    {
#pragma HLS PIPELINE off
        T x = a * b + T(j + 1);

        acc_add += x;
        acc_sub -= x;
        acc_mul *= T(1) + x / T(16);
        acc_div /= T(1) + x / T(16);
    }

    out[idx++] = double(acc_add);
    out[idx++] = double(acc_sub);
    out[idx++] = double(acc_mul);
    out[idx++] = double(acc_div);
    */

    // ---------------------------------------------------------------------
    // Self-assignment style expressions
    // ---------------------------------------------------------------------

    /*
    T s = a;
    s = s + b;
    s = s * T(2);
    s = s - a;
    s = s / T(2);

    out[idx++] = double(s);
    */

    // ---------------------------------------------------------------------
    // More parenthesized expressions, useful for checking overload behavior
    // ---------------------------------------------------------------------

    /*
    T expr1 = a + b * T(2);
    T expr2 = (a + b) * T(2);
    T expr3 = a / (b + T(1));
    T expr4 = (a - b) / (a + b + T(1));

    out[idx++] = double(expr1);
    out[idx++] = double(expr2);
    out[idx++] = double(expr3);
    out[idx++] = double(expr4);
    */

    // ---------------------------------------------------------------------
    // Fill remaining outputs with a recognizable value
    // ---------------------------------------------------------------------

fill_loop:
    for (; idx < 64; idx++)
    {
#pragma HLS UNROLL
        out[idx] = -9999.0;
    }
}