#pragma once

template <typename T>
void test(double in_a, double in_b, double out[10])
{
    T a = (T)in_a;
    T b = (T)in_b;
    double da = (double)a;

    T add = a + b;
    double dadd = (double)add;

    T sub = a - b;
    double dsub = (double)sub;

    T mul = a * b;
    double dmul = (double)mul;

    T div = a / b;
    double ddiv = (double)div;

    T mac = a * b + a;
    double dmac = (double)mac;

    T acc1 = 0;
    T acc2 = 0;
    T acc3 = 0;
    T acc4 = 0;

acc_loop:
    for (int i = 0; i < 1; i++)
    {
        acc1 += ((a * b) + T(i));
        acc2 -= a * b + T(i);
        acc3 *= a * b + T(i);
        acc4 /= a * b + T(i);
    }

    out[0] = da;
    out[1] = dadd;
    out[2] = dsub;
    out[3] = dmul;
    out[4] = ddiv;
    out[5] = dmac;
    out[7] = (double)acc2;
    out[8] = (double)acc3;
    out[9] = (double)acc4;
}