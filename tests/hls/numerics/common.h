#pragma once

template <typename T>
void test(double in_a, double in_b, double out[10])
{
    T a = (T)in_a;
    T b = (T)in_b;

    T add = a + b;
    T sub = a - b;
    T mul = a * b;
    T div = a / b;
    T mac = a * b + a;
    T acc1 = 0;
    T acc2 = 0;
    T acc3 = 0;
    T acc4 = 0;
    for (int i = 0; i < 10; i++)
    {
        acc1 += ((a * b) + T(i));
        acc2 -= a * b + T(i);
        acc3 *= a * b + T(i);
        acc4 /= a * b + T(i);
    }

    out[0] = (double)add;
    out[1] = (double)sub;
    out[2] = (double)mul;
    out[3] = (double)div;
    out[4] = (double)mac;
    out[5] = (double)acc1;
    out[6] = (double)acc2;
    out[7] = (double)acc3;
    out[8] = (double)acc4;
    out[9] = 0.0;
}