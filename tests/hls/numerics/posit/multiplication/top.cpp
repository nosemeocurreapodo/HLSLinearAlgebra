#include "hls_numerics/Posit.h"

extern "C" void top(double in_a, double in_b, double &out)
{
#pragma HLS INTERFACE s_axilite port = in_a bundle = control
#pragma HLS INTERFACE s_axilite port = in_b bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    Posit<32, 3> fx_a(in_a);
    Posit<32, 3> fx_b(in_b);
    Posit<32, 3> res = fx_a * fx_b;
    out = (double)res;
}