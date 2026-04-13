#include "hls_numerics/FloatX.h"

extern "C" void top(double in_a, double in_b, double &out)
{
#pragma HLS INTERFACE s_axilite port = in_a bundle = control
#pragma HLS INTERFACE s_axilite port = in_b bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    FloatX<32, 8> fx_a(in_a);
    FloatX<32, 8> fx_b(in_b);
    FloatX<32, 8> res = fx_a / fx_b;
    out = (double)(res);
}