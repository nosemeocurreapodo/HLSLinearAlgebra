#include "hls_numerics/Posit.h"

extern "C" void top(double in_a, double &out)
{
#pragma HLS INTERFACE s_axilite port = in_a bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    Posit<32, 3> fx_a(in_a);
    out = (double)(fx_a);
}