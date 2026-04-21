#include "hls_numerics/FixedX.h"
// #include "hls_numerics/Posit.h"
// not supported on pynq-z1
// #include "ap_float.h"

// Select numeric format via compile-time defines:
// -DFORMAT_POSIT  -> use Posit<32,3>
// -DFORMAT_FLOAT  -> use native float
// (default)        -> use FloatX<32,8>

extern "C" void top(double in_a, double in_b, double &out)
{
#pragma HLS INTERFACE s_axilite port = in_a bundle = control
#pragma HLS INTERFACE s_axilite port = in_b bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    FixedX<32, 8> a = (FixedX<32, 8>)in_a;
    FixedX<32, 8> b = (FixedX<32, 8>)in_b;
    FixedX<33, 9> res = a + b;
    // ap_float<32, 8> a = (ap_float<32, 8>)in_a;
    // ap_float<32, 8> b = (ap_float<32, 8>)in_b;
    // ap_float<32, 8> res = a + b;
    out = (double)res;
}