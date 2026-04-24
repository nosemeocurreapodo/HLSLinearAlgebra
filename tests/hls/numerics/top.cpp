#include "hls_numerics/FixedX.h"
#include "hls_numerics/FloatX.h"
#include "hls_numerics/Posit.h"
#include "common.h"
// not supported on pynq-z1
// #include "ap_float.h"

// Select numeric format via compile-time defines:
// -DFORMAT_POSIT  -> use Posit<32,3>
// -DFORMAT_FLOAT  -> use native float
// (default)        -> use FloatX<32,8>

#define FORMAT_FLOATX

extern "C" void top(double in_a, double in_b, double out[10])
{
#pragma HLS INTERFACE s_axilite port = in_a bundle = control
#pragma HLS INTERFACE s_axilite port = in_b bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // #pragma HLS PIPELINE

#if defined(FORMAT_FIXED)
    using T = ap_fixed<32, 16>;
#elif defined(FORMAT_FIXEDX)
    using T = FixedX<32, 16>;
#elif defined(FORMAT_FLOAT)
    using T = float;
#elif defined(FORMAT_FLOATX)
    using T = FloatX<32, 8>;
#elif defined(FORMAT_POSIT)
    using T = Posit<32, 3>;
#else
    using T = int;
#endif

    test<T>(in_a, in_b, out);
}