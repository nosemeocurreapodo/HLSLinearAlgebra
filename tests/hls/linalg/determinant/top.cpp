#include "linalg/linalg.h"
#include "hls_numerics/FloatX.h"
#include "hls_numerics/Posit.h"

extern "C" void top(double in_0,
                    double in_1,
                    double in_2,
                    double in_3,
                    double in_4,
                    double in_5,
                    double in_6,
                    double in_7,
                    double in_8,
                    double &out)
{
#pragma HLS INTERFACE s_axilite port = in_0 bundle = control
#pragma HLS INTERFACE s_axilite port = in_1 bundle = control
#pragma HLS INTERFACE s_axilite port = in_2 bundle = control
#pragma HLS INTERFACE s_axilite port = in_3 bundle = control
#pragma HLS INTERFACE s_axilite port = in_4 bundle = control
#pragma HLS INTERFACE s_axilite port = in_5 bundle = control
#pragma HLS INTERFACE s_axilite port = in_6 bundle = control
#pragma HLS INTERFACE s_axilite port = in_7 bundle = control
#pragma HLS INTERFACE s_axilite port = in_8 bundle = control

#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    // Select scalar type via compile-time defines:
    // -DFORMAT_POSIT -> use Posit<32,3>
    // -DFORMAT_FLOAT -> use native float
    // (default)       -> use FloatX<32,8>
#if defined(FORMAT_POSIT)
    using Scalar = Posit<32, 3>;
#elif defined(FORMAT_FLOAT)
    using Scalar = float;
#else
    using Scalar = FloatX<32, 8>;
#endif

    linalg::Mat3<Scalar> M;
    M(0, 0) = (Scalar)in_0;
    M(0, 1) = (Scalar)in_1;
    M(0, 2) = (Scalar)in_2;
    M(1, 0) = (Scalar)in_3;
    M(1, 1) = (Scalar)in_4;
    M(1, 2) = (Scalar)in_5;
    M(2, 0) = (Scalar)in_6;
    M(2, 1) = (Scalar)in_7;
    M(2, 2) = (Scalar)in_8;

    out = (double)M.determinant();
}
