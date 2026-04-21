#include "linalg/linalg.h"
#include "linalg/ldlt_solver.h"
#include "hls_numerics/FixedX.h"
#include "hls_numerics/FloatX.h"
// #include "hls_numerics/Posit.h"

using T = float;
//using T = FloatX<32, 8>;
// using T = FixedX<32, 8>;

// Top-level HLS function for LDLT solver (fixed N=6)
extern "C" void top(
    double in_0, double in_1, double in_2, double in_3, double in_4, double in_5,
    double in_6, double in_7, double in_8, double in_9, double in_10, double in_11,
    double in_12, double in_13, double in_14, double in_15, double in_16, double in_17,
    double in_18, double in_19, double in_20, double in_21, double in_22, double in_23,
    double in_24, double in_25, double in_26, double in_27, double in_28, double in_29,
    double in_30, double in_31, double in_32, double in_33, double in_34, double in_35,
    double b0, double b1, double b2, double b3, double b4, double b5,
    double &out_0, double &out_1, double &out_2, double &out_3, double &out_4, double &out_5)
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
#pragma HLS INTERFACE s_axilite port = in_9 bundle = control
#pragma HLS INTERFACE s_axilite port = in_10 bundle = control
#pragma HLS INTERFACE s_axilite port = in_11 bundle = control
#pragma HLS INTERFACE s_axilite port = in_12 bundle = control
#pragma HLS INTERFACE s_axilite port = in_13 bundle = control
#pragma HLS INTERFACE s_axilite port = in_14 bundle = control
#pragma HLS INTERFACE s_axilite port = in_15 bundle = control
#pragma HLS INTERFACE s_axilite port = in_16 bundle = control
#pragma HLS INTERFACE s_axilite port = in_17 bundle = control
#pragma HLS INTERFACE s_axilite port = in_18 bundle = control
#pragma HLS INTERFACE s_axilite port = in_19 bundle = control
#pragma HLS INTERFACE s_axilite port = in_20 bundle = control
#pragma HLS INTERFACE s_axilite port = in_21 bundle = control
#pragma HLS INTERFACE s_axilite port = in_22 bundle = control
#pragma HLS INTERFACE s_axilite port = in_23 bundle = control
#pragma HLS INTERFACE s_axilite port = in_24 bundle = control
#pragma HLS INTERFACE s_axilite port = in_25 bundle = control
#pragma HLS INTERFACE s_axilite port = in_26 bundle = control
#pragma HLS INTERFACE s_axilite port = in_27 bundle = control
#pragma HLS INTERFACE s_axilite port = in_28 bundle = control
#pragma HLS INTERFACE s_axilite port = in_29 bundle = control
#pragma HLS INTERFACE s_axilite port = in_30 bundle = control
#pragma HLS INTERFACE s_axilite port = in_31 bundle = control
#pragma HLS INTERFACE s_axilite port = in_32 bundle = control
#pragma HLS INTERFACE s_axilite port = in_33 bundle = control
#pragma HLS INTERFACE s_axilite port = in_34 bundle = control
#pragma HLS INTERFACE s_axilite port = in_35 bundle = control

#pragma HLS INTERFACE s_axilite port = b0 bundle = control
#pragma HLS INTERFACE s_axilite port = b1 bundle = control
#pragma HLS INTERFACE s_axilite port = b2 bundle = control
#pragma HLS INTERFACE s_axilite port = b3 bundle = control
#pragma HLS INTERFACE s_axilite port = b4 bundle = control
#pragma HLS INTERFACE s_axilite port = b5 bundle = control

#pragma HLS INTERFACE s_axilite port = out_0 bundle = control
#pragma HLS INTERFACE s_axilite port = out_1 bundle = control
#pragma HLS INTERFACE s_axilite port = out_2 bundle = control
#pragma HLS INTERFACE s_axilite port = out_3 bundle = control
#pragma HLS INTERFACE s_axilite port = out_4 bundle = control
#pragma HLS INTERFACE s_axilite port = out_5 bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    linalg::Mat6<T> A;

    // Fill matrix (row-major order)
    A(0, 0) = in_0;
    A(0, 1) = in_1;
    A(0, 2) = in_2;
    A(0, 3) = in_3;
    A(0, 4) = in_4;
    A(0, 5) = in_5;
    A(1, 0) = in_6;
    A(1, 1) = in_7;
    A(1, 2) = in_8;
    A(1, 3) = in_9;
    A(1, 4) = in_10;
    A(1, 5) = in_11;
    A(2, 0) = in_12;
    A(2, 1) = in_13;
    A(2, 2) = in_14;
    A(2, 3) = in_15;
    A(2, 4) = in_16;
    A(2, 5) = in_17;
    A(3, 0) = in_18;
    A(3, 1) = in_19;
    A(3, 2) = in_20;
    A(3, 3) = in_21;
    A(3, 4) = in_22;
    A(3, 5) = in_23;
    A(4, 0) = in_24;
    A(4, 1) = in_25;
    A(4, 2) = in_26;
    A(4, 3) = in_27;
    A(4, 4) = in_28;
    A(4, 5) = in_29;
    A(5, 0) = in_30;
    A(5, 1) = in_31;
    A(5, 2) = in_32;
    A(5, 3) = in_33;
    A(5, 4) = in_34;
    A(5, 5) = in_35;

    linalg::Vec6<T> b;
    b(0) = (T)b0;
    b(1) = (T)b1;
    b(2) = (T)b2;
    b(3) = (T)b3;
    b(4) = (T)b4;
    b(5) = (T)b5;

    // Compute LDLT and solve
    linalg::LDLT<T, 6> solver;
    solver.compute(A);
    linalg::Vec6<T> x = solver.solve(b);

    out_0 = (double)x(0);
    out_1 = (double)x(1);
    out_2 = (double)x(2);
    out_3 = (double)x(3);
    out_4 = (double)x(4);
    out_5 = (double)x(5);
}
