#include "linalg/linalgx.h"
#include "linalg/ldlt_solver_view_inplace.h"
// #include "hls_numerics/FloatX.h"
// #include "hls_numerics/Posit.h"

using T = double;

// Top-level HLS function for LDLT solver (fixed N=6)
extern "C" void top(
    double *mat_in,
    double *vec_in_out,
    int rows,
    int cols)
{
#pragma HLS INTERFACE m_axi port = mat_in bundle = gmem1
#pragma HLS INTERFACE m_axi port = vec_in_out bundle = gmem2

#pragma HLS INTERFACE s_axilite port = rows bundle = control
#pragma HLS INTERFACE s_axilite port = cols bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE

    linalg::MatView<T> A(mat_in, rows, cols);
    linalg::VecView<T> b(vec_in_out, rows);

    linalg::ldlt_factorize<T, 16>(A);
    linalg::ldlt_solve<T, 16>(A, b);
}
