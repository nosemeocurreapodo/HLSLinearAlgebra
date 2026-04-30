#include "hls_numerics/Posit.h"

// Define the custom float type for synthesis
// Using 32 bits with 8 exponent bits, similar to standard float
constexpr int N_BITS = 32;
constexpr int E_BITS = 3;
using Ps = Posit<N_BITS, E_BITS>;

/**
 * @brief Top-level function for HLS synthesis to test FloatX operations.
 *
 * This function takes two double-precision inputs, converts them to FloatX,
 * performs basic arithmetic operations (add, subtract, multiply, divide),
 * and returns the results as double-precision outputs.
 *
 * @param in_a      First input value.
 * @param in_b      Second input value.
 * @param out_add   Result of a + b.
 * @param out_sub   Result of a - b.
 * @param out_mul   Result of a * b.
 * @param out_div   Result of a / b.
 */
void posit_top(double in_a, double in_b, double& out_add, double& out_sub, double& out_mul, double& out_div) {
    // HLS pragmas to define the interface for the hardware kernel
    #pragma HLS INTERFACE s_axilite port=in_a bundle=control
    #pragma HLS INTERFACE s_axilite port=in_b bundle=control
    #pragma HLS INTERFACE s_axilite port=out_add bundle=control
    #pragma HLS INTERFACE s_axilite port=out_sub bundle=control
    #pragma HLS INTERFACE s_axilite port=out_mul bundle=control
    #pragma HLS INTERFACE s_axilite port=out_div bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Convert inputs from double to FloatX
    Ps fx_a(in_a);
    Ps fx_b(in_b);

    // Perform arithmetic operations and convert back to double for output
    out_add = (double)(fx_a + fx_b);
    out_sub = (double)(fx_a - fx_b);
    out_mul = (double)(fx_a * fx_b);
    out_div = (double)(fx_a / fx_b);
}