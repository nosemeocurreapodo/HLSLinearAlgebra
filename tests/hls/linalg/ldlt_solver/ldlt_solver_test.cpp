#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>
#include "linalg/linalg.h"

// Declare the HLS top-level function
extern "C" void top(
    double in_0, double in_1, double in_2, double in_3, double in_4, double in_5,
    double in_6, double in_7, double in_8, double in_9, double in_10, double in_11,
    double in_12, double in_13, double in_14, double in_15, double in_16, double in_17,
    double in_18, double in_19, double in_20, double in_21, double in_22, double in_23,
    double in_24, double in_25, double in_26, double in_27, double in_28, double in_29,
    double in_30, double in_31, double in_32, double in_33, double in_34, double in_35,
    double b0, double b1, double b2, double b3, double b4, double b5,
    double &out_0, double &out_1, double &out_2, double &out_3, double &out_4, double &out_5);

static bool check_near(const std::string &name, double expected, double actual, double tol, double &max_err)
{
    double err = std::fabs(expected - actual);
    if (err > max_err) max_err = err;
    if (err > tol) {
        std::cerr << "ERROR: " << name << " expected=" << std::setprecision(12) << expected
                  << " actual=" << actual << " abs_err=" << err << std::endl;
        return true;
    }
    return false;
}

int main()
{
    constexpr int N = 6;
    constexpr double tol = 1e-5;
    double max_err = 0.0;

    std::mt19937 gen(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Build a random SPD matrix
    linalg::Mat<double, N, N> A_linalg;
    Eigen::Matrix<double, N, N> M;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            M(i, j) = dist(gen);
        }
    }
    Eigen::Matrix<double, N, N> A_eig = M.transpose() * M + 0.5 * Eigen::Matrix<double, N, N>::Identity();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_linalg(i, j) = A_eig(i, j);

    // Random RHS
    Eigen::Matrix<double, N, 1> b_eig;
    linalg::Vec<double, N> b_linalg;
    for (int i = 0; i < N; ++i)
    {
        b_eig(i) = dist(gen);
        b_linalg(i) = b_eig(i);
    }

    // Call HLS top (inlined interface uses lots of scalar args)
    double out[6] = {0};
    top(
        A_linalg(0,0), A_linalg(0,1), A_linalg(0,2), A_linalg(0,3), A_linalg(0,4), A_linalg(0,5),
        A_linalg(1,0), A_linalg(1,1), A_linalg(1,2), A_linalg(1,3), A_linalg(1,4), A_linalg(1,5),
        A_linalg(2,0), A_linalg(2,1), A_linalg(2,2), A_linalg(2,3), A_linalg(2,4), A_linalg(2,5),
        A_linalg(3,0), A_linalg(3,1), A_linalg(3,2), A_linalg(3,3), A_linalg(3,4), A_linalg(3,5),
        A_linalg(4,0), A_linalg(4,1), A_linalg(4,2), A_linalg(4,3), A_linalg(4,4), A_linalg(4,5),
        A_linalg(5,0), A_linalg(5,1), A_linalg(5,2), A_linalg(5,3), A_linalg(5,4), A_linalg(5,5),
        b_linalg(0), b_linalg(1), b_linalg(2), b_linalg(3), b_linalg(4), b_linalg(5),
        out[0], out[1], out[2], out[3], out[4], out[5]);

    // Compute reference using linalg LDLT
    linalg::LDLT<double, N> solver;
    solver.compute(A_linalg);
    linalg::Vec<double, N> x_ref = solver.solve(b_linalg);

    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (check_near("ldlt_solution", x_ref(i), out[i], tol, max_err)) ++errors;
    }

    if (errors == 0)
    {
        std::cout << "SUCCESS: LDLT HLS test passed (max abs err = " << max_err << ")" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "FAILURE: LDLT HLS test failed with " << errors << " mismatches (max abs err = " << max_err << ")" << std::endl;
        return 1;
    }
}
