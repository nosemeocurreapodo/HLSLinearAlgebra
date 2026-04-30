#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>
// #include <Eigen/Core>
#include "linalg/linalg.h"
#include "linalg/ldlt_solver.h"

// Declare the HLS top-level function
extern "C" void top(
    double *mat_in,
    double *vec_in_out,
    int rows,
    int cols);

static bool check_near(const std::string &name, double expected, double actual, double tol, double &max_err)
{
    double err = std::fabs(expected - actual);
    if (err > max_err)
        max_err = err;
    if (err > tol)
    {
        std::cerr << "ERROR: " << name << " expected=" << std::setprecision(12) << expected
                  << " actual=" << actual << " abs_err=" << err << std::endl;
        return true;
    }
    return false;
}

int main()
{
    constexpr int N = 16;
    constexpr double tol = 1e-5;
    double max_err = 0.0;

    std::mt19937 gen(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Build a random SPD matrix
    // Eigen::Matrix<double, N, N> M;
    linalg::Mat<double, N, N> M;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            M(i, j) = dist(gen);
        }
    }

    linalg::Mat<double, N, N> A_linalg = M.transpose() * M + 0.5 * linalg::Mat<double, N, N>::Identity();
    // linalg::Mat<double, N, N> A_linalg;

    // for (int i = 0; i < N; ++i)
    //     for (int j = 0; j < N; ++j)
    //         A_linalg(i, j) = A_eig(i, j);

    // Random RHS
    // Eigen::Matrix<double, N, 1> b_eig;
    linalg::Vec<double, N> b_linalg;
    for (int i = 0; i < N; ++i)
    {
        b_linalg(i) = dist(gen);
        // b_linalg(i) = b_eig(i);
    }

    // Compute reference using linalg LDLT
    linalg::LDLT<double, N> solver;
    solver.compute(A_linalg);
    linalg::Vec<double, N> x_ref = solver.solve(b_linalg);

    // Call HLS top (inlined interface uses lots of scalar args)
    top(A_linalg.data(), b_linalg.data(), N, N);

    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (check_near("ldlt_solution", x_ref(i), b_linalg(i), tol, max_err))
            ++errors;
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
