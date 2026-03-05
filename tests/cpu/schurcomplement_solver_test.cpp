// test_schur_solver.cpp
//
// GTest for SchurSolverDepthPose (CHOLMOD + LAPACK Schur complement)
// Compares against dense reference solve on the full system.
//
// Build requirements:
// - gtest
// - SuiteSparse / CHOLMOD
// - LAPACKe (or LAPACK + LAPACKE)
//
// NOTE: This test assumes:
// - SparseMatCSC stores CSC (lower triangle), finalized() sorts rows in each column
// - SchurSolverDepthPose from your previous header
//
// If your pose DOF differs, just change n_pose.

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <lapacke.h>

#include "linalg/linalg.h"
#include "linalg/linalgx.h"
#include "linalg/ldlt_solverx.h"
#include "linalg/sparsemat_csc.h"
#include "linalg/schurcomplement_solver.h" // <-- the class I provided: SchurSolverDepthPose + DenseCM

using namespace linalg;

static double max_abs_diff(const Vecx<double> &a, const Vecx<double> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("size mismatch");
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a(i) - b(i)));
    return m;
}

TEST(SchurComplementSolver, MatchesDenseReference)
{
    // Small but non-trivial sizes
    const int n_mesh = 15;
    const int n_pose = 8;
    const int n = n_mesh + n_pose;

    // Build a guaranteed SPD system by constructing H = J^T J + lambda I
    const int m = 80;           // number of residuals
    const double lambda = 1e-2; // damping to make SPD robust

    std::mt19937 rng(12345);
    std::normal_distribution<double> nd(0.0, 1.0);

    // J: m x n (row-major for convenience here)
    Matx<double> J(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            J(i, j) = nd(rng);

    // residual r: m
    Vecx<double> r(m);
    for (int i = 0; i < m; ++i)
        r(i) = nd(rng);

    // Compute H = J^T J + lambda I (dense, col-major)
    Matx<double> H(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < m; ++k)
                sum += J(k, i) * J(k, j);
            H(i, j) = sum;
            H(j, i) = sum;
        }
        H(i, i) += lambda;
    }

    // Compute g = J^T r  (so system is H x = g)
    Vecx<double> g(n);
    for (int j = 0; j < n; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k)
            sum += J(k, j) * r(k);
        g(j) = sum;
    }

    // Extract blocks:
    // Hvv (mesh x mesh), Hvp (mesh x pose), Hpp (pose x pose)
    // gv, gp
    Vecx<double> gv(n_mesh), gp(n_pose);
    for (int i = 0; i < n_mesh; ++i)
        gv(i) = g(i);
    for (int i = 0; i < n_pose; ++i)
        gp(i) = g(n_mesh + i);

    Matx<double> Hvp(n_mesh, n_pose);
    Matx<double> Hpp(n_pose, n_pose);

    for (int pc = 0; pc < n_pose; ++pc)
    {
        for (int vr = 0; vr < n_mesh; ++vr)
            Hvp(vr, pc) = H(vr, n_mesh + pc);
    }

    for (int c = 0; c < n_pose; ++c)
    {
        for (int r2 = 0; r2 < n_pose; ++r2)
            Hpp(r2, c) = H(n_mesh + r2, n_mesh + c);
    }

    // Build Hvv as SparseMatCSC (lower triangle). For this test, use a FULL lower triangle pattern.
    // This avoids any "pattern mismatch" surprises.
    SparseMatCSC Hvv_pattern(n_mesh, /*store_lower_only=*/true);
    for (int i = 0; i < n_mesh; ++i)
        for (int j = 0; j <= i; ++j)
            Hvv_pattern.add(i, j, 1.0); // placeholder
    Hvv_pattern.finalize();

    // Build numeric Hvv values with the EXACT SAME pattern construction order
    // (safe because finalize sorts, and pattern is full lower triangle).
    SparseMatCSC Hvv_values(n_mesh, /*store_lower_only=*/true);
    for (int i = 0; i < n_mesh; ++i)
        for (int j = 0; j <= i; ++j)
            Hvv_values.add(i, j, H(i, j));
    Hvv_values.finalize();

    // --- Solve with Schur complement solver ---
    SchurSolver schur;
    schur.set_pose_dim(n_pose);

    // Optional: a custom permutation for mesh-only factorization.
    // For this test just let CHOLMOD choose.
    schur.analyze_mesh_pattern(Hvv_pattern, nullptr);
    schur.factorize_mesh_numeric(Hvv_values);

    Vecx<double> dv, dp;
    schur.solve(Hvp, Hpp, gv, gp, dv, dp, /*lambda_p=*/0.0);

    ASSERT_EQ((int)dv.size(), n_mesh);
    ASSERT_EQ((int)dp.size(), n_pose);

    // Combine into x_schur
    Vecx<double> x_schur(n);
    for (int i = 0; i < n_mesh; ++i)
        x_schur(i) = dv(i);
    for (int i = 0; i < n_pose; ++i)
        x_schur(n_mesh + i) = dp(i);

    // --- Dense reference solve on full system H x = g ---
    Matx<double> H_ref = H; // dposv overwrites A
    Vecx<double> x_ref = g; // dposv overwrites RHS with solution

    int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L',
                             n, 1,
                             H_ref.data(), n,
                             x_ref.data(), n);
    ASSERT_EQ(info, 0);

    // Compare solutions
    const double err = max_abs_diff(x_schur, x_ref);
    EXPECT_LT(err, 1e-8);

    // Also check residual ||H x - g||_inf is small
    Vecx<double> Hx(n);
    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += H(i, j) * x_schur(j);
        Hx(i) = sum;
    }

    double res_inf = 0.0;
    for (int i = 0; i < n; ++i)
        res_inf = std::max(res_inf, std::abs(Hx(i) - g(i)));

    EXPECT_LT(res_inf, 1e-8);

    // --- Dense reference solve on full system H x = g ---
    LDLTx<double> ldlt_solver(n);

    ldlt_solver.compute(H);
    Vecx<double> x_ref_2 = ldlt_solver.solve(g);

    // Compare solutions
    const double err_2 = max_abs_diff(x_schur, x_ref_2);
    EXPECT_LT(err_2, 1e-8);
}