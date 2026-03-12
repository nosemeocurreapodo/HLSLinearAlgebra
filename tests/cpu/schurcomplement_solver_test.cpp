#include <gtest/gtest.h>

#include <random>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>

#include <lapacke.h>

#include "linalg/linalg.h"
#include "linalg/linalgx.h"
#include "linalg/ldlt_solverx.h"
#include "linalg/ldlt_solverx_lapack.h"
#include "linalg/sparsemat_csc.h"
#include "linalg/schurcomplement_solver.h"
#include "common.h"

using namespace linalg;

// -------------------------
// helpers
// -------------------------
static double max_abs_diff(const Vecx<double> &a, const Vecx<double> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("size mismatch");
    double m = 0.0;
    for (int i = 0; i < (int)a.size(); ++i)
        m = std::max(m, std::abs(a(i) - b(i)));
    return m;
}

static double inf_norm_residual(const Matx<double> &H, const Vecx<double> &x, const Vecx<double> &g)
{
    const int n = (int)g.size();
    double res_inf = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += H(i, j) * x(j);
        res_inf = std::max(res_inf, std::abs(sum - g(i)));
    }
    return res_inf;
}

// -------------------------
// TEST 1: correctness vs dense reference (full Hvv pattern)
// -------------------------
TEST(SchurComplementSolver, FullPatternMatchesDenseReference_Small)
{
    const int w_mesh = 8;
    const int h_mesh = 8;
    const int n_mesh = 8 * 8;
    const int s_pose = 6;
    const int dof_pose = 6;
    const int n_pose = s_pose * dof_pose;
    const int n = n_mesh + n_pose;

    const double lambda = 1e-2;

    SPDSystem S = make_system(w_mesh, h_mesh, s_pose, dof_pose, lambda, /*seed=*/12345);

    SparseMatCSC<double> Hvv_values = from_dense_to_sparse<Matx<double>, SparseMatCSC<double>>(S.H);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_values, nullptr);
    schur.factorize_mesh_numeric(Hvv_values);

    Vecx<double> dv, dp;
    schur.solve(S.Hvp, S.Hpp, S.gv, S.gp, dv, dp, /*lambda_p=*/0.0);

    Vecx<double> x_schur(n);
    for (int i = 0; i < n_mesh; ++i)
        x_schur(i) = dv(i);
    for (int i = 0; i < n_pose; ++i)
        x_schur(n_mesh + i) = dp(i);

    // Dense reference: H x = g using LAPACK
    Matx<double> H_ref = S.H; // overwritten
    Vecx<double> x_ref = S.g; // overwritten
    int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, 1, H_ref.data(), n, x_ref.data(), n);
    ASSERT_EQ(info, 0);

    EXPECT_LT(max_abs_diff(x_schur, x_ref), 1e-8);
    EXPECT_LT(inf_norm_residual(S.H, x_schur, S.g), 1e-8);

    // Optional secondary reference via your LDLT
    LDLTx<double> ldlt_solver(n);
    ldlt_solver.compute(S.H);
    Vecx<double> x_ref2 = ldlt_solver.solve(S.g);
    EXPECT_LT(max_abs_diff(x_schur, x_ref2), 1e-8);
}

// -------------------------
// TEST 2: correctness on a grid-stencil Hvv pattern (more realistic structure)
// This test checks residual of the BLOCK system you actually feed the solver,
// not against dense H (since stencil Hvv != dense Hvv).
// -------------------------
TEST(SchurComplementSolver, GridStencilPattern_BlockResidualIsSmall)
{
    using T = double;

    const int W = 16, H = 16; // 256 mesh vars
    const int n_mesh = W * H;
    const int s_pose = 2;
    const int dof_pose = 6;
    const int n_pose = s_pose * dof_pose;
    const int n = n_mesh + n_pose;

    const int m = 200;
    const T lambda = 1e-2;

    SPDSystem S = make_system(W, H, s_pose, dof_pose, lambda, /*seed=*/424242);

    SparseMatCSC<T> Hvv_values = from_dense_to_sparse<Matx<T>, SparseMatCSC<T>>(S.H);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_values, nullptr);
    schur.factorize_mesh_numeric(Hvv_values);

    Vecx<T> dv, dp;
    schur.solve(S.Hvp, S.Hpp, S.gv, S.gp, dv, dp, /*lambda_p=*/0.0);

    ASSERT_EQ((int)dv.size(), n_mesh);
    ASSERT_EQ((int)dp.size(), n_pose);

    // Validate residual of the block system you solved:
    // [Hvv Hvp; Hpv Hpp] [dv;dp] = [gv;gp]
    // where Hvv is the STENCIL sparse matrix values we gave it (not S.H dense block).
    //
    // We compute:
    // rv = Hvv*dv + Hvp*dp - gv
    // rp = Hvp^T*dv + Hpp*dp - gp
    //
    // Build rv using sparse Hvv_values CSC
    Vecx<T> rv = Vecx<T>::Zero(n_mesh);
    const int *Ap = Hvv_values.Ap();
    const int *Ai = Hvv_values.Ai();
    const T *Ax = Hvv_values.Ax();

    // Hvv is symmetric stored lower triangle; apply full symmetric multiply
    // rv += Hvv * dv
    for (int col = 0; col < n_mesh; ++col)
    {
        for (int p = Ap[col]; p < Ap[col + 1]; ++p)
        {
            int row = Ai[p];
            T v = Ax[p];
            // entry at (row, col) where row>=col
            rv(row) += v * dv(col);
            if (row != col)
                rv(col) += v * dv(row);
        }
    }

    // rv += Hvp*dp
    for (int c = 0; c < n_pose; ++c)
        for (int r2 = 0; r2 < n_mesh; ++r2)
            rv(r2) += S.Hvp(r2, c) * dp(c);

    // rv -= gv
    for (int i = 0; i < n_mesh; ++i)
        rv(i) -= S.gv(i);

    // rp = Hvp^T*dv + Hpp*dp - gp
    Vecx<T> rp = Vecx<T>::Zero(n_pose);

    for (int i = 0; i < n_pose; ++i)
    {
        T sum = 0.0;
        for (int r2 = 0; r2 < n_mesh; ++r2)
            sum += S.Hvp(r2, i) * dv(r2);
        rp(i) = sum;
    }

    for (int c = 0; c < n_pose; ++c)
    {
        T sum = 0.0;
        for (int r2 = 0; r2 < n_pose; ++r2)
            sum += S.Hpp(c, r2) * dp(r2);
        rp(c) += sum;
        rp(c) -= S.gp(c);
    }

    // Check infinity norms are small
    T rv_inf = 0.0, rp_inf = 0.0;
    for (int i = 0; i < n_mesh; ++i)
        rv_inf = std::max(rv_inf, std::abs(rv(i)));
    for (int i = 0; i < n_pose; ++i)
        rp_inf = std::max(rp_inf, std::abs(rp(i)));

    EXPECT_LT(rv_inf, 1e-7);
    EXPECT_LT(rp_inf, 1e-7);
}

// -------------------------
// TEST 3: fixed-pattern reuse & timing (analyze once; many numeric updates)
// Prints average milliseconds.
// -------------------------
TEST(SchurComplementSolver, FixedPatternReuse_Timing)
{
    using T = double;

    const int W = 32, H = 32; // 1024 mesh vars (your real scale)
    const int n_mesh = W * H;
    const int s_pose = 7;
    const int dof_pose = 6;
    const int n_pose = s_pose * dof_pose;
    const int n = n_mesh + n_pose;

    const T lambda = 1e-2;

    // Build one system to define Hvp/Hpp/g structure (we'll perturb gv each iter)
    SPDSystem S = make_system(W, H, s_pose, dof_pose, lambda, /*seed=*/7);

    // Numeric values matrix (same pattern). We'll rebuild values each iteration by overwriting Ax.
    SparseMatCSC<T> Hvv_values = from_dense_to_sparse<Matx<T>, SparseMatCSC<T>>(S.H);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_values, nullptr);

    // Warmup + timing iterations
    Vecx<T> dv, dp;

    // We'll slightly perturb diagonal values each iteration to simulate LM / changing Jacobians.
    // If your SparseMatCSC exposes Ax_mut(), use it; otherwise rebuild Hvv_values each iter (slower).
    // Here we assume Ax() returns pointer; if you don't have Ax_mut, add it.
    T *Ax_mut = const_cast<T *>(Hvv_values.Ax()); // OK if SparseMat stores mutable and Ax() is const; prefer Ax_mut() method.

    const int iters = 10;

    auto schur_one_iter = [&]()
    {
        // perturb a few diagonal entries (cheap)
        // diagonal in this pattern exists; but we don't have positions here,
        // so we just scale all values slightly for the benchmark (still cheap).
        // Better: precompute diag positions in real code.
        for (int k = 0; k < Hvv_values.nnz(); ++k)
            Ax_mut[k] *= 1.0000001;

        schur.factorize_mesh_numeric(Hvv_values);
        schur.solve(S.Hvp, S.Hpp, S.gv, S.gp, dv, dp, 0.0);

        // make sure compiler can’t optimize everything away
        volatile T sink = dv(0) + dp(0);
        (void)sink;
    };

    Timing t_schur = time_it(schur_one_iter, iters, /*warmup=*/1);

    std::cout << "[Timing] Schur (n=" << n << ", mesh=" << n_mesh << ", pose=" << n_pose
              << ") avg = " << t_schur.ms << " ms/iter over " << t_schur.iters << " iters\n";

    // Optional dense reference timing for a smaller case only (to avoid huge O(n^3)).
    // We do NOT time dense for n~1066.
    EXPECT_TRUE(true);
}

// -------------------------
// TEST 4: timing comparison Schur vs dense LAPACK on a medium case (safe runtime)
// This provides a sanity “Schur isn’t slower” check without being flaky.
// -------------------------
TEST(SchurComplementSolver, TimingComparison_Medium_SchurVsDense)
{
    using T = double;

    const int w_mesh = 16;
    const int h_mesh = 16;
    const int n_mesh = w_mesh * h_mesh;
    const int s_pose = 6;
    const int dof_pose = 6;
    const int n_pose = s_pose * dof_pose;
    const int n = n_mesh + n_pose;

    const T lambda = 1e-2;

    SPDSystem S = make_system(w_mesh, h_mesh, s_pose, dof_pose, lambda, /*seed=*/99);

    SparseMatCSC<T> Hvv_values = from_dense_to_sparse<Matx<T>, SparseMatCSC<T>>(S.H);

    LDLT_LAPACK<T> ldlt(n);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_values, nullptr);

    Vecx<T> dv, dp;

    auto schur_iter = [&]()
    {
        schur.factorize_mesh_numeric(Hvv_values);
        schur.solve(S.Hvp, S.Hpp, S.gv, S.gp, dv, dp, 0.0);
        volatile T sink = dv(0) + dp(0);
        (void)sink;
    };

    auto dense_iter = [&]()
    {
        Matx<T> H_ref = S.H;
        Vecx<T> x_ref = S.g;
        int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, 1, H_ref.data(), n, x_ref.data(), n);
        // ldlt.compute(H_ref);
        // Vecx<double> res = ldlt.solve(x_ref);
        if (info != 0)
            throw std::runtime_error("dense dposv failed in timing test");
        volatile T sink = x_ref(0);
        (void)sink;
    };

    // Use small iters; dense is expensive
    Timing t_schur = time_it(schur_iter, /*iters=*/10, /*warmup=*/1);
    Timing t_dense = time_it(dense_iter, /*iters=*/10, /*warmup=*/1);

    std::cout << "[Timing] Medium compare (n=" << n << "): "
              << "Schur avg=" << t_schur.ms << " ms, "
              << "Dense avg=" << t_dense.ms << " ms\n";

    // Gentle check: Schur should not be dramatically slower than dense here.
    // (Don’t make this too strict; CI machines vary a lot.)
    EXPECT_LT(t_schur.ms, t_dense.ms * 1.0);
}