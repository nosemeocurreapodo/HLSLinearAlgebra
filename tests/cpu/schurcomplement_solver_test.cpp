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

// Build SPD system H = J^T J + lambda I, g = J^T r
struct SPDSystem
{
    int n_mesh = 0;
    int n_pose = 0;
    int n = 0;
    int m = 0;

    Matx<double> H; // n x n
    Vecx<double> g; // n

    Matx<double> Hvp; // n_mesh x n_pose
    Matx<double> Hpp; // n_pose x n_pose
    Vecx<double> gv;  // n_mesh
    Vecx<double> gp;  // n_pose
};

// Generates a random SPD normal-equation-like system
static SPDSystem make_system(int n_mesh, int n_pose, int m, double lambda, uint32_t seed)
{
    SPDSystem S;
    S.n_mesh = n_mesh;
    S.n_pose = n_pose;
    S.n = n_mesh + n_pose;
    S.m = m;

    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    Matx<double> J(m, S.n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < S.n; ++j)
            J(i, j) = nd(rng);

    Vecx<double> r(m);
    for (int i = 0; i < m; ++i)
        r(i) = nd(rng);

    S.H = Matx<double>(S.n, S.n);
    for (int i = 0; i < S.n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < m; ++k)
                sum += J(k, i) * J(k, j);

            S.H(i, j) = sum;
            S.H(j, i) = sum;
        }
        S.H(i, i) += lambda;
    }

    S.g = Vecx<double>(S.n);
    for (int j = 0; j < S.n; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < m; ++k)
            sum += J(k, j) * r(k);
        S.g(j) = sum;
    }

    S.gv = Vecx<double>(n_mesh);
    S.gp = Vecx<double>(n_pose);
    for (int i = 0; i < n_mesh; ++i)
        S.gv(i) = S.g(i);
    for (int i = 0; i < n_pose; ++i)
        S.gp(i) = S.g(n_mesh + i);

    S.Hvp = Matx<double>(n_mesh, n_pose);
    for (int pc = 0; pc < n_pose; ++pc)
        for (int vr = 0; vr < n_mesh; ++vr)
            S.Hvp(vr, pc) = S.H(vr, n_mesh + pc);

    S.Hpp = Matx<double>(n_pose, n_pose);
    for (int c = 0; c < n_pose; ++c)
        for (int r2 = 0; r2 < n_pose; ++r2)
            S.Hpp(r2, c) = S.H(n_mesh + r2, n_mesh + c);

    return S;
}

// Hvv pattern builders
static SparseMatCSC build_full_lower_pattern(int n_mesh)
{
    SparseMatCSC P(n_mesh, /*store_lower_only=*/true);
    for (int i = 0; i < n_mesh; ++i)
        for (int j = 0; j <= i; ++j)
            P.add(i, j, 1.0);
    P.finalize();
    return P;
}

// 2D grid stencil pattern (4-neighbor) for n_mesh = W*H
static SparseMatCSC build_grid4_pattern(int W, int H)
{
    const int n_mesh = W * H;
    auto vid = [W](int x, int y)
    { return y * W + x; };

    SparseMatCSC P(n_mesh, /*store_lower_only=*/true);

    // always diagonal
    for (int v = 0; v < n_mesh; ++v)
        P.add(v, v, 1.0);

    // connect left and up (lower triangle)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            int v = vid(x, y);
            if (x > 0)
                P.add(v, vid(x - 1, y), 1.0);
            if (y > 0)
                P.add(v, vid(x, y - 1), 1.0);
        }

    P.finalize();
    return P;
}

// Build numeric Hvv_values matching *full-lower* pattern directly from S.H block.
static SparseMatCSC build_full_lower_values(const SPDSystem &S)
{
    SparseMatCSC A(S.n_mesh, /*store_lower_only=*/true);
    for (int i = 0; i < S.n_mesh; ++i)
        for (int j = 0; j <= i; ++j)
            A.add(i, j, S.H(i, j));
    A.finalize();
    return A;
}

// Build numeric Hvv_values matching grid stencil pattern by sampling S.H on stencil entries.
// NOTE: this is a "pattern-consistent" numeric matrix, not necessarily equal to S.Hvv (dense).
// That’s fine for testing the solver pipeline; correctness is checked against dense solve only
// when Hvv pattern is full. For stencil pattern, we check internal residual consistency
// of the reconstructed full solution against the *same* block system we feed the solver.
static SparseMatCSC build_grid4_values_from_denseH(const SPDSystem &S, int W, int H)
{
    SparseMatCSC A(S.n_mesh, /*store_lower_only=*/true);
    auto vid = [W](int x, int y)
    { return y * W + x; };

    for (int v = 0; v < S.n_mesh; ++v)
        A.add(v, v, S.H(v, v));

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            int v = vid(x, y);
            if (x > 0)
            {
                int u = vid(x - 1, y);
                A.add(v, u, S.H(v, u));
            }
            if (y > 0)
            {
                int u = vid(x, y - 1);
                A.add(v, u, S.H(v, u));
            }
        }

    A.finalize();
    return A;
}

struct Timing
{
    double ms = 0.0;
    int iters = 0;
};

template <class Fn>
static Timing time_it(Fn &&fn, int iters, int warmup = 1)
{
    for (int i = 0; i < warmup; ++i)
        fn();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        fn();
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return Timing{ms / iters, iters};
}

// -------------------------
// TEST 1: correctness vs dense reference (full Hvv pattern)
// -------------------------
TEST(SchurComplementSolver, FullPatternMatchesDenseReference_Small)
{
    const int n_mesh = 15;
    const int n_pose = 8;
    const int n = n_mesh + n_pose;

    const int m = 80;
    const double lambda = 1e-2;

    SPDSystem S = make_system(n_mesh, n_pose, m, lambda, /*seed=*/12345);

    SparseMatCSC Hvv_pattern = build_full_lower_pattern(n_mesh);
    SparseMatCSC Hvv_values = build_full_lower_values(S);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_pattern, nullptr);
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
    const int W = 16, H = 16; // 256 mesh vars
    const int n_mesh = W * H;
    const int n_pose = 12; // e.g. 2 poses * 6 dof
    const int n = n_mesh + n_pose;

    const int m = 200;
    const double lambda = 1e-2;

    SPDSystem S = make_system(n_mesh, n_pose, m, lambda, /*seed=*/424242);

    SparseMatCSC Hvv_pattern = build_grid4_pattern(W, H);
    SparseMatCSC Hvv_values = build_grid4_values_from_denseH(S, W, H);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_pattern, nullptr);
    schur.factorize_mesh_numeric(Hvv_values);

    Vecx<double> dv, dp;
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
    Vecx<double> rv = Vecx<double>::Zero(n_mesh);
    const int *Ap = Hvv_values.Ap();
    const int *Ai = Hvv_values.Ai();
    const double *Ax = Hvv_values.Ax();

    // Hvv is symmetric stored lower triangle; apply full symmetric multiply
    // rv += Hvv * dv
    for (int col = 0; col < n_mesh; ++col)
    {
        for (int p = Ap[col]; p < Ap[col + 1]; ++p)
        {
            int row = Ai[p];
            double v = Ax[p];
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
    Vecx<double> rp = Vecx<double>::Zero(n_pose);

    for (int i = 0; i < n_pose; ++i)
    {
        double sum = 0.0;
        for (int r2 = 0; r2 < n_mesh; ++r2)
            sum += S.Hvp(r2, i) * dv(r2);
        rp(i) = sum;
    }

    for (int c = 0; c < n_pose; ++c)
    {
        double sum = 0.0;
        for (int r2 = 0; r2 < n_pose; ++r2)
            sum += S.Hpp(c, r2) * dp(r2);
        rp(c) += sum;
        rp(c) -= S.gp(c);
    }

    // Check infinity norms are small
    double rv_inf = 0.0, rp_inf = 0.0;
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
    const int W = 32, H = 32; // 1024 mesh vars (your real scale)
    const int n_mesh = W * H;
    const int n_pose = 42; // 7 poses * 6 dof
    const int n = n_mesh + n_pose;

    // Keep m moderate; we mainly want timing of solve path
    const int m = 300;
    const double lambda = 1e-2;

    // Build one system to define Hvp/Hpp/g structure (we'll perturb gv each iter)
    SPDSystem S = make_system(n_mesh, n_pose, m, lambda, /*seed=*/7);

    // Use a grid stencil pattern for Hvv (realistic mesh locality)
    SparseMatCSC Hvv_pattern = build_grid4_pattern(W, H);

    // Numeric values matrix (same pattern). We'll rebuild values each iteration by overwriting Ax.
    SparseMatCSC Hvv_values = build_grid4_values_from_denseH(S, W, H);

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_pattern, nullptr);

    // Warmup + timing iterations
    Vecx<double> dv, dp;

    // We'll slightly perturb diagonal values each iteration to simulate LM / changing Jacobians.
    // If your SparseMatCSC exposes Ax_mut(), use it; otherwise rebuild Hvv_values each iter (slower).
    // Here we assume Ax() returns pointer; if you don't have Ax_mut, add it.
    double *Ax_mut = const_cast<double *>(Hvv_values.Ax()); // OK if SparseMat stores mutable and Ax() is const; prefer Ax_mut() method.

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
        volatile double sink = dv(0) + dp(0);
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
    const int n_mesh = 64 * 64;
    const int n_pose = 8 * 7;
    const int n = n_mesh + n_pose;

    const int m = 250;
    const double lambda = 1e-2;

    SPDSystem S = make_system(n_mesh, n_pose, m, lambda, /*seed=*/99);

    // Full-lower Hvv pattern so Schur matches dense reference exactly
    SparseMatCSC Hvv_pattern = build_full_lower_pattern(n_mesh);
    SparseMatCSC Hvv_values = build_full_lower_values(S);

    LDLT_LAPACK<double> ldlt;

    SchurSolver schur;
    schur.set_pose_dim(n_pose);
    schur.analyze_mesh_pattern(Hvv_pattern, nullptr);

    Vecx<double> dv, dp;

    auto schur_iter = [&]()
    {
        schur.factorize_mesh_numeric(Hvv_values);
        schur.solve(S.Hvp, S.Hpp, S.gv, S.gp, dv, dp, 0.0);
        volatile double sink = dv(0) + dp(0);
        (void)sink;
    };

    auto dense_iter = [&]()
    {
        Matx<double> H_ref = S.H;
        Vecx<double> x_ref = S.g;
        int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, 1, H_ref.data(), n, x_ref.data(), n);
        // ldlt.compute(H_ref);
        // Vecx<double> res = ldlt.solve(x_ref);
        if (info != 0)
            throw std::runtime_error("dense dposv failed in timing test");
        volatile double sink = x_ref(0);
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