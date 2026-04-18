// test_ldlt_eigen.cpp

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

// Adjust this to the actual path of your linalg headers
#include "linalg/linalg.h"
#include "linalg/linalgx.h"
#include "linalg/ldlt_solver.h"
#include "linalg/ldlt_solverx.h" // if needed, depending on your structure
#include "linalg/ldlt_solverx_lapack.h"
#include "linalg/dense_cholmod_solverx.h"
#include "common.h"

constexpr double kTol = 1e-8;

using namespace linalg;

// Generic helper to run a bunch of random SPD systems for size N
template <int size>
void TestLDLTxSolverRandomSPD(int num_systems = 5, int num_rhs_per_system = 3)
{
    // using MatN    = linalg::Matx<double, N, N>;
    // using VecN    = linalg::Vec<double, N>;
    // using LDLTN   = linalg::LDLT<double, N>;
    // using MatrixN = Eigen::Matrix<double, N, N>;
    // using VectorN = Eigen::Matrix<double, N, 1>;

    // Deterministic RNG so tests are repeatable
    std::mt19937 gen(12345 + size);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int sys = 0; sys < num_systems; ++sys)
    {
        // Build a random SPD matrix A_eig = M^T * M + alpha * I
        Eigen::MatrixX<double> M(size, size);
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                M(i, j) = dist(gen);
            }
        }

        Eigen::MatrixX<double> A_eig = M.transpose() * M;
        A_eig += 0.5 * Eigen::MatrixX<double>::Identity(size, size); // strengthen positive-definiteness

        // Copy to your linalg::MatN
        Mat<double, size, size> A_linalg;
        Matx<double> Ax_linalg(size, size);
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                A_linalg(i, j) = A_eig(i, j);
                Ax_linalg(i, j) = A_eig(i, j);
            }
        }

        LDLT<double, size> ldlt_linalg;
        ldlt_linalg.compute(A_linalg);

        // Factorization with your LDLT
        LDLTx<double> ldltx_linalg(size);
        ldltx_linalg.compute(Ax_linalg);

        LDLT_LAPACK<double> ldlt_lapack(size);
        ldlt_lapack.compute(Ax_linalg);

        DENSE_CHOLESKY_CHOLMOD<double> cholesky_lapack(size);
        cholesky_lapack.compute(Ax_linalg);

        // Factorization with Eigen's LDLT
        Eigen::LDLT<Eigen::MatrixX<double>> ldlt_eig(A_eig);

        Eigen::LLT<Eigen::MatrixX<double>> llt_eig(A_eig);

        // ldlt_eig.compute(A_eig);
        ASSERT_EQ(ldlt_eig.info(), Eigen::Success);

        // Solve several RHS with the same factorization
        for (int rhs = 0; rhs < num_rhs_per_system; ++rhs)
        {
            // Random RHS
            Eigen::VectorX<double> b_eig(size);
            for (int i = 0; i < size; ++i)
            {
                b_eig(i) = dist(gen);
            }

            // Copy b to your VecN
            Vec<double, size> b_linalg;
            Vecx<double> bx_linalg(size);
            for (int i = 0; i < size; ++i)
            {
                b_linalg(i) = b_eig(i);
                bx_linalg(i) = b_eig(i);
            }

            // Solve with your solver
            Vec<double, size> x_linalg = ldlt_linalg.solve(b_linalg);

            Vecx<double> xx_linalg = ldltx_linalg.solve(bx_linalg);

            Vecx<double> x_lapack = ldlt_lapack.solve(bx_linalg);

            Vecx<double> x_cholesky = cholesky_lapack.solve(bx_linalg);

            // Solve with Eigen's LDLT
            Eigen::VectorX<double> x_eig = ldlt_eig.solve(b_eig);

            Eigen::VectorX<double> x_eig2 = llt_eig.solve(b_eig);

            // Compare component-wise
            for (int i = 0; i < size; ++i)
            {
                EXPECT_NEAR(x_eig(i), x_linalg(i), kTol)
                    << "Eigen Mismatch at size N=" << size
                    << ", system " << sys
                    << ", rhs " << rhs
                    << ", index " << i;

                EXPECT_NEAR(x_eig(i), xx_linalg(i), kTol)
                    << "Eigen Mismatch at size N=" << size
                    << ", system " << sys
                    << ", rhs " << rhs
                    << ", index " << i;

                EXPECT_NEAR(x_eig(i), x_lapack(i), kTol)
                    << "Lapack Mismatch at size N=" << size
                    << ", system " << sys
                    << ", rhs " << rhs
                    << ", index " << i;

                // EXPECT_NEAR(x_eig(i), x_cholesky(i), kTol)
                //     << "Lapack Mismatch at size N=" << size
                //     << ", system " << sys
                //     << ", rhs " << rhs
                //     << ", index " << i;

                EXPECT_NEAR(x_eig(i), x_eig2(i), kTol)
                    << "Lapack Mismatch at size N=" << size
                    << ", system " << sys
                    << ", rhs " << rhs
                    << ", index " << i;
            }
        }
    }
}

// --- Actual tests for various sizes ---

TEST(LDLTx_solver, RandomSPD_32x32)
{
    // A bit larger to make sure things still behave.
    TestLDLTxSolverRandomSPD<16 * 16>();
}

TEST(LDLTx_solver, TimingComparison)
{
    using T = float;

    const int w_mesh = 16;
    const int h_mesh = 16;
    const int n_mesh = w_mesh * h_mesh;
    const int s_pose = 3;
    const int dof_pose = 6;
    const int n_pose = s_pose * dof_pose;
    const int n = n_mesh + n_pose;

    const int m = 250;
    const T lambda = 1e-2;

    SPDSystem S = make_system(w_mesh, h_mesh, s_pose, dof_pose, lambda, /*seed=*/99);

    Eigen::MatrixX<T> Hd_eigen(n, n);
    Eigen::SparseMatrix<T> Hs_eigen = from_dense_to_sparse<Matx<T>, Eigen::SparseMatrix<T>>(S.H);
    Eigen::VectorX<T> g_eigen(n);

    Mat<T, n, n> H_linalg;
    Mat<T, n, 1> g_linalg;

    for (int i = 0; i < n; i++)
    {
        g_eigen(i) = S.g(i);
        g_linalg(i, 0) = S.g(i);
        for (int j = 0; j < n; j++)
        {
            Hd_eigen(i, j) = S.H(i, j);
            H_linalg(i, j) = S.H(i, j);
        }
    }

    LDLT<T, n> ldlt_linalg;
    LDLTx<T> ldltx_linalg(n);
    LDLT_LAPACK<T> ldlt_lapack(n);
    DENSE_CHOLESKY_CHOLMOD<T> cholesky_lapack(n);
    Eigen::LDLT<Eigen::MatrixX<T>> ldlt_eig(n);
    Eigen::LLT<Eigen::MatrixX<T>> llt_eig(n);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> ldlt_sparse_eig;

    ldlt_sparse_eig.analyzePattern(Hs_eigen);

    auto ldlt_linalg_iter = [&]()
    {
        ldlt_linalg.compute(H_linalg);
        Mat<T, n, 1> x_ref = ldlt_linalg.solve(g_linalg);
        volatile T sink = x_ref(0, 0);
        (void)sink;
    };

    auto ldltx_linalg_iter = [&]()
    {
        ldltx_linalg.compute(S.H);
        Vecx<T> x_ref(n);
        VecView<T> x_ref_view(x_ref.data(), n);
        ldltx_linalg.solve(S.g);
        volatile T sink = x_ref(0);
        (void)sink;
    };

    auto ldlt_lapack_iter = [&]()
    {
        ldlt_lapack.compute(S.H);
        Vecx<T> x_ref = ldlt_lapack.solve(S.g);
        volatile T sink = x_ref(0);
        (void)sink;
    };

    auto cholesky_lapack_iter = [&]()
    {
        cholesky_lapack.compute(S.H);
        Vecx<T> x_ref = cholesky_lapack.solve(S.g);
        volatile T sink = x_ref(0);
        (void)sink;
    };

    auto ldlt_eigen_iter = [&]()
    {
        ldlt_eig.compute(Hd_eigen);
        Eigen::VectorX<T> x_ref = ldlt_eig.solve(g_eigen);
        volatile T sink = x_ref(0);
        (void)sink;
    };

    auto llt_eigen_iter = [&]()
    {
        llt_eig.compute(Hd_eigen);
        Eigen::VectorX<T> x_ref = llt_eig.solve(g_eigen);
        volatile T sink = x_ref(0);
        (void)sink;
    };

    auto ldlt_sparse_eigen_iter = [&]()
    {
        ldlt_sparse_eig.factorize(Hs_eigen);
        Eigen::VectorX<T> x_ref = ldlt_sparse_eig.solve(g_eigen);
        volatile T sink = x_ref(0);
        (void)sink;
    };

    // Use small iters; dense is expensive
    Timing t_linalg = time_it(ldlt_linalg_iter, /*iters=*/10, /*warmup=*/1);
    Timing t_linalgx = time_it(ldltx_linalg_iter, /*iters=*/10, /*warmup=*/1);

    Timing t_lapack = time_it(ldlt_lapack_iter, /*iters=*/10, /*warmup=*/1);
    Timing t_clapack = time_it(cholesky_lapack_iter, /*iters=*/10, /*warmup=*/1);

    Timing t_eigen = time_it(ldlt_eigen_iter, /*iters=*/10, /*warmup=*/1);
    Timing t_eigen2 = time_it(llt_eigen_iter, /*iters=*/10, /*warmup=*/1);
    Timing t_eigen3 = time_it(ldlt_sparse_eigen_iter, /*iters=*/10, /*warmup=*/1);

    std::cout << "[Timing] Medium compare (n=" << n << "): "
              << "Linalg avg=" << t_linalg.ms << " ms, "
              << "Linalgc avg=" << t_linalgx.ms << " ms, "
              << "Lapack avg=" << t_lapack.ms << " ms, "
              << "cLapack avg=" << t_clapack.ms << " ms, "
              << "Eigen avg=" << t_eigen.ms << " ms, "
              << "Eigen avg=" << t_eigen2.ms << " ms "
              << "Eigen avg=" << t_eigen3.ms << " ms\n";

    // Gentle check: Schur should not be dramatically slower than dense here.
    // (Don’t make this too strict; CI machines vary a lot.)
    EXPECT_LT(t_linalg.ms, t_lapack.ms * 1.0);
    EXPECT_LT(t_linalg.ms, t_eigen.ms * 1.0);
}