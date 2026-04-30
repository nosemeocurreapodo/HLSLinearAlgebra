// test_matx_eigen.cpp

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>

#include "linalg/linalgx.h"

using linalg::Matx;
using linalg::Vecx;
using linalg::VecOrient;

namespace {

constexpr double kTol = 1e-9;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

template <typename T>
Matx<T> MatxFromEigen(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M)
{
    Matx<T> out(M.rows(), M.cols());
    for (int r = 0; r < M.rows(); ++r)
    {
        for (int c = 0; c < M.cols(); ++c)
        {
            out(r, c) = M(r, c);
        }
    }
    return out;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
EigenFromMatx(const Matx<T> &M)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> out(M.rows(), M.cols());
    for (int r = 0; r < M.rows(); ++r)
    {
        for (int c = 0; c < M.cols(); ++c)
        {
            out(r, c) = M(r, c);
        }
    }
    return out;
}

template <typename T>
double FrobeniusDiff(const Matx<T> &M,
                     const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &E)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> M_e = EigenFromMatx(M);
    return (M_e - E).norm();
}

Eigen::MatrixXd randomMatrix(int rows, int cols, std::mt19937 &gen)
{
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::MatrixXd M(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            M(r, c) = dist(gen);
    return M;
}

} // namespace

//------------------------------------------------------------------------------
// Basic Zero / Identity behavior
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, ZeroAndIdentitySmall)
{
    const int rows = 7;
    const int cols = 5;

    // Zero
    Eigen::MatrixXd Z_e = Eigen::MatrixXd::Zero(rows, cols);
    Matx<double>   Z_m = Matx<double>::Zero(rows, cols);

    EXPECT_NEAR(0.0, FrobeniusDiff(Z_m, Z_e), kTol);

    // Identity via static factory
    const int n = 10;
    Eigen::MatrixXd I_e = Eigen::MatrixXd::Identity(n, n);
    Matx<double>    I_m = Matx<double>::Identity(n, n);

    EXPECT_NEAR(0.0, FrobeniusDiff(I_m, I_e), kTol);

    // Identity via setZero + setIdentity on an existing matrix
    Matx<double> A(n, n);
    A.setZero();
    A.setIdentity();

    EXPECT_NEAR(0.0, FrobeniusDiff(A, I_e), kTol);
}

//------------------------------------------------------------------------------
// Transpose tests (including a large one)
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, TransposeLarge)
{
    std::mt19937 gen(1234);

    const int rows = 80;
    const int cols = 120;

    Eigen::MatrixXd A_e = randomMatrix(rows, cols, gen);
    Matx<double>    A_m = MatxFromEigen<double>(A_e);

    Matx<double>    AT_m = A_m.transpose();
    Eigen::MatrixXd AT_e = A_e.transpose();

    EXPECT_EQ(AT_m.rows(), AT_e.rows());
    EXPECT_EQ(AT_m.cols(), AT_e.cols());
    EXPECT_NEAR(0.0, FrobeniusDiff(AT_m, AT_e), kTol);
}

//------------------------------------------------------------------------------
// Addition / subtraction on reasonably large matrices
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, AdditionAndSubtractionLarge)
{
    std::mt19937 gen(4321);

    const int rows = 80;
    const int cols = 70;

    Eigen::MatrixXd A_e = randomMatrix(rows, cols, gen);
    Eigen::MatrixXd B_e = randomMatrix(rows, cols, gen);

    Matx<double> A_m = MatxFromEigen<double>(A_e);
    Matx<double> B_m = MatxFromEigen<double>(B_e);

    Matx<double> C_m = A_m + B_m;
    Eigen::MatrixXd C_e = A_e + B_e;
    EXPECT_NEAR(0.0, FrobeniusDiff(C_m, C_e), kTol);

    Matx<double> D_m = A_m - B_m;
    Eigen::MatrixXd D_e = A_e - B_e;
    EXPECT_NEAR(0.0, FrobeniusDiff(D_m, D_e), kTol);
}

//------------------------------------------------------------------------------
// Matrix-matrix multiplication: rectangular and square big cases
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, MultiplyRectangularLarge)
{
    std::mt19937 gen(999);

    const int m = 100;
    const int k = 60;
    const int n = 80;

    Eigen::MatrixXd A_e = randomMatrix(m, k, gen);
    Eigen::MatrixXd B_e = randomMatrix(k, n, gen);

    Matx<double> A_m = MatxFromEigen<double>(A_e);
    Matx<double> B_m = MatxFromEigen<double>(B_e);

    Matx<double> C_m = A_m * B_m;
    Eigen::MatrixXd C_e = A_e * B_e;

    EXPECT_EQ(C_m.rows(), C_e.rows());
    EXPECT_EQ(C_m.cols(), C_e.cols());
    EXPECT_NEAR(0.0, FrobeniusDiff(C_m, C_e), kTol);
}

TEST(MatxEigenCompat, MultiplySquareVeryLarge)
{
    std::mt19937 gen(2025);

    const int n = 200; // This is your "large" SPD-ish playpen

    Eigen::MatrixXd A_e = randomMatrix(n, n, gen);
    Eigen::MatrixXd B_e = randomMatrix(n, n, gen);

    Matx<double> A_m = MatxFromEigen<double>(A_e);
    Matx<double> B_m = MatxFromEigen<double>(B_e);

    Matx<double> C_m = A_m * B_m;
    Eigen::MatrixXd C_e = A_e * B_e;

    EXPECT_EQ(C_m.rows(), C_e.rows());
    EXPECT_EQ(C_m.cols(), C_e.cols());
    EXPECT_NEAR(0.0, FrobeniusDiff(C_m, C_e), 1e-8); // a bit looser for big products
}

//------------------------------------------------------------------------------
// Scalar operations & unary minus
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, ScalarMultiplyAndDivideLarge)
{
    std::mt19937 gen(111);

    const int rows = 120;
    const int cols = 90;

    Eigen::MatrixXd A_e = randomMatrix(rows, cols, gen);
    Matx<double>    A_m = MatxFromEigen<double>(A_e);

    const double s = 1.2345;

    Matx<double> S1_m = A_m * s;
    Matx<double> S2_m = s * A_m;
    Matx<double> Q_m  = A_m / s;

    Eigen::MatrixXd S_e = A_e * s;
    Eigen::MatrixXd Q_e = A_e / s;

    EXPECT_NEAR(0.0, FrobeniusDiff(S1_m, S_e), kTol);
    EXPECT_NEAR(0.0, FrobeniusDiff(S2_m, S_e), kTol);
    EXPECT_NEAR(0.0, FrobeniusDiff(Q_m, Q_e), kTol);
}

TEST(MatxEigenCompat, UnaryMinus)
{
    std::mt19937 gen(222);

    const int rows = 50;
    const int cols = 40;

    Eigen::MatrixXd A_e = randomMatrix(rows, cols, gen);
    Matx<double>    A_m = MatxFromEigen<double>(A_e);

    Matx<double> Neg_m = -A_m;
    Eigen::MatrixXd Neg_e = -A_e;

    EXPECT_NEAR(0.0, FrobeniusDiff(Neg_m, Neg_e), kTol);
}

//------------------------------------------------------------------------------
// Norm (Frobenius norm)
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, NormMatchesEigen)
{
    std::mt19937 gen(333);

    const int rows = 100;
    const int cols = 200;

    Eigen::MatrixXd A_e = randomMatrix(rows, cols, gen);
    Matx<double>    A_m = MatxFromEigen<double>(A_e);

    double n_e = A_e.norm();
    double n_m = A_m.norm();

    EXPECT_NEAR(n_e, n_m, kTol);
}

//------------------------------------------------------------------------------
// conv() vs Frobenius inner product
//------------------------------------------------------------------------------

TEST(MatxEigenCompat, ConvMatchesFrobeniusInnerProduct)
{
    std::mt19937 gen(444);

    const int rows = 64;
    const int cols = 64;

    Eigen::MatrixXd A_e = randomMatrix(rows, cols, gen);
    Eigen::MatrixXd B_e = randomMatrix(rows, cols, gen);

    Matx<double> A_m = MatxFromEigen<double>(A_e);
    Matx<double> B_m = MatxFromEigen<double>(B_e);

    double conv_m = A_m.template dot<double>(B_m);

    // Frobenius inner product: sum_ij A(i,j)*B(i,j)
    double conv_e = (A_e.array() * B_e.array()).sum();

    EXPECT_NEAR(conv_e, conv_m, kTol);
}

//------------------------------------------------------------------------------
// Vecx sanity: compare to Eigen vectors (no dot() here because of the size_ issue)
//------------------------------------------------------------------------------
/*
TEST(VecxEigenCompat, ColumnVectorMatchesEigen)
{
    std::mt19937 gen(555);
    std::normal_distribution<> dist(0.0, 1.0);

    const int n = 150;

    Eigen::VectorXd v_e(n);
    for (int i = 0; i < n; ++i)
        v_e(i) = dist(gen);

    Vecx<double, VecOrient::Column> v_m(n);
    for (int i = 0; i < n; ++i)
        v_m(i) = v_e(i);

    // Treat v_m as an n×1 Matx
    const Matx<double> &m_ref = v_m;
    Eigen::MatrixXd v_eM = v_e; // n×1

    EXPECT_EQ(m_ref.rows(), n);
    EXPECT_EQ(m_ref.cols(), 1);
    EXPECT_NEAR(0.0, FrobeniusDiff(m_ref, v_eM), kTol);
}

TEST(VecxEigenCompat, RowVectorMatchesEigen)
{
    std::mt19937 gen(666);
    std::normal_distribution<> dist(0.0, 1.0);

    const int n = 160;

    Eigen::RowVectorXd v_e(n);
    for (int i = 0; i < n; ++i)
        v_e(i) = dist(gen);

    Vecx<double, VecOrient::Row> v_m(n);
    for (int i = 0; i < n; ++i)
        v_m(i) = v_e(i);

    const Matx<double> &m_ref = v_m;
    Eigen::MatrixXd v_eM(1, n);
    v_eM.row(0) = v_e;

    EXPECT_EQ(m_ref.rows(), 1);
    EXPECT_EQ(m_ref.cols(), n);
    EXPECT_NEAR(0.0, FrobeniusDiff(m_ref, v_eM), kTol);
}
*/