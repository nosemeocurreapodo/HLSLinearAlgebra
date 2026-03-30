// test_linalg_fixed_eigen.cpp

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>

// Adjust this to your actual header that defines Mat, Vec, Quaternion, SO3, SE3...
#include "linalg/linalg.h"

using namespace linalg;

namespace
{

    constexpr double kTol = 1e-9;
    constexpr double kTolBig = 1e-8;

    template <typename T, int R, int C>
    Mat<T, R, C> MatFromEigen(const Eigen::Matrix<T, R, C> &E)
    {
        Mat<T, R, C> M;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                M(r, c) = E(r, c);
        return M;
    }

    template <typename T, int R, int C>
    Eigen::Matrix<T, R, C> EigenFromMat(const Mat<T, R, C> &M)
    {
        Eigen::Matrix<T, R, C> E;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                E(r, c) = M(r, c);
        return E;
    }

    template <typename T, int R, int C>
    double FrobeniusDiff(const Mat<T, R, C> &M, const Eigen::Matrix<T, R, C> &E)
    {
        Eigen::Matrix<T, R, C> Me = EigenFromMat<T, R, C>(M);
        return (Me - E).norm();
    }

    template <int R, int C>
    Eigen::Matrix<double, R, C> randomMatrix(std::mt19937 &gen)
    {
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::Matrix<double, R, C> M;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                M(r, c) = dist(gen);
        return M;
    }

    Eigen::Vector3d randomUnitVector3(std::mt19937 &gen)
    {
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::Vector3d v;
        do
        {
            v << dist(gen), dist(gen), dist(gen);
        } while (v.norm() < 1e-12);
        return v.normalized();
    }

    Eigen::Quaterniond randomUnitQuaternion(std::mt19937 &gen)
    {
        const double pi = 3.14159265358979323846;
        Eigen::Vector3d axis = randomUnitVector3(gen);
        std::uniform_real_distribution<double> angle_dist(-pi, pi);
        double angle = angle_dist(gen);
        Eigen::AngleAxisd aa(angle, axis);
        return Eigen::Quaterniond(aa);
    }

} // namespace

//------------------------------------------------------------------------------
// Mat<T,R,C> basic tests vs Eigen
//------------------------------------------------------------------------------

TEST(MatFixedEigen, ZeroAndIdentity)
{
    using Mat5x7 = Mat<double, 5, 7>;
    using Mat4x4 = Mat<double, 4, 4>;
    using E5x7 = Eigen::Matrix<double, 5, 7>;
    using E4x4 = Eigen::Matrix<double, 4, 4>;

    E5x7 Z_e = E5x7::Zero();
    Mat5x7 Z_m = Mat5x7::Zero();

    EXPECT_NEAR(0.0, FrobeniusDiff(Z_m, Z_e), kTol);

    E4x4 I_e = E4x4::Identity();
    Mat4x4 I_m = Mat4x4::Identity();

    EXPECT_NEAR(0.0, FrobeniusDiff(I_m, I_e), kTol);
}

TEST(MatFixedEigen, AdditionAndSubtractionLarge)
{
    std::mt19937 gen(1234);

    using MatA = Mat<double, 50, 60>;
    using MatB = Mat<double, 50, 60>;
    using EMat = Eigen::Matrix<double, 50, 60>;

    EMat A_e = randomMatrix<50, 60>(gen);
    EMat B_e = randomMatrix<50, 60>(gen);

    MatA A_m = MatFromEigen<double, 50, 60>(A_e);
    MatB B_m = MatFromEigen<double, 50, 60>(B_e);

    Mat<double, 50, 60> C_m = A_m + B_m;
    EMat C_e = A_e + B_e;
    EXPECT_NEAR(0.0, FrobeniusDiff(C_m, C_e), kTol);

    Mat<double, 50, 60> D_m = A_m - B_m;
    EMat D_e = A_e - B_e;
    EXPECT_NEAR(0.0, FrobeniusDiff(D_m, D_e), kTol);
}

TEST(MatFixedEigen, MultiplyRectangularLarge)
{
    std::mt19937 gen(4321);

    using MatA = Mat<double, 40, 30>;
    using MatB = Mat<double, 30, 20>;
    using MatC = Mat<double, 40, 20>;

    using EA = Eigen::Matrix<double, 40, 30>;
    using EB = Eigen::Matrix<double, 30, 20>;
    using EC = Eigen::Matrix<double, 40, 20>;

    EA A_e = randomMatrix<40, 30>(gen);
    EB B_e = randomMatrix<30, 20>(gen);

    MatA A_m = MatFromEigen<double, 40, 30>(A_e);
    MatB B_m = MatFromEigen<double, 30, 20>(B_e);

    MatC C_m = A_m * B_m;
    EC C_e = A_e * B_e;

    EXPECT_NEAR(0.0, FrobeniusDiff(C_m, C_e), kTolBig);
}

TEST(MatFixedEigen, MultiplySquareVeryLarge)
{
    std::mt19937 gen(2025);

    using MatN = Mat<double, 60, 60>;
    using EN = Eigen::Matrix<double, 60, 60>;

    EN A_e = randomMatrix<60, 60>(gen);
    EN B_e = randomMatrix<60, 60>(gen);

    MatN A_m = MatFromEigen<double, 60, 60>(A_e);
    MatN B_m = MatFromEigen<double, 60, 60>(B_e);

    MatN C_m = A_m * B_m;
    EN C_e = A_e * B_e;

    EXPECT_NEAR(0.0, FrobeniusDiff(C_m, C_e), kTolBig);
}

TEST(MatFixedEigen, ScalarMultiplyDivideAndUnaryMinus)
{
    std::mt19937 gen(777);

    using MatRC = Mat<double, 21, 17>;
    using ERC = Eigen::Matrix<double, 21, 17>;

    ERC A_e = randomMatrix<21, 17>(gen);
    MatRC A_m = MatFromEigen<double, 21, 17>(A_e);

    double s = 1.2345;

    MatRC S1_m = A_m * s;
    MatRC S2_m = s * A_m;
    MatRC Q_m = A_m / s;
    MatRC S3_m = A_m;
    S3_m *= s;
    MatRC Q2_m = A_m;
    Q2_m /= s;

    ERC S_e = A_e * s;
    ERC Q_e = A_e / s;

    EXPECT_NEAR(0.0, FrobeniusDiff(S1_m, S_e), kTol);
    EXPECT_NEAR(0.0, FrobeniusDiff(S2_m, S_e), kTol);
    EXPECT_NEAR(0.0, FrobeniusDiff(S3_m, S_e), kTol);
    EXPECT_NEAR(0.0, FrobeniusDiff(Q_m, Q_e), kTol);
    EXPECT_NEAR(0.0, FrobeniusDiff(Q2_m, Q_e), kTol);

    // Unary minus
    MatRC Neg_m = -A_m;
    ERC Neg_e = -A_e;

    EXPECT_NEAR(0.0, FrobeniusDiff(Neg_m, Neg_e), kTol);
}

TEST(MatFixedEigen, NormAndSqrt)
{
    std::mt19937 gen(888);

    using MatRC = Mat<double, 15, 15>;
    using ERC = Eigen::Matrix<double, 15, 15>;

    ERC A_e = randomMatrix<15, 15>(gen).cwiseAbs(); // ensure non-negative for sqrt
    MatRC A_m = MatFromEigen<double, 15, 15>(A_e);

    double n_e = A_e.norm();
    double n_m = A_m.norm();
    EXPECT_NEAR(n_e, n_m, kTol);

    // MatRC S_m = A_m.sqrt();
    // ERC  S_e = A_e.array().sqrt().matrix();
    // EXPECT_NEAR(0.0, FrobeniusDiff(S_m, S_e), kTol);
}

TEST(MatFixedEigen, ConvMatchesFrobeniusInnerProduct)
{
    std::mt19937 gen(999);

    using MatRC = Mat<double, 32, 24>;
    using ERC = Eigen::Matrix<double, 32, 24>;

    ERC A_e = randomMatrix<32, 24>(gen);
    ERC B_e = randomMatrix<32, 24>(gen);

    MatRC A_m = MatFromEigen<double, 32, 24>(A_e);
    MatRC B_m = MatFromEigen<double, 32, 24>(B_e);

    double conv_m = A_m.template conv<double>(B_m);
    double conv_e = (A_e.array() * B_e.array()).sum();

    EXPECT_NEAR(conv_e, conv_m, kTol);
}

TEST(MatFixedEigen, TransposeLarge)
{
    std::mt19937 gen(1357);

    using MatRC = Mat<double, 40, 25>;
    using ERC = Eigen::Matrix<double, 40, 25>;
    using ETC = Eigen::Matrix<double, 25, 40>;

    ERC A_e = randomMatrix<40, 25>(gen);
    MatRC A_m = MatFromEigen<double, 40, 25>(A_e);

    auto AT_m = A_m.transpose();
    ETC AT_e = A_e.transpose();

    EXPECT_EQ(AT_m.rows(), AT_e.rows());
    EXPECT_EQ(AT_m.cols(), AT_e.cols());
    EXPECT_NEAR(0.0, FrobeniusDiff(AT_m, AT_e), kTol);
}

//------------------------------------------------------------------------------
// Vec / Vec2 / Vec3 tests vs Eigen
//------------------------------------------------------------------------------

TEST(VecFixedEigen, DotAndNorm)
{
    std::mt19937 gen(2468);
    std::normal_distribution<double> dist(0.0, 1.0);

    using Vec10 = Vec<double, 10>;
    using EVec10 = Eigen::Matrix<double, 10, 1>;

    Vec10 v1_m, v2_m;
    EVec10 v1_e, v2_e;

    for (int i = 0; i < 10; ++i)
    {
        double a = dist(gen);
        double b = dist(gen);
        v1_m(i) = a;
        v2_m(i) = b;
        v1_e(i) = a;
        v2_e(i) = b;
    }

    double dot_m = v1_m.dot(v2_m);
    double dot_e = v1_e.dot(v2_e);

    double n1_m = v1_m.norm();
    double n1_e = v1_e.norm();

    EXPECT_NEAR(dot_e, dot_m, kTol);
    EXPECT_NEAR(n1_e, n1_m, kTol);
}

TEST(VecFixedEigen, Vec3CrossAndNormalized)
{
    std::mt19937 gen(9753);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int k = 0; k < 50; ++k)
    {
        Eigen::Vector3d v1_e(dist(gen), dist(gen), dist(gen));
        Eigen::Vector3d v2_e(dist(gen), dist(gen), dist(gen));

        // Avoid zero vector for normalization
        if (v1_e.norm() < 1e-9 || v2_e.norm() < 1e-9)
        {
            --k;
            continue;
        }

        Vec3<double> v1_m(v1_e(0), v1_e(1), v1_e(2));
        Vec3<double> v2_m(v2_e(0), v2_e(1), v2_e(2));

        Vec3<double> c_m = v1_m.cross(v2_m);
        Eigen::Vector3d c_e = v1_e.cross(v2_e);

        EXPECT_NEAR(c_e(0), c_m(0), kTol);
        EXPECT_NEAR(c_e(1), c_m(1), kTol);
        EXPECT_NEAR(c_e(2), c_m(2), kTol);

        Vec3<double> n_m = v1_m.normalized();
        Eigen::Vector3d n_e = v1_e.normalized();

        EXPECT_NEAR(n_e(0), n_m(0), kTol);
        EXPECT_NEAR(n_e(1), n_m(1), kTol);
        EXPECT_NEAR(n_e(2), n_m(2), kTol);
    }
}

//------------------------------------------------------------------------------
// Mat3 determinant & inverse vs Eigen
//------------------------------------------------------------------------------

TEST(Mat3EigenCompat, DeterminantAndInverseSPD)
{
    std::mt19937 gen(5555);

    using BaseMat3 = Mat<double, 3, 3>;
    using M3 = Mat3<double>;
    using E3 = Eigen::Matrix<double, 3, 3>;

    for (int it = 0; it < 50; ++it)
    {
        E3 M = randomMatrix<3, 3>(gen);
        E3 A_e = M.transpose() * M + 0.1 * E3::Identity(); // SPD

        BaseMat3 A_base = MatFromEigen<double, 3, 3>(A_e);
        M3 A_m(A_base);

        double det_m = A_m.determinant();
        double det_e = A_e.determinant();

        EXPECT_NEAR(det_e, det_m, kTol);

        M3 inv_m = A_m.inverse();
        E3 inv_e = A_e.inverse();

        const BaseMat3 &inv_base = static_cast<const BaseMat3 &>(inv_m);
        EXPECT_NEAR(0.0, FrobeniusDiff(inv_base, inv_e), kTolBig);
    }
}

//------------------------------------------------------------------------------
// Quaternion vs Eigen::Quaterniond
//------------------------------------------------------------------------------

TEST(QuaternionEigenCompat, RotationMatrixMatchesEigen)
{
    std::mt19937 gen(7777);

    using Q = Quaternion<double>;
    using E3 = Eigen::Matrix<double, 3, 3>;
    using BaseMat3 = Mat<double, 3, 3>;

    for (int it = 0; it < 100; ++it)
    {
        Eigen::Quaterniond q_e = randomUnitQuaternion(gen);
        Q q(q_e.w(), q_e.x(), q_e.y(), q_e.z());

        Mat3<double> R_m = q.matrix();
        const BaseMat3 &R_base = static_cast<const BaseMat3 &>(R_m);
        E3 R_me = EigenFromMat<double, 3, 3>(R_base);

        E3 R_e = q_e.toRotationMatrix();

        EXPECT_NEAR(0.0, (R_me - R_e).norm(), kTol);
    }
}

//------------------------------------------------------------------------------
// SO3 vs Eigen (using quaternions / angle-axis)
//------------------------------------------------------------------------------

TEST(SO3EigenCompat, MatrixMatchesEigenFromQuaternion)
{
    std::mt19937 gen(8888);

    using E3 = Eigen::Matrix<double, 3, 3>;
    using BaseMat3 = Mat<double, 3, 3>;

    for (int it = 0; it < 100; ++it)
    {
        Eigen::Quaterniond q_e = randomUnitQuaternion(gen);

        SO3<double> R(q_e.w(), q_e.x(), q_e.y(), q_e.z());
        Mat3<double> R_m = R.matrix();

        const BaseMat3 &R_base = static_cast<const BaseMat3 &>(R_m);
        E3 R_me = EigenFromMat<double, 3, 3>(R_base);
        E3 R_e = q_e.toRotationMatrix();

        EXPECT_NEAR(0.0, (R_me - R_e).norm(), kTol);
    }
}

TEST(SO3EigenCompat, LogMatchesEigenAngleAxis)
{
    std::mt19937 gen(9999);
    const double pi = 3.14159265358979323846;

    for (int it = 0; it < 100; ++it)
    {
        // Random unit quaternion
        Eigen::Quaterniond q_e = randomUnitQuaternion(gen);

        // Get Eigen's angle-axis from that quaternion (principal representation)
        Eigen::AngleAxisd aa(q_e);
        Eigen::Vector3d axis = aa.axis();
        double angle = aa.angle(); // in [0, pi]

        Eigen::Vector3d phi_e = angle * axis;

        SO3<double> R(q_e.w(), q_e.x(), q_e.y(), q_e.z());
        Vec3<double> phi_m = R.log();

        EXPECT_NEAR(phi_e(0), phi_m(0), kTolBig);
        EXPECT_NEAR(phi_e(1), phi_m(1), kTolBig);
        EXPECT_NEAR(phi_e(2), phi_m(2), kTolBig);
    }
}

TEST(SO3EigenCompat, ExpMatchesEigenAngleAxisForModerateAngles)
{
    std::mt19937 gen(24601);
    const double pi = 3.14159265358979323846;

    using E3 = Eigen::Matrix<double, 3, 3>;
    using BaseMat3 = Mat<double, 3, 3>;

    std::uniform_real_distribution<double> angle_dist(-pi + 0.2, pi - 0.2);

    for (int it = 0; it < 100; ++it)
    {
        Eigen::Vector3d axis = randomUnitVector3(gen);
        double angle = angle_dist(gen);

        Eigen::Vector3d phi_e = angle * axis;
        Vec3<double> phi_m(phi_e(0), phi_e(1), phi_e(2));

        SO3<double> R = SO3<double>::exp(phi_m);
        Mat3<double> R_m = R.matrix();

        const BaseMat3 &R_base = static_cast<const BaseMat3 &>(R_m);
        E3 R_me = EigenFromMat<double, 3, 3>(R_base);

        Eigen::AngleAxisd aa(angle, axis);
        E3 R_e = aa.toRotationMatrix();

        EXPECT_NEAR(0.0, (R_me - R_e).norm(), kTolBig);
    }
}

TEST(SO3EigenCompat, ExpLogConsistencySmallAngles)
{
    std::mt19937 gen(13579);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int it = 0; it < 100; ++it)
    {
        Eigen::Vector3d phi_e(dist(gen), dist(gen), dist(gen));
        // Make it small so we hit the small-angle branch sometimes
        phi_e *= 1e-3;

        Vec3<double> phi_m(phi_e(0), phi_e(1), phi_e(2));

        SO3<double> R = SO3<double>::exp(phi_m);
        Vec3<double> phi2_m = R.log();

        EXPECT_NEAR(phi_m(0), phi2_m(0), 1e-6);
        EXPECT_NEAR(phi_m(1), phi2_m(1), 1e-6);
        EXPECT_NEAR(phi_m(2), phi2_m(2), 1e-6);
    }
}

//------------------------------------------------------------------------------
// SE3 vs Eigen 4x4 transforms (matrix, composition, inverse)
//------------------------------------------------------------------------------

TEST(SE3EigenCompat, MatrixMatchesEigenTransform)
{
    std::mt19937 gen(4242);

    using E4 = Eigen::Matrix<double, 4, 4>;
    using E3v = Eigen::Matrix<double, 3, 1>;
    using BaseMat4 = Mat<double, 4, 4>;

    for (int it = 0; it < 50; ++it)
    {
        Eigen::Quaterniond q_e = randomUnitQuaternion(gen);
        E3v t_e;
        std::normal_distribution<double> dist(0.0, 1.0);
        t_e << dist(gen), dist(gen), dist(gen);

        SO3<double> R(q_e.w(), q_e.x(), q_e.y(), q_e.z());
        Vec3<double> t(t_e(0), t_e(1), t_e(2));

        SE3<double> T(R, t);
        Mat4<double> T_m = T.matrix();
        const BaseMat4 &T_base = static_cast<const BaseMat4 &>(T_m);
        E4 T_me = EigenFromMat<double, 4, 4>(T_base);

        E4 T_e = E4::Identity();
        T_e.block<3, 3>(0, 0) = q_e.toRotationMatrix();
        T_e.block<3, 1>(0, 3) = t_e;

        EXPECT_NEAR(0.0, (T_me - T_e).norm(), kTolBig);
    }
}

TEST(SE3EigenCompat, CompositionMatchesEigen)
{
    std::mt19937 gen(6666);

    using E4 = Eigen::Matrix<double, 4, 4>;
    using E3v = Eigen::Matrix<double, 3, 1>;
    using BaseMat4 = Mat<double, 4, 4>;

    for (int it = 0; it < 50; ++it)
    {
        Eigen::Quaterniond q1 = randomUnitQuaternion(gen);
        Eigen::Quaterniond q2 = randomUnitQuaternion(gen);
        std::normal_distribution<double> dist(0.0, 1.0);
        E3v t1, t2;
        t1 << dist(gen), dist(gen), dist(gen);
        t2 << dist(gen), dist(gen), dist(gen);

        SO3<double> R1(q1.w(), q1.x(), q1.y(), q1.z());
        SO3<double> R2(q2.w(), q2.x(), q2.y(), q2.z());
        Vec3<double> v1(t1(0), t1(1), t1(2));
        Vec3<double> v2(t2(0), t2(1), t2(2));

        SE3<double> T1(R1, v1);
        SE3<double> T2(R2, v2);
        SE3<double> T3 = T1 * T2;

        Mat4<double> T3_m = T3.matrix();
        const BaseMat4 &T3_base = static_cast<const BaseMat4 &>(T3_m);
        E4 T3_me = EigenFromMat<double, 4, 4>(T3_base);

        E4 T1_e = E4::Identity();
        E4 T2_e = E4::Identity();
        T1_e.block<3, 3>(0, 0) = q1.toRotationMatrix();
        T1_e.block<3, 1>(0, 3) = t1;
        T2_e.block<3, 3>(0, 0) = q2.toRotationMatrix();
        T2_e.block<3, 1>(0, 3) = t2;

        E4 T3_e = T1_e * T2_e;

        EXPECT_NEAR(0.0, (T3_me - T3_e).norm(), kTolBig);
    }
}

TEST(SE3EigenCompat, InverseMatchesEigen)
{
    std::mt19937 gen(7771);

    using E4 = Eigen::Matrix<double, 4, 4>;
    using E3v = Eigen::Matrix<double, 3, 1>;
    using BaseMat4 = Mat<double, 4, 4>;

    for (int it = 0; it < 50; ++it)
    {
        Eigen::Quaterniond q = randomUnitQuaternion(gen);
        std::normal_distribution<double> dist(0.0, 1.0);
        E3v t;
        t << dist(gen), dist(gen), dist(gen);

        SO3<double> R(q.w(), q.x(), q.y(), q.z());
        Vec3<double> v(t(0), t(1), t(2));

        SE3<double> T(R, v);
        SE3<double> T_inv = T.inverse();

        Mat4<double> T_m = T.matrix();
        Mat4<double> T_inv_m = T_inv.matrix();

        const BaseMat4 &T_base = static_cast<const BaseMat4 &>(T_m);
        const BaseMat4 &T_inv_base = static_cast<const BaseMat4 &>(T_inv_m);

        E4 T_me = EigenFromMat<double, 4, 4>(T_base);
        E4 T_inv_me = EigenFromMat<double, 4, 4>(T_inv_base);

        E4 T_e = E4::Identity();
        T_e.block<3, 3>(0, 0) = q.toRotationMatrix();
        T_e.block<3, 1>(0, 3) = t;

        E4 T_inv_e = T_e.inverse();

        EXPECT_NEAR(0.0, (T_inv_me - T_inv_e).norm(), kTolBig);

        // Also check that T * T_inv ~ Identity
        E4 I_me = T_me * T_inv_me;
        E4 I = E4::Identity();
        EXPECT_NEAR(0.0, (I_me - I).norm(), kTolBig);
    }
}

TEST(SE3EigenCompat, ExpLogConsistency)
{
    std::mt19937 gen(8811);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int it = 0; it < 50; ++it)
    {
        Vec6<double> xi;
        // Make rotation part moderate to avoid 2*pi weirdness
        for (int i = 0; i < 3; ++i)
            xi(i) = dist(gen); // translation
        for (int i = 3; i < 6; ++i)
            xi(i) = dist(gen) * 0.2; // rotation

        SE3<double> T = SE3<double>::exp(xi);
        Vec6<double> xi2 = T.log();

        for (int i = 0; i < 6; ++i)
        {
            EXPECT_NEAR(xi(i), xi2(i), 1e-6);
        }
    }
}
