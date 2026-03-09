#pragma once

#include "linalg/linalg.h"
#include "linalg/linalgx.h"

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

// Build numeric Hvv_values matching *full-lower* pattern directly from S.H block.
template <typename InMatType, typename OutMatType>
static OutMatType from_dense_to_sparse(const InMatType &S)
{
    OutMatType A(S.rows(), S.cols());
    for (int i = 0; i < S.rows(); ++i)
        for (int j = 0; j < S.cols(); ++j)
            if (S(i, j) != 0.0)
                A.insert(i, j) = S(i, j);
    // A.insert(i, j, S(i, j));
    A.makeCompressed();
    return A;
}

// Build SPD system H = J^T J + lambda I, g = J^T r
template <typename T>
struct SPDSystem
{
    int n_mesh = 0;
    int n_pose = 0;
    int n = 0;

    linalg::Matx<T> H; // n x n
    linalg::Vecx<T> g; // n

    linalg::Matx<T> Hvp; // n_mesh x n_pose
    linalg::Matx<T> Hpp; // n_pose x n_pose
    linalg::Matx<T> Hvv; // n_pose x n_pose
    linalg::Vecx<T> gv;  // n_mesh
    linalg::Vecx<T> gp;  // n_pose
};

// Generates a random SPD normal-equation-like system
template <typename T>
static SPDSystem<T> make_system(int w_mesh, int h_mesh, int s_pose, int dof_pose, T lambda, uint32_t seed)
{
    SPDSystem<T> S;
    S.n_mesh = w_mesh * h_mesh;
    S.n_pose = s_pose * dof_pose;
    S.n = S.n_mesh + S.n_pose;

    std::mt19937 rng(seed);
    std::normal_distribution<T> nd(0.0, 1.0);

    int n_tris = (w_mesh - 1) * (h_mesh - 1) * 2;

    linalg::Matx<T> J = linalg::Matx<T>::Zero(n_tris, S.n);

    for (int i = 0; i < n_tris; ++i)
    {
        for (int y = 0; y < h_mesh; ++y)
        {
            for (int x = 0; x < w_mesh; ++x)
            {
                if (x > 0 && y > 0)
                {
                    int n_0 = w_mesh * y + x;
                    int n_1 = w_mesh * y + x - 1;
                    int n_2 = w_mesh * (y - 1) + x;
                    int n_3 = w_mesh * (y - 1) + x - 1;

                    J(i, n_0) = nd(rng);
                    J(i, n_1) = nd(rng);
                    J(i, n_2) = nd(rng);
                    J(i, n_3) = nd(rng);
                }
            }
        }
    }

    for (int i = 0; i < n_tris; ++i)
    {
        for (int j = 0; j < S.n_pose; ++j)
        {
            J(i, j + S.n_mesh) = nd(rng);
        }
    }

    linalg::Vecx<T> r(n_tris);
    for (int i = 0; i < n_tris; ++i)
        r(i) = nd(rng);

    S.H = linalg::Matx<T>(S.n, S.n);

    for (int i = 0; i < S.n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            T sum = 0.0;
            for (int k = 0; k < n_tris; ++k)
                sum += J(k, i) * J(k, j);

            S.H(i, j) = sum;
            S.H(j, i) = sum;
        }
        S.H(i, i) += lambda;
    }

    S.g = linalg::Vecx<T>(S.n);
    for (int j = 0; j < S.n; ++j)
    {
        T sum = 0.0;
        for (int k = 0; k < n_tris; ++k)
            sum += J(k, j) * r(k);
        S.g(j) = sum;
    }

    S.gv = linalg::Vecx<T>(S.n_mesh);
    S.gp = linalg::Vecx<T>(S.n_pose);
    for (int i = 0; i < S.n_mesh; ++i)
        S.gv(i) = S.g(i);
    for (int i = 0; i < S.n_pose; ++i)
        S.gp(i) = S.g(S.n_mesh + i);

    S.Hvv = linalg::Matx<T>(S.n_mesh, S.n_mesh);
    for (int vc = 0; vc < S.n_mesh; ++vc)
        for (int vr = 0; vr < S.n_mesh; ++vr)
            S.Hvv(vr, vc) = S.H(vr, vc);

    S.Hvp = linalg::Matx<T>(S.n_mesh, S.n_pose);
    for (int pc = 0; pc < S.n_pose; ++pc)
        for (int vr = 0; vr < S.n_mesh; ++vr)
            S.Hvp(vr, pc) = S.H(vr, S.n_mesh + pc);

    S.Hpp = linalg::Matx<T>(S.n_pose, S.n_pose);
    for (int c = 0; c < S.n_pose; ++c)
        for (int r2 = 0; r2 < S.n_pose; ++r2)
            S.Hpp(r2, c) = S.H(S.n_mesh + r2, S.n_mesh + c);

    return S;
}