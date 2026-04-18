#pragma once

#include "linalg/linalgx.h"

#ifdef USE_VITIS
#include "hls_math.h"
namespace math = hls;
#else
#include <cmath>
#include <type_traits>
namespace math = std;
#endif

namespace linalg
{
    enum class Status
    {
        Success = 0,
        DimensionMismatch,
        NotSquare,
        SingularPivot
    };

    // In-place factorization:
    //   A = L * D * L^T
    //
    // Storage layout after compute():
    //   - A(i,i) stores D(i)
    //   - A(j,i) for j>i stores L(j,i)
    //   - diagonal of L is implicit 1
    //
    // The upper triangle is ignored after factorization.
    template <typename T>
    Status ldlt_factorize(MatView<T> A)
    {
#pragma HLS inline off

        if (A.rows() != A.cols())
        {
            return Status::NotSquare;
        }

        const int n = A.rows();

    fac_out_loop:
        for (int i = 0; i < n; ++i)
        {
            // Compute D(i)
            T d = A(i, i);

        fac_diag_loop:
            for (int k = 0; k < i; ++k)
            {
                const T Lik = A(i, k);
                d -= Lik * Lik * A(k, k);
            }

            if (abs(d) <= static_cast<T>(1e-9))
            {
                return Status::SingularPivot;
            }

            A(i, i) = d;
            const T inv_d = T(1) / d;

        // Compute L(j,i), j > i
        fac_low_out_loop:
            for (int j = i + 1; j < n; ++j)
            {
                T lij = A(j, i);

            fac_low_in_loop:
                for (int k = 0; k < i; ++k)
                {
                    lij -= A(j, k) * A(i, k) * A(k, k);
                }
                A(j, i) = lij * inv_d;
            }
        }

        return Status::Success;
    }

    // In-place solve: b := x
    template <typename T, int max_n = 16>
    Status ldlt_solve(const MatView<T> A, VecView<T> b)
    {
#pragma HLS inline off

        if (A.rows() != A.cols())
        {
            return Status::NotSquare;
        }
        if (b.size() != A.rows())
            return Status::DimensionMismatch;

        const int n = b.size();
        if (n == 0)
            return Status::Success;

        T b_buf[max_n];

        for (int i = 0; i < max_n; i++)
        {
#pragma HLS unroll off

            if (i < b.size())
                b_buf[i] = b(i);
        }

    // Forward solve: L y = b
    solve_forw_out_loop:
        // for (int i = 0; i < n; ++i)
        for (int i = 0; i < max_n; ++i)
        {
// #pragma HLS loop_tripcount min = 6 max = 64
#pragma HLS unroll off
            // #pragma HLS loop_flatten

            if (i >= n)
                continue;

            // T sum = b(i);
            T sum = T(0);
        solve_forw_in_loop:
            // for (int j = 0; j < i; ++j)
            for (int j = 0; j < max_n; ++j)
            {
                // #pragma HLS loop_tripcount min = 6 max = 64

                if (j >= i)
                    continue;

                sum -= A(i, j) * b_buf[j];
                // sum -= A(i, j) * b(j);
            }
            // b(i) = sum;
            // b(i) += sum;
            b_buf[i] += sum;
        }

        // Diagonal solve: D z = y

    solve_diag_loop:
        for (int i = 0; i < n; ++i)
        {
            const T d = A(i, i);
            if (abs(d) <= static_cast<T>(1e-9))
                return Status::SingularPivot;
            // b(i) /= d;
            b_buf[i] = b_buf[i] / d;
        }

    // Backward solve: L^T x = z
    solve_back_out_loop:
        for (int ii = n - 1; ii >= 0; --ii)
        {
            T sum = b_buf[ii];

        solve_back_in_loop:
            for (int j = ii + 1; j < n; ++j)
            {
                sum -= A(j, ii) * b_buf[j];
            }
            b_buf[ii] = sum;
        }

        for (int i = 0; i < max_n; i++)
        {
            if (i < b.size())
                b(i) = b_buf[i];
        }

        return Status::Success;
    }
}