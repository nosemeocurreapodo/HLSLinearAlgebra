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
    template <typename T, int max_n>
    Status ldlt_factorize(MatView<T> A)
    {
#pragma HLS inline off

        if (A.rows() != A.cols())
            return Status::NotSquare;

        if (A.rows() != max_n)
            return Status::DimensionMismatch;

        const int n = A.rows();

        T A1_buf[max_n];
#pragma HLS BIND_STORAGE variable = A1_buf type = ram_1p

        T A2_buf[max_n];
#pragma HLS BIND_STORAGE variable = A2_buf type = ram_1p

        T D_buf[max_n];
#pragma HLS BIND_STORAGE variable = D_buf type = ram_1p

        for (int j = 0; j < n; ++j)
        {
            D_buf[j] = A(j, j);
        }

    fac_out_loop:
        for (int i = 0; i < n; ++i)
        // for (int i = 0; i < max_n; ++i)
        {
#pragma HLS loop_tripcount min = max_n max = max_n

            if (i >= n)
                continue;

            for (int j = 0; j < n; ++j)
            {
                A1_buf[j] = A(i, j);
                A2_buf[j] = A(j, i);
            }

            // Compute D(i)
            T d = A(i, i);

        fac_diag_loop:
            for (int k = 0; k < i; ++k)
            // for (int k = 0; k < max_n; ++k)
            {
#pragma HLS loop_tripcount min = max_n max = max_n

                if (k >= i)
                    continue;

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
            // for (int j = i + 1; j < max_n; ++j)
            {
#pragma HLS loop_tripcount min = max_n max = max_n

                if (j >= n)
                    continue;

                T lij = A(j, i);

            fac_low_in_loop:
                for (int k = 0; k < i; ++k)
                // for (int k = 0; k < max_n; ++k)
                {
#pragma HLS loop_tripcount min = max_n max = max_n

                    if (k >= i)
                        continue;

                    lij -= A(j, k) * A(i, k) * A(k, k);
                }
                A(j, i) = lij * inv_d;
            }
        }

        return Status::Success;
    }

    // In-place solve: b := x
    template <typename T, int max_n>
    Status ldlt_solve(const MatView<T> A, VecView<T> b)
    {
#pragma HLS inline off

        if (A.rows() != A.cols())
            return Status::NotSquare;
        if (b.size() != A.rows())
            return Status::DimensionMismatch;
        if (b.size() != max_n)
            return Status::DimensionMismatch;

        const int n = b.size();
        // if (n == 0)
        //     return Status::Success;

        T b_buf[max_n];
#pragma HLS BIND_STORAGE variable = b_buf type = ram_1p

        T A_buf[max_n];
#pragma HLS BIND_STORAGE variable = A_buf type = ram_1p

        T D_buf[max_n];
#pragma HLS BIND_STORAGE variable = D_buf type = ram_1p

    solve_b_in:
        for (int i = 0; i < n; i++)
        // for (int i = 0; i < max_n; i++)
        {
#pragma HLS loop_tripcount min = max_n max = max_n
            // #pragma HLS inline off

            // #pragma HLS unroll off
            // #pragma HLS pipeline II = 1

            // if (i >= n)
            //     continue;

            b_buf[i] = b(i);
            D_buf[i] = A(i, i);
        }

    // Forward solve: L y = b
    solve_forw_out_loop:
        for (int i = 0; i < n; ++i)
        // for (int i = 0; i < max_n; ++i)
        {
#pragma HLS loop_tripcount min = max_n max = max_n

            // #pragma HLS unroll off
            //  #pragma HLS loop_flatten

            // if (i >= n)
            //     continue;

        solve_forw_A_loop:
            for (int j = 0; j < n; ++j)
            // for (int j = 0; j < max_n; ++j)
            {
#pragma HLS loop_tripcount min = max_n max = max_n

                // if (j >= n)
                //     continue;

                A_buf[j] = A(i, j);
            }

            T sum = b_buf[i];
            // T sum = T(0);
        solve_forw_in_loop:
            for (int j = 0; j < i; ++j)
            // for (int j = 0; j < max_n; ++j)
            {
#pragma HLS loop_tripcount min = max_n max = max_n

                if (j >= i)
                    continue;

                sum -= A_buf[j] * b_buf[j];
                // sum -= A(i, j) * b_buf[j];
            }
            b_buf[i] = sum;
            // b(i) += sum;
            // b_buf[i] += sum;
        }

        // Diagonal solve: D z = y

    solve_diag_loop:
        for (int i = 0; i < n; ++i)
        // for (int i = 0; i < max_n; ++i)
        {
#pragma HLS loop_tripcount min = max_n max = max_n

            // if (i >= n)
            //     continue;

            // const T d = A(i, i);
            const T d = D_buf[i];
            if (abs(d) <= static_cast<T>(1e-9))
                return Status::SingularPivot;
            // b(i) /= d;
            b_buf[i] = b_buf[i] / d;
        }

    // Backward solve: L^T x = z
    solve_back_out_loop:
        for (int ii = n - 1; ii >= 0; --ii)
        // for (int ii = max_n - 1; ii >= 0; --ii)
        {
#pragma HLS loop_tripcount min = max_n max = max_n

            // if (ii >= n)
            //     continue;

        solve_back_A_loop:
            for (int j = 0; j < n; ++j)
            // for (int j = 0; j < max_n; ++j)
            {
#pragma HLS loop_tripcount min = max_n max = max_n

                // if (j >= n)
                //     continue;

                A_buf[j] = A(j, ii);
            }

            T sum = b_buf[ii];

        solve_back_in_loop:
            for (int j = ii + 1; j < n; ++j)
            // for (int j = ii + 1; j < max_n; ++j)
            // for (int j = 0; j < max_n; ++j)
            {
#pragma HLS loop_tripcount min = max_n max = max_n

                if (j < ii + 1) // || j >= n)
                    continue;

                // sum -= A(j, ii) * b_buf[j];
                sum -= A_buf[j] * b_buf[j];
                //  b_buf[ii] -= A(j, ii) * b_buf[j];
            }
            b_buf[ii] = sum;
        }

    solve_b_out:
        for (int i = 0; i < n; i++)
        // for (int i = 0; i < max_n; i++)
        {
#pragma HLS loop_tripcount min = max_n max = max_n

            // if (i >= n)
            //     continue;

            b(i) = b_buf[i];
        }

        return Status::Success;
    }
}