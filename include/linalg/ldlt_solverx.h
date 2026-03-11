#pragma once

#include "linalg/linalgx.h"

#ifdef USE_VITIS
// #define HLS_PRAGMA(x) _Pragma(#x)
// #define HLS_INLINE HLS_PRAGMA(HLS INLINE)
// #define HLS_UNROLL HLS_PRAGMA(HLS UNROLL)
// #define HLS_PIPELINE HLS_PRAGMA(HLS PIPELINE)
// #define HLS_ARRAY_PARTITION(var, type, dim) HLS_PRAGMA(HLS ARRAY_PARTITION variable = var type = type dim = dim)
#include "hls_math.h"
namespace math = hls;
#else
// #define HLS_PRAGMA(x)
// #define HLS_INLINE
// #define HLS_UNROLL
// #define HLS_PIPELINE
// #define HLS_ARRAY_PARTITION(var, type, dim)
#include <cmath>
namespace math = std;
#endif

namespace linalg
{
    template <typename Mat>
    class LDLTx
    {
    public:
        // LDLTx()
        //     : LDLTx(0)
        //{
        // }

        LDLTx(int size)
            : size_(size)
        {
        }

        // Compute the LDLT decomposition of a matrix A.
        // This must be called before solve().
        void compute(const Mat &A)
        {
            A_ = A;
            ldlt_decompose();
        }

        // Solve A x = b for x, given b.
        // Assumes compute() has been called.
        Mat &solve(const Mat &b)
        {
            // 1) Solve L y = b (Forward substitution)
            forward_substitution(L_, b, y_);

            // 3) Solve D z = y
            diagonal_solve(D_, y_, z_);

            // 4) Solve L^T x = z
            back_substitution_transpose(L_, z_, x_);

            return x_;
        }

        /*
        Vecx<Type> solve(const Vecx<Type> &b)
        {
            Vecx<Type> x = b; // reuse as workspace

            // Forward: L y = b  (x becomes y)
            for (int i = 0; i < size_; ++i)
            {
                Type sum = x(i);
                for (int j = 0; j < i; ++j)
                    sum -= L_(i, j) * x(j);
                x(i) = sum;
            }

            // Diagonal: D z = y  (x becomes z)
            for (int i = 0; i < size_; ++i)
                x(i) *= invD_(i); // precompute invD_

            // Backward: L^T x = z (x becomes solution)
            for (int i = size_ - 1; i >= 0; --i)
            {
                Type sum = x(i);
                for (int j = i + 1; j < size_; ++j)
                    sum -= L_(j, i) * x(j);
                x(i) = sum;
            }

            return x;
        }
        */
    private:
        void ldlt_decompose()
        {
            // Initialize L to identity and D to zero.
            L_.setIdentity();
            // D_.setZero();
            // invD_.setZero();

            for (int i = 0; i < size_; ++i)
            {
                // 1) Compute D[i] = A[i][i] - sum_{k=0 to i-1}(L[i][k]^2 * D[k])
                auto sum = A_(i, i);
                for (int k = 0; k < i; ++k)
                {
                    sum -= L_(i, k) * L_(i, k) * D_(k, 0);
                }
                D_(i, 0) = sum;

                if (math::fabs(D_(i, 0)) < 1e-12)
                {
                    // Matrix is not positive definite or is singular.
                    // A robust implementation would throw or return an error.
                }

                // 2) Compute L[j][i] for j = i+1..n-1
                invD_(i, 0) = 1 / D_(i, 0);
                for (int j = i + 1; j < size_; ++j)
                {
                    auto val = A_(j, i);
                    // Subtract the part contributed by previous columns
                    for (int k = 0; k < i; ++k)
                    {
                        val -= L_(j, k) * L_(i, k) * D_(k, 0);
                    }
                    // L[j][i] = (A[j][i] - ...) / D[i]
                    L_(j, i) = val * invD_(i, 0);
                }
            }
        }

        // Forward substitution for L y = b
        // L is lower triangular with diagonal = 1.0
        void forward_substitution(const Mat &L, const Mat &b, Mat &y)
        {
            for (int i = 0; i < size_; ++i)
            {
                auto sum = b(i, 0);
                for (int j = 0; j < i; ++j)
                {
                    sum -= L(i, j) * y(j, 0);
                }
                // L(i, i) is 1.0
                y(i, 0) = sum;
            }
        }

        // Diagonal solve for D z = y
        // D is diagonal, stored as a vector. z[i] = y[i] / D[i]
        void diagonal_solve(const Mat &D, const Mat &y, Mat &z)
        {
            for (int i = 0; i < size_; ++i)
            {
                z(i, 0) = y(i, 0) / D(i, 0);
            }
        }

        // Back substitution for L^T x = z
        // L is lower-triangular, so L^T is upper-triangular.
        void back_substitution_transpose(const Mat &L, const Mat &z, Mat &x)
        {
            for (int i = size_ - 1; i >= 0; --i)
            {
                auto sum = z(i, 0);
                for (int j = i + 1; j < size_; ++j)
                {
                    sum -= L(j, i) * x(j, 0); // L^T[i][j] = L[j][i]
                }
                // L[i][i] = 1.0
                x(i, 0) = sum;
            }
        }

        Mat A_;
        Mat L_;
        Mat D_;    // Store diagonal of D as a vector
        Mat invD_; // Store inverse of diagonal of D as a vector

        Mat y_;
        Mat x_;
        Mat z_;

        int size_;
    };
}
