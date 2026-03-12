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
    template <typename T>
    class LDLTx
    {
    public:

        // LDLTx()
        //     : LDLTx(0)
        //{
        // }

        LDLTx(int size)
            : size_(size), A_(size, size), L_(size, size), D_(size), invD_(size), y_(size), x_(size), z_(size)
        {
        }

        // Compute the LDLT decomposition of a matrix A.
        // This must be called before solve().
        void compute(const Matx<T> &A)
        {
            A_ = A;
            ldlt_decompose();
        }

        // Solve A x = b for x, given b.
        // Assumes compute() has been called.
        Vecx<T> solve(const Vecx<T> &b)
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
                    sum -= L_(i, k) * L_(i, k) * D_(k);
                }
                D_(i) = sum;

                if (math::fabs(D_(i)) < 1e-12)
                {
                    // Matrix is not positive definite or is singular.
                    // A robust implementation would throw or return an error.
                }

                // 2) Compute L[j][i] for j = i+1..n-1
                invD_(i) = 1 / D_(i);
                for (int j = i + 1; j < size_; ++j)
                {
                    auto val = A_(j, i);
                    // Subtract the part contributed by previous columns
                    for (int k = 0; k < i; ++k)
                    {
                        val -= L_(j, k) * L_(i, k) * D_(k);
                    }
                    // L[j][i] = (A[j][i] - ...) / D[i]
                    L_(j, i) = val * invD_(i);
                }
            }
        }

        // Forward substitution for L y = b
        // L is lower triangular with diagonal = 1.0
        void forward_substitution(const Matx<T> &L, const Vecx<T> &b, Vecx<T> &y)
        {
            for (int i = 0; i < size_; ++i)
            {
                auto sum = b(i);
                for (int j = 0; j < i; ++j)
                {
                    sum -= L(i, j) * y(j);
                }
                // L(i, i) is 1.0
                y(i) = sum;
            }
        }

        // Diagonal solve for D z = y
        // D is diagonal, stored as a vector. z[i] = y[i] / D[i]
        void diagonal_solve(const Vecx<T> &D, const Vecx<T> &y, Vecx<T> &z)
        {
            for (int i = 0; i < size_; ++i)
            {
                z(i) = y(i) / D(i);
            }
        }

        // Back substitution for L^T x = z
        // L is lower-triangular, so L^T is upper-triangular.
        void back_substitution_transpose(const Matx<T> &L, const Vecx<T> &z, Vecx<T> &x)
        {
            for (int i = size_ - 1; i >= 0; --i)
            {
                auto sum = z(i);
                for (int j = i + 1; j < size_; ++j)
                {
                    sum -= L(j, i) * x(j); // L^T[i][j] = L[j][i]
                }
                // L[i][i] = 1.0
                x(i) = sum;
            }
        }

        Matx<T> A_;
        Matx<T> L_;
        Vecx<T> D_;    // Store diagonal of D as a vector
        Vecx<T> invD_; // Store inverse of diagonal of D as a vector

        Vecx<T> y_;
        Vecx<T> x_;
        Vecx<T> z_;

        int size_;
    };
}
