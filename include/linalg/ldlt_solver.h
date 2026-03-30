#pragma once

#include "linalg/linalg.h"

namespace linalg
{
    template <typename Type, int size>
    class LDLT
    {
    public:
        using MatN = Mat<Type, size, size>;
        using VecN = Mat<Type, size, 1>;

        LDLT() {}

        // Compute the LDLT decomposition of a matrix A.
        // This must be called before solve().
        void compute(const MatN &A)
        {
            A_ = A;
            ldlt_decompose();
        }

        // Solve A x = b for x, given b.
        // Assumes compute() has been called.
        VecN solve(const VecN &b)
        {
            // 1) Solve L y = b (Forward substitution)
            forward_substitution(L_, b, y_);

            // 3) Solve D z = y
            diagonal_solve(D_, y_, z_);

            // 4) Solve L^T x = z
            back_substitution_transpose(L_, z_, x_);

            return x_;
        }

    private:
        void ldlt_decompose()
        {
            // Initialize L to identity and D to zero.
            L_.setIdentity();
            D_.setZero();

            for (int i = 0; i < size; ++i)
            {
                // 1) Compute D[i] = A[i][i] - sum_{k=0 to i-1}(L[i][k]^2 * D[k])
                Type sum = A_(i, i);
                for (int k = 0; k < i; ++k)
                {
                    sum -= L_(i, k) * L_(i, k) * D_(k, 0);
                }
                D_(i, 0) = sum;

                if (fabs(D_(i, 0)) < Type(1e-12))
                {
                    // Matrix is not positive definite or is singular.
                    // A robust implementation would throw or return an error.
                }

                // 2) Compute L[j][i] for j = i+1..n-1
                Type inv_Di = Type(1) / D_(i, 0);
                for (int j = i + 1; j < size; ++j)
                {
                    Type val = A_(j, i);
                    // Subtract the part contributed by previous columns
                    for (int k = 0; k < i; ++k)
                    {
                        val -= L_(j, k) * L_(i, k) * D_(k, 0);
                    }
                    // L[j][i] = (A[j][i] - ...) / D[i]
                    L_(j, i) = val * inv_Di;
                }
            }
        }

        // Forward substitution for L y = b
        // L is lower triangular with diagonal = 1.0
        void forward_substitution(const MatN &L, const VecN &b, VecN &y)
        {
            for (int i = 0; i < size; ++i)
            {
                Type sum = b(i, 0);
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
        void diagonal_solve(const VecN &D, const VecN &y, VecN &z)
        {
            for (int i = 0; i < size; ++i)
            {
                z(i, 0) = y(i, 0) / D(i, 0);
            }
        }

        // Back substitution for L^T x = z
        // L is lower-triangular, so L^T is upper-triangular.
        void back_substitution_transpose(const MatN &L, const VecN &z, VecN &x)
        {
            for (int i = size - 1; i >= 0; --i)
            {
                Type sum = z(i, 0);
                for (int j = i + 1; j < size; ++j)
                {
                    sum -= L(j, i) * x(j, 0); // L^T[i][j] = L[j][i]
                }
                // L[i][i] = 1.0
                x(i, 0) = sum;
            }
        }

        MatN A_;
        MatN L_;
        VecN D_; // Store diagonal of D as a vector

        VecN y_;
        VecN x_;
        VecN z_;
    };
}
