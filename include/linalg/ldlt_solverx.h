#pragma once

#include "linalg/linalgx.h"
#include "linalg/ldlt_solver_view_inplace.h"

namespace linalg
{
    template <typename T>
    class LDLTx
    {
    public:
        LDLTx(int n) : A_(n, n), b_(n) {}

        // Compute the LDLT decomposition of a matrix A.
        // This must be called before solve().
        void compute(const Matx<T> &A)
        {
            A_ = A;
            MatView<T> view(A_.data(), A_.rows(), A_.cols());
            ldlt_factorize(view);
        }

        // Solve A x = b for x, given b.
        // Assumes compute() has been called.
        Vecx<T> solve(const Vecx<T> &b)
        {
            b_ = b;
            MatView<T> mview(A_.data(), A_.rows(), A_.cols());
            VecView<T> vview(b_.data(), b_.size());
            ldlt_solve(mview, vview);
            return b_;
        }

    private:
        Matx<T> A_;
        Vecx<T> b_;
    };
}
