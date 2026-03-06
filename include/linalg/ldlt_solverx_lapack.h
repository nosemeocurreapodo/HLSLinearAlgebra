#pragma once
#include <vector>
#include <stdexcept>
#include <lapacke.h>

#include "linalg/linalgx.h"

namespace linalg
{
    template <typename Type>
    class LDLT_LAPACK
    {
    public:
        LDLT_LAPACK() : LDLT_LAPACK(0) {}
        explicit LDLT_LAPACK(int n) : n_(n), a_(n * n), ipiv_(n) {}

        // Factorize A (symmetric). Only one triangle is referenced; we use lower ('L').
        void compute(const Matx<Type> &A)
        {
            // Pack Matx -> column-major buffer a_
            // a_(r,c) in col-major is a_[r + c*n_]
            for (int r = 0; r < n_; ++r)
                for (int c = 0; c < n_; ++c)
                    a_[r + c * n_] = A(r, c);

            // Factorization in-place
            // A = L*D*L^T with pivoting (Bunch–Kaufman)
            lapack_int info = LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L',
                                             (lapack_int)n_, a_.data(), (lapack_int)n_,
                                             ipiv_.data());
            if (info < 0)
                throw std::runtime_error("LAPACKE_dsytrf: illegal argument at position " + std::to_string(-info));
            if (info > 0)
                throw std::runtime_error("LAPACKE_dsytrf: factorization failed (singular pivot/block) at k=" + std::to_string(info));
        }

        Vecx<Type> solve(const Vecx<Type> &b) const
        {
            Vecx<Type> x = b; // LAPACK overwrites RHS with solution

            lapack_int nrhs = 1;
            lapack_int info = LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L',
                                             (lapack_int)n_, nrhs,
                                             a_.data(), (lapack_int)n_,
                                             ipiv_.data(),
                                             x.data(), (lapack_int)n_);
            if (info < 0)
                throw std::runtime_error("LAPACKE_dsytrs: illegal argument at position " + std::to_string(-info));

            return x;
        }

    private:
        int n_;
        std::vector<Type> a_;          // factorized matrix (in-place)
        std::vector<lapack_int> ipiv_; // pivots from SYTRF
    };

} // namespace linalg
