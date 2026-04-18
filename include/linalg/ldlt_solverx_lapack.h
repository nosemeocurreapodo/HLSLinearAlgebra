#pragma once
#include <vector>
#include <stdexcept>
#include <lapacke.h>

#include "linalg/linalgx.h"

namespace linalg
{
    template <typename T>
    struct LapackLDLT;

    template <>
    struct LapackLDLT<float>
    {
        static lapack_int sytrf(int n, float *a, int lda, lapack_int *ipiv)
        {
            return LAPACKE_ssytrf(LAPACK_COL_MAJOR, 'L',
                                  static_cast<lapack_int>(n),
                                  a,
                                  static_cast<lapack_int>(lda),
                                  ipiv);
        }

        static lapack_int sytrs(int n, int nrhs, const float *a, int lda,
                                const lapack_int *ipiv, float *b, int ldb)
        {
            return LAPACKE_ssytrs(LAPACK_COL_MAJOR, 'L',
                                  static_cast<lapack_int>(n),
                                  static_cast<lapack_int>(nrhs),
                                  a,
                                  static_cast<lapack_int>(lda),
                                  ipiv,
                                  b,
                                  static_cast<lapack_int>(ldb));
        }
    };

    template <>
    struct LapackLDLT<double>
    {
        static lapack_int sytrf(int n, double *a, int lda, lapack_int *ipiv)
        {
            return LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L',
                                  static_cast<lapack_int>(n),
                                  a,
                                  static_cast<lapack_int>(lda),
                                  ipiv);
        }

        static lapack_int sytrs(int n, int nrhs, const double *a, int lda,
                                const lapack_int *ipiv, double *b, int ldb)
        {
            return LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L',
                                  static_cast<lapack_int>(n),
                                  static_cast<lapack_int>(nrhs),
                                  a,
                                  static_cast<lapack_int>(lda),
                                  ipiv,
                                  b,
                                  static_cast<lapack_int>(ldb));
        }
    };

    template <typename T>
    inline void check_supported_type()
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "LDLT_LAPACK only supports float and double");
    }

    template <typename Type>
    class LDLT_LAPACK
    {
    public:
        // LDLT_LAPACK() : LDLT_LAPACK(0) {}
        explicit LDLT_LAPACK(int n) : n_(n), ipiv_(n), a_(n, n)
        {
            check_supported_type<Type>();
        }

        // Factorize A (symmetric). Only one triangle is referenced; we use lower ('L').
        void compute(Matx<Type> &A)
        {
            // Pack Matx -> column-major buffer a_
            // a_(r,c) in col-major is a_[r + c*n_]
            // for (int r = 0; r < n_; ++r)
            //    for (int c = 0; c < n_; ++c)
            //        a_[r + c * n_] = A(r, c);

            a_ = A; // Factorization in-place
            // Factorization in-place
            // A = L*D*L^T with pivoting (Bunch–Kaufman)
            lapack_int info = LapackLDLT<Type>::sytrf(n_, a_.data(), n_, ipiv_.data());
            if (info < 0)
                throw std::runtime_error("LAPACKE_dsytrf: illegal argument at position " + std::to_string(-info));
            if (info > 0)
                throw std::runtime_error("LAPACKE_dsytrf: factorization failed (singular pivot/block) at k=" + std::to_string(info));
        }

        Vecx<Type> solve(const Vecx<Type> &b) const
        {
            Vecx<Type> x = b; // LAPACK overwrites RHS with solution

            lapack_int info = LapackLDLT<Type>::sytrs(n_, 1,
                                                      a_.data(), n_,
                                                      ipiv_.data(),
                                                      x.data(), n_);
            if (info < 0)
                throw std::runtime_error("LAPACKE_dsytrs: illegal argument at position " + std::to_string(-info));

            return x;
        }

    private:
        int n_;
        // std::vector<Type> a_;          // factorized matrix (in-place)
        // std::vector<lapack_int> ipiv_; // pivots from SYTRF
        Matx<Type> a_;
        Vecx<lapack_int> ipiv_;
    };

} // namespace linalg
