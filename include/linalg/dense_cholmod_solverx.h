#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>
#include <lapacke.h>

#include "linalgx.h"

namespace linalg
{
    template <typename T>
    struct LapackCholmod;

    template <>
    struct LapackCholmod<float>
    {

        static lapack_int posv(int n, int nrhs, float *a, int lda,
                               float *b, int ldb)
        {
            return LAPACKE_sposv(LAPACK_COL_MAJOR, 'L',
                                 static_cast<lapack_int>(n),
                                 static_cast<lapack_int>(nrhs),
                                 a,
                                 static_cast<lapack_int>(lda),
                                 b,
                                 static_cast<lapack_int>(ldb));
        }
    };

    template <>
    struct LapackCholmod<double>
    {

        static lapack_int posv(int n, int nrhs, double *a, int lda,
                               double *b, int ldb)
        {
            return LAPACKE_dposv(LAPACK_COL_MAJOR, 'L',
                                 static_cast<lapack_int>(n),
                                 static_cast<lapack_int>(nrhs),
                                 a,
                                 static_cast<lapack_int>(lda),
                                 b,
                                 static_cast<lapack_int>(ldb));
        }
    };

    // template <typename T>
    // inline void check_supported_type()
    //{
    //     static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
    //                   "LDLT_LAPACK only supports float and double");
    // }

    template <typename Type>
    class DENSE_CHOLESKY_CHOLMOD
    {
    public:
        DENSE_CHOLESKY_CHOLMOD(int size)
            : H_ref_(size, size), x_ref_(size), size_(size)
        {
        }

        ~DENSE_CHOLESKY_CHOLMOD()
        {
        }

        DENSE_CHOLESKY_CHOLMOD(const DENSE_CHOLESKY_CHOLMOD &) = delete;
        DENSE_CHOLESKY_CHOLMOD &operator=(const DENSE_CHOLESKY_CHOLMOD &) = delete;

        void compute(const Matx<Type> &A)
        {
            H_ref_ = A;
        }

        Vecx<Type> solve(const Vecx<Type> &b)
        {
            Vecx<Type> x = b; // LAPACK overwrites RHS with solution
            // x_ref_ = b; // overwritten
            int info = LapackCholmod<Type>::posv(size_, 1,
                                                 H_ref_.data(), size_,
                                                 x.data(), size_);

            return x_ref_;
        }

    private:
        Matx<Type> H_ref_;
        Vecx<Type> x_ref_;
        int size_;
    };
} // namespace linalg