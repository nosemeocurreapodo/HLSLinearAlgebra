#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>

extern "C" {
#include <cholmod.h>
}

#include "sparsemat_csc.h"   // your SparseMatCSC from earlier

namespace linalg
{
    class CHOLESKY_CHOLMOD
    {
    public:
        CHOLESKY_CHOLMOD()
        {
            cholmod_start(&cc_);

            // Good defaults; CHOLMOD will often pick supernodal automatically.
            // You can force supernodal if you want:
            // cc_.supernodal = CHOLMOD_SUPERNODAL;
        }

        ~CHOLESKY_CHOLMOD()
        {
            free_factor();
            cholmod_finish(&cc_);
        }

        CHOLESKY_CHOLMOD(const CHOLESKY_CHOLMOD&) = delete;
        CHOLESKY_CHOLMOD& operator=(const CHOLESKY_CHOLMOD&) = delete;

        int n() const { return n_; }
        bool is_computed() const { return factor_ != nullptr; }

        // Factorize SPD symmetric A stored in CSC (lower triangle only recommended).
        void compute(const SparseMatCSC &A)
        {
            if (!A.finalized())
                throw std::runtime_error("CHOLESKY_CHOLMOD::compute: A must be finalized()");
            if (A.n() <= 0)
                throw std::invalid_argument("CHOLESKY_CHOLMOD::compute: A.n() <= 0");

            n_ = A.n();
            free_factor();

            // Create a cholmod_sparse view that points to A's internal buffers (no copy).
            cholmod_sparse M;
            std::memset(&M, 0, sizeof(M));

            M.nrow  = n_;
            M.ncol  = n_;
            M.nzmax = A.nnz();

            // CHOLMOD wants non-const pointers; it will not modify if packed/sorted.
            M.p = const_cast<linalg::Index*>(A.Ap());
            M.i = const_cast<linalg::Index*>(A.Ai());
            M.x = const_cast<double*>(A.Ax());

            M.z = nullptr;
            M.stype  = -1;              // -1 => lower triangle stored
            M.itype  = CHOLMOD_INT;     // matches Index=int in SparseMatCSC
            M.xtype  = CHOLMOD_REAL;
            M.dtype  = CHOLMOD_DOUBLE;
            M.sorted = 1;               // our finalize() sorts each column
            M.packed = 1;

            // Analyze + factorize
            factor_ = cholmod_analyze(&M, &cc_);
            if (!factor_)
                throw std::runtime_error("CHOLMOD: analyze failed");

            int ok = cholmod_factorize(&M, factor_, &cc_);
            if (!ok || cc_.status != CHOLMOD_OK)
                throw std::runtime_error("CHOLMOD: factorize failed (is A truly SPD?)");
        }

        std::vector<double> solve(const std::vector<double> &b) const
        {
            if (!factor_) throw std::runtime_error("CHOLESKY_CHOLMOD::solve: compute() not called");
            if ((int)b.size() != n_) throw std::invalid_argument("CHOLESKY_CHOLMOD::solve: wrong b size");

            // Wrap b as cholmod_dense (no copy)
            cholmod_dense B;
            std::memset(&B, 0, sizeof(B));
            B.nrow = n_;
            B.ncol = 1;
            B.nzmax = n_;
            B.d = n_;
            B.x = const_cast<double*>(b.data());
            B.xtype = CHOLMOD_REAL;
            B.dtype = CHOLMOD_DOUBLE;

            cholmod_dense *X = cholmod_solve(CHOLMOD_A, factor_, &B, &cc_);
            if (!X)
                throw std::runtime_error("CHOLMOD: solve failed");

            std::vector<double> x(n_);
            std::memcpy(x.data(), X->x, sizeof(double) * n_);

            cholmod_free_dense(&X, &cc_);
            return x;
        }

    private:
        void free_factor()
        {
            if (factor_)
            {
                cholmod_free_factor(&factor_, &cc_);
                factor_ = nullptr;
            }
        }

        int n_ = 0;
        mutable cholmod_common cc_{};
        cholmod_factor *factor_ = nullptr;
    };
} // namespace linalg