#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <cstring>   // std::memcpy

// SuiteSparse headers
extern "C" {
#include <amd.h>
#include <ldl.h>
}

namespace linalg
{
    // Simple view of a sparse CSC matrix (n x n)
    // Ap: column pointers size n+1
    // Ai: row indices size nnz
    // Ax: values size nnz
    //
    // Requirements:
    // - CSC must represent a symmetric matrix.
    // - For best performance, store only ONE triangle (typically LOWER)
    //   and include the diagonal. LDL can work with symmetric structure;
    //   but you must be consistent.
    template <typename T>
    struct SparseCSCView
    {
        int n = 0;
        const int *Ap = nullptr;
        const int *Ai = nullptr;
        const T *Ax = nullptr;
    };

    // Sparse LDL^T factorization using SuiteSparse LDL + AMD ordering
    //
    // Type: SuiteSparse LDL is typically used with double. We'll enforce double
    // to avoid silent precision/ABI mismatches.
    class LDLT_SUITSPARSE
    {
    public:
        LDLT_SUITSPARSE() = default;
        explicit LDLT_SUITSPARSE(int n) { reserve(n, 0); }

        int n() const { return n_; }
        bool is_computed() const { return computed_; }

        // Reserve internal buffers (optional)
        void reserve(int n, int nnz_hint)
        {
            if (n < 0) throw std::invalid_argument("LDLT_SUITSPARSE::reserve: n < 0");
            n_ = n;
            computed_ = false;

            // Permutation buffers
            P_.assign(n_, 0);
            Pinv_.assign(n_, 0);

            // LDL symbolic/numeric work buffers
            Parent_.assign(n_, 0);
            Lnz_.assign(n_, 0);
            Flag_.assign(n_, 0);

            Pattern_.assign(n_, 0);
            Y_.assign(n_, 0.0);

            // Factor storage (sizes finalized at compute)
            Lp_.assign(n_ + 1, 0);
            Li_.clear();
            Lx_.clear();
            D_.assign(n_, 0.0);

            // Solve buffers
            rhs_perm_.assign(n_, 0.0);
            x_perm_.assign(n_, 0.0);

            // Optional hint (avoid realloc churn)
            if (nnz_hint > 0)
            {
                Li_.reserve(nnz_hint);
                Lx_.reserve(nnz_hint);
            }
        }

        // Compute factorization of A (symmetric) given in CSC.
        //
        // Notes:
        // - Uses AMD ordering on A's pattern.
        // - Factorizes PAP^T = LDL^T.
        // - Stores L (unit lower triangular, without diagonal) + D.
        void compute(const SparseCSCView<double> &A)
        {
            if (A.n <= 0) throw std::invalid_argument("LDLT_SUITSPARSE::compute: A.n <= 0");
            if (!A.Ap || !A.Ai || !A.Ax) throw std::invalid_argument("LDLT_SUITSPARSE::compute: null CSC pointers");
            if (A.n != n_) reserve(A.n, A.Ap[A.n]); // adopt size

            const int n = n_;
            const int nnz = A.Ap[n];

            if (nnz < 0) throw std::invalid_argument("LDLT_SUITSPARSE::compute: nnz < 0");

            // 1) AMD ordering: P such that A(P,P) reduces fill
            // AMD wants CSC pattern (Ap, Ai). Values ignored.
            {
                std::vector<double> Control(AMD_CONTROL, 0.0);
                std::vector<double> Info(AMD_INFO, 0.0);
                amd_defaults(Control.data());

                int status = amd_order(n, A.Ap, A.Ai, P_.data(), Control.data(), Info.data());
                if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED)
                {
                    throw std::runtime_error("LDLT_SUITSPARSE: amd_order failed, status=" + std::to_string(status));
                }
                // Build inverse permutation
                for (int k = 0; k < n; ++k) Pinv_[P_[k]] = k;
            }

            // 2) Symbolic factorization: determine L structure
            // LDL_symbolic needs:
            // - A in CSC
            // - Permutation P (and Pinv)
            // - Outputs: Lp, Parent, Lnz, Flag
            //
            // It sets Lp (size n+1) and Lnz (nnz in each column of L),
            // from which we can size Li/Lx.
            int sym_ok = LDL_symbolic(
                n,
                A.Ap,
                A.Ai,
                Lp_.data(),
                Parent_.data(),
                Lnz_.data(),
                Flag_.data(),
                P_.data(),
                Pinv_.data());

            if (!sym_ok)
                throw std::runtime_error("LDLT_SUITSPARSE: LDL_symbolic failed (matrix/pattern issue?)");

            // Lp_ now filled; total nnz of L is Lp_[n]
            const int lnz_total = Lp_[n];
            if (lnz_total < 0)
                throw std::runtime_error("LDLT_SUITSPARSE: invalid L nnz from symbolic factorization");

            Li_.assign(lnz_total, 0);
            Lx_.assign(lnz_total, 0.0);
            D_.assign(n, 0.0);

            // 3) Numeric factorization
            // LDL_numeric computes L and D for PAP^T.
            // It returns "d" (success) flag; if a zero pivot occurs, it can fail.
            int num_ok = LDL_numeric(
                n,
                A.Ap,
                A.Ai,
                A.Ax,
                Lp_.data(),
                Parent_.data(),
                Lnz_.data(),
                Li_.data(),
                Lx_.data(),
                D_.data(),
                Y_.data(),
                Pattern_.data(),
                Flag_.data(),
                P_.data(),
                Pinv_.data());

            if (!num_ok)
                throw std::runtime_error("LDLT_SUITSPARSE: LDL_numeric failed (zero/near-zero pivot?)");

            computed_ = true;
        }

        // Solve Ax = b using stored LDL^T of PAP^T
        std::vector<double> solve(const std::vector<double> &b) const
        {
            if (!computed_) throw std::runtime_error("LDLT_SUITSPARSE::solve: compute() not called");
            if ((int)b.size() != n_) throw std::invalid_argument("LDLT_SUITSPARSE::solve: b has wrong size");

            const int n = n_;

            // rhs_perm_ = P*b  (LDL uses "perm" as x = b[P] style)
            // Use LDL_perm: x = b[P]
            LDL_perm(n, rhs_perm_.data(), b.data(), P_.data());

            // Solve (L D L^T) x_perm_ = rhs_perm_ in-place using LDL_solve
            std::memcpy(x_perm_.data(), rhs_perm_.data(), sizeof(double) * n);
            LDL_solve(n, x_perm_.data(), Lp_.data(), Li_.data(), Lx_.data(), D_.data());

            // x = P^T * x_perm_
            // LDL_permt: x = b[Pinv]  (equivalent to applying inverse permutation)
            std::vector<double> x(n, 0.0);
            LDL_permt(n, x.data(), x_perm_.data(), P_.data()); // P^T

            return x;
        }

        // Optional: expose permutation/factors if you want them
        const std::vector<int>& P() const { return P_; }
        const std::vector<int>& Pinv() const { return Pinv_; }
        const std::vector<int>& Lp() const { return Lp_; }
        const std::vector<int>& Li() const { return Li_; }
        const std::vector<double>& Lx() const { return Lx_; }
        const std::vector<double>& D() const { return D_; }

    private:
        int n_ = 0;
        bool computed_ = false;

        // Permutation (AMD)
        std::vector<int> P_, Pinv_;

        // LDL symbolic outputs / workspace
        std::vector<int> Lp_, Parent_, Lnz_, Flag_;
        std::vector<int> Pattern_;
        std::vector<double> Y_;

        // LDL numeric outputs
        std::vector<int> Li_;
        std::vector<double> Lx_;
        std::vector<double> D_;

        // Solve buffers
        mutable std::vector<double> rhs_perm_;
        mutable std::vector<double> x_perm_;
    };

} // namespace linalg