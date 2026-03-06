#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>
#include <algorithm>

extern "C"
{
#include <cholmod.h>
}

#include <lapacke.h>

// Requires your SparseMatCSC (CSC, lower triangle, sorted, finalized) from earlier.
// Assumes:
// - Index = int
// - Ax is double
#include "sparsemat_csc.h"

namespace linalg
{
    // -----------------------------
    // Schur complement solver for
    // [ Hvv  Hvp ] [dv] = [gv]
    // [ Hpv  Hpp ] [dp]   [gp]
    //
    // Assumptions:
    // - SPD overall (GN/LM normal equations), and Hvv is SPD.
    // - Hvv sparse (mesh grid), size n_mesh x n_mesh
    // - pose block small: n_pose x n_pose (dense)
    // - Hvp dense-ish: n_mesh x n_pose
    //
    // Solves using:
    // - CHOLMOD factorization of Hvv
    // - Schur: S = Hpp - Hpv * inv(Hvv) * Hvp
    // - Dense LAPACK (Cholesky) on S
    // -----------------------------
    class SchurSolver
    {
    public:
        SchurSolver()
        {
            cholmod_start(&cc_);
            // cc_.supernodal = CHOLMOD_SUPERNODAL; // force supernodal
            // cc_.final_ll = 1;                    // LL' factor (often beneficial)
            // cc_.final_super = 1;                 // keep supernodal form
        }
        ~SchurSolver()
        {
            reset();
            cholmod_finish(&cc_);
        }

        SchurSolver(const SchurSolver &) = delete;
        SchurSolver &operator=(const SchurSolver &) = delete;

        void reset()
        {
            if (factor_)
                cholmod_free_factor(&factor_, &cc_);
            factor_ = nullptr;
            analyzed_ = false;
            n_mesh_ = n_pose_ = 0;
            // B_.clear();
            // X_.clear();
            // y_.clear();
            // Z_.clear();
            // S_.clear();
            // b_.clear();
            // dp_.clear();
            // dv_.clear();
        }

        // Analyze once for fixed-pattern Hvv.
        // Hvv must be finalized CSC, LOWER triangle only, sorted/packed.
        // Optional permutation:
        //  - if nullptr: CHOLMOD chooses ordering
        //  - else: user_perm size n_mesh giving elimination order
        void analyze_mesh_pattern(const SparseMatCSC &Hvv_pattern, const std::vector<int> *user_perm = nullptr)
        {
            if (!Hvv_pattern.finalized())
                throw std::runtime_error("analyze_mesh_pattern: Hvv_pattern must be finalized()");
            if (Hvv_pattern.n() <= 0)
                throw std::invalid_argument("analyze_mesh_pattern: n<=0");

            n_mesh_ = Hvv_pattern.n();

            if (factor_)
                cholmod_free_factor(&factor_, &cc_);
            factor_ = nullptr;

            cholmod_sparse A = make_cholmod_view_(Hvv_pattern);

            if (user_perm)
            {
                if ((int)user_perm->size() != n_mesh_)
                    throw std::invalid_argument("analyze_mesh_pattern: user_perm wrong size");

                factor_ = cholmod_analyze_p(&A,
                                            const_cast<int *>(user_perm->data()),
                                            nullptr, 0,
                                            &cc_);
            }
            else
            {
                factor_ = cholmod_analyze(&A, &cc_);
            }

            if (!factor_ || cc_.status != CHOLMOD_OK)
                throw std::runtime_error("CHOLMOD analyze failed");

            analyzed_ = true;
        }

        // Set pose dimension once (e.g. 7 poses * 6 dof = 42).
        void set_pose_dim(int n_pose)
        {
            if (n_pose <= 0)
                throw std::invalid_argument("set_pose_dim: n_pose<=0");
            n_pose_ = n_pose;

            // Allocate work buffers sized from n_mesh_ if known later.
            if (n_mesh_ > 0)
                alloc_work_();
        }

        // Factorize Hvv numeric for current iteration.
        // Hvv_values must have SAME pattern as analyzed Hvv_pattern (same Ap/Ai layout).
        void factorize_mesh_numeric(const SparseMatCSC &Hvv_values)
        {
            if (!analyzed_)
                throw std::runtime_error("factorize_mesh_numeric: call analyze_mesh_pattern() first");
            if (!Hvv_values.finalized())
                throw std::runtime_error("factorize_mesh_numeric: Hvv_values must be finalized()");
            if (Hvv_values.n() != n_mesh_)
                throw std::runtime_error("factorize_mesh_numeric: dimension mismatch");
            if (n_pose_ <= 0)
                throw std::runtime_error("factorize_mesh_numeric: call set_pose_dim() first");

            // Ensure buffers exist
            alloc_work_();

            cholmod_sparse A = make_cholmod_view_(Hvv_values);

            int ok = cholmod_factorize(&A, factor_, &cc_);
            if (!ok || cc_.status != CHOLMOD_OK)
                throw std::runtime_error("CHOLMOD factorize failed (Hvv not SPD?)");
        }

        // Solve full system using Schur complement.
        //
        // Inputs:
        //  - Hvp: (n_mesh x n_pose) dense column-major
        //  - Hpp: (n_pose x n_pose) dense column-major (lower triangle used)
        //  - gv:  (n_mesh)
        //  - gp:  (n_pose)
        //
        // Outputs:
        //  - dv (n_mesh)
        //  - dp (n_pose)
        //
        // Optional LM damping:
        //  - lambda_v added to diag(Hvv) (you should do this in Hvv_values before factorize)
        //  - lambda_p added to diag(Hpp) inside this function (cheap)
        void solve(const Matx<double> &Hvp,
                   const Matx<double> &Hpp,
                   const Vecx<double> &gv,
                   const Vecx<double> &gp,
                   Vecx<double> &dv_out,
                   Vecx<double> &dp_out,
                   double lambda_p = 0.0)
        {
            if (!factor_)
                throw std::runtime_error("solve: factorize_mesh_numeric() not called");
            if (Hvp.rows() != n_mesh_ || Hvp.cols() != n_pose_)
                throw std::invalid_argument("solve: Hvp has wrong shape");
            if (Hpp.rows() != n_pose_ || Hpp.cols() != n_pose_)
                throw std::invalid_argument("solve: Hpp has wrong shape");
            if ((int)gv.size() != n_mesh_)
                throw std::invalid_argument("solve: gv wrong size");
            if ((int)gp.size() != n_pose_)
                throw std::invalid_argument("solve: gp wrong size");

            // Work buffers
            alloc_work_();

            // ------------------------------------------------------------
            // Step A: Solve Hvv^{-1} [ gv | Hvp ] in one CHOLMOD call
            //
            // B = [gv, Hvp]  size: n_mesh x (1+n_pose)
            // X = Hvv^{-1} B = [ y,  Z ] same size
            // ------------------------------------------------------------
            // Fill B
            // col 0 = gv
            for (int r = 0; r < n_mesh_; ++r)
                B_(r, 0) = gv(r);

            // cols 1..n_pose = Hvp columns
            for (int c = 0; c < n_pose_; ++c)
                for (int r = 0; r < n_mesh_; ++r)
                    B_(r, 1 + c) = Hvp(r, c);

            cholmod_dense Bc = make_dense_view_(B_);
            cholmod_dense *Xc = cholmod_solve(CHOLMOD_A, factor_, &Bc, &cc_);
            if (!Xc || cc_.status != CHOLMOD_OK)
                throw std::runtime_error("CHOLMOD solve failed (multi-RHS)");

            // Copy Xc into X_ (so we can free Xc)
            {
                const double *xptr = static_cast<const double *>(Xc->x);
                std::memcpy(X_.data(), xptr, sizeof(double) * (size_t)n_mesh_ * (size_t)(1 + n_pose_));
                cholmod_free_dense(&Xc, &cc_);
            }

            // Extract y and Z views
            // y = X_(:,0)
            for (int r = 0; r < n_mesh_; ++r)
                y_(r) = X_(r, 0);

            // Z = X_(:, 1..)
            for (int c = 0; c < n_pose_; ++c)
                for (int r = 0; r < n_mesh_; ++r)
                    Z_(r, c) = X_(r, 1 + c);

            // ------------------------------------------------------------
            // Step B: Build Schur system
            //
            // S = Hpp - Hpv * Z  = Hpp - Hvp^T * Z   (n_pose x n_pose)
            // b = gp  - Hpv * y  = gp  - Hvp^T * y   (n_pose)
            // ------------------------------------------------------------
            // Start S = Hpp
            // S_ = Matx<double>(n_pose_, n_pose_);
            // S_.setZero();
            for (int j = 0; j < n_pose_; ++j)
                for (int i = 0; i < n_pose_; ++i)
                    S_(i, j) = Hpp(i, j);

            // Add damping to pose block if requested
            if (lambda_p != 0.0)
                for (int i = 0; i < n_pose_; ++i)
                    S_(i, i) += lambda_p;

            // Compute T = Hvp^T * Z  (n_pose x n_pose)
            // S -= T
            for (int i = 0; i < n_pose_; ++i)
            {
                for (int j = 0; j < n_pose_; ++j)
                {
                    double sum = 0.0;
                    // sum over mesh rows
                    for (int r = 0; r < n_mesh_; ++r)
                        sum += Hvp(r, i) * Z_(r, j);

                    S_(i, j) -= sum;
                }
            }

            // b = gp - Hvp^T * y
            // b_ = Vecx<double>(n_pose_);
            // b_.setZero();
            for (int i = 0; i < n_pose_; ++i)
            {
                double sum = 0.0;
                for (int r = 0; r < n_mesh_; ++r)
                    sum += Hvp(r, i) * y_(r);

                b_(i) = gp(i) - sum;
            }

            // ------------------------------------------------------------
            // Step C: Solve S dp = b  (S should be SPD if your overall system is SPD)
            // Use LAPACKE_dposv (Cholesky solve), column-major
            // ------------------------------------------------------------
            dp_ = b_; // RHS overwritten

            // LAPACK expects leading dimension = n_pose
            // dposv overwrites S with its Cholesky factor.
            int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L',
                                     n_pose_, 1,
                                     S_.data(), n_pose_,
                                     dp_.data(), n_pose_);
            if (info != 0)
                throw std::runtime_error("LAPACKE_dposv failed, info=" + std::to_string(info));

            // ------------------------------------------------------------
            // Step D: Recover dv = y - Z dp
            // ------------------------------------------------------------
            // dv_ = Vecx<double>::Zero(n_mesh_);
            // dv_.setZero();
            for (int r = 0; r < n_mesh_; ++r)
            {
                double zdp = 0.0;
                for (int j = 0; j < n_pose_; ++j)
                    zdp += Z_(r, j) * dp_(j);

                dv_(r) = y_(r) - zdp;
            }

            dv_out = dv_;
            dp_out = dp_;
        }

    private:
        void alloc_work_()
        {
            if (n_mesh_ <= 0 || n_pose_ <= 0)
                return;

            const int rhs_cols = 1 + n_pose_;
            if (B_.rows() != n_mesh_ || B_.cols() != rhs_cols)
                // B_.resize(n_mesh_, rhs_cols);
                B_ = Matx<double>(n_mesh_, rhs_cols);

            if (X_.rows() != n_mesh_ || X_.cols() != rhs_cols)
                // X_.resize(n_mesh_, rhs_cols);
                X_ = Matx<double>(n_mesh_, rhs_cols);

            if ((int)y_.size() != n_mesh_)
                // y_.assign(n_mesh_, 0.0);
                y_ = Vecx<double>::Zero(n_mesh_);

            if (Z_.rows() != n_mesh_ || Z_.cols() != n_pose_)
                // Z_.resize(n_mesh_, n_pose_);
                Z_ = Matx<double>(n_mesh_, n_pose_);

            if ((int)b_.size() != n_pose_)
                // b_.assing(n_pose_, 0.0);
                b_ = Vecx<double>::Zero(n_pose_);

            if ((int)dp_.size() != n_pose_)
                // dp_.assign(n_pose_, 0.0);
                dp_ = Vecx<double>::Zero(n_pose_);

            if ((int)dv_.size() != n_mesh_)
                // dv_.assign(n_mesh_, 0.0);
                dv_ = Vecx<double>(n_mesh_);

            if (S_.rows() != n_pose_ || S_.cols() != n_pose_)
                S_ = Matx<double>(n_pose_, n_pose_);

            if (b_.size() != n_pose_)
                b_ = Vecx<double>(n_pose_);

            if (dv_.size() != n_mesh_)
                // dv_ = Vecx<double>::Zero(n_mesh_);
                dv_ = Vecx<double>(n_mesh_);
        }

        static cholmod_dense make_dense_view_(Matx<double> &M)
        {
            cholmod_dense D;
            std::memset(&D, 0, sizeof(D));
            D.nrow = M.rows();
            D.ncol = M.cols();
            D.nzmax = (size_t)M.rows() * (size_t)M.cols();
            D.d = M.rows(); // leading dimension for col-major
            D.x = M.data();
            D.xtype = CHOLMOD_REAL;
            D.dtype = CHOLMOD_DOUBLE;
            return D;
        }

        cholmod_sparse make_cholmod_view_(const SparseMatCSC &A) const
        {
            cholmod_sparse M;
            std::memset(&M, 0, sizeof(M));
            M.nrow = A.n();
            M.ncol = A.n();
            M.nzmax = A.nnz();

            // CHOLMOD wants non-const pointers; packed/sorted means it won't modify.
            M.p = const_cast<Index *>(A.Ap());
            M.i = const_cast<Index *>(A.Ai());
            M.x = const_cast<double *>(A.Ax());

            M.z = nullptr;
            M.stype = -1;          // lower triangle stored
            M.itype = CHOLMOD_INT; // Index=int
            M.xtype = CHOLMOD_REAL;
            M.dtype = CHOLMOD_DOUBLE;
            M.sorted = 1;
            M.packed = 1;
            return M;
        }

        int n_mesh_ = 0;
        int n_pose_ = 0;

        bool analyzed_ = false;
        cholmod_common cc_{};
        cholmod_factor *factor_ = nullptr;

        // Multi-RHS: B = [gv | Hvp], X = inv(Hvv)*B
        Matx<double> B_;
        Matx<double> X_;

        // y = inv(Hvv)*gv
        Vecx<double> y_;

        // Z = inv(Hvv)*Hvp  (n_mesh x n_pose)
        Matx<double> Z_;

        // Schur system S dp = b
        Matx<double> S_;
        Vecx<double> b_;

        // Outputs cached
        Vecx<double> dp_;
        Vecx<double> dv_;
    };

} // namespace linalg