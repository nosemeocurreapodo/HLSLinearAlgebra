#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cstring>

namespace linalg
{
    // If your SuiteSparse/CHOLMOD is built with SuiteSparse_long indices,
    // you can switch this to int64_t and set cholmod itype accordingly.
    using Index = int;

    struct Triplet
    {
        Index r;
        Index c;
        double v;
    };

    // Simple CSC sparse matrix for CHOLMOD
    // - Stores only LOWER triangle if store_lower_only=true (recommended for SPD).
    // - finalize() compresses triplets into CSC, sorts, and merges duplicates.
    class SparseMatCSC
    {
    public:
        SparseMatCSC() = default;

        explicit SparseMatCSC(Index n, bool store_lower_only = true)
            : n_(n), store_lower_only_(store_lower_only)
        {
            if (n_ < 0) throw std::invalid_argument("SparseMatCSC: n < 0");
        }

        void resize(Index n, bool store_lower_only = true)
        {
            if (n < 0) throw std::invalid_argument("SparseMatCSC::resize: n < 0");
            n_ = n;
            store_lower_only_ = store_lower_only;
            clear();
        }

        void clear()
        {
            triplets_.clear();
            Ap_.clear();
            Ai_.clear();
            Ax_.clear();
            finalized_ = false;
        }

        Index n() const { return n_; }
        Index nnz() const { return (Index)Ax_.size(); }
        bool finalized() const { return finalized_; }

        // Reserve triplets to reduce reallocs.
        void reserve_triplets(size_t nnz_hint)
        {
            triplets_.reserve(nnz_hint);
        }

        // Add an entry (i,j,val). If store_lower_only_, it is folded into lower triangle.
        // For SPD, you typically add each nonzero once; duplicates are allowed and will be summed.
        void add(Index i, Index j, double val)
        {
            if (n_ <= 0) throw std::runtime_error("SparseMatCSC::add: matrix size not set");
            if (i < 0 || j < 0 || i >= n_ || j >= n_) throw std::out_of_range("SparseMatCSC::add: index out of range");

            if (store_lower_only_)
            {
                if (i < j) std::swap(i, j); // fold to lower triangle
            }

            triplets_.push_back({i, j, val});
            finalized_ = false;
        }

        // Build CSC from triplets:
        // - counts per column
        // - fills unsorted CSC
        // - sorts rows within each column
        // - merges duplicates by summing values
        void finalize()
        {
            if (n_ <= 0) throw std::runtime_error("SparseMatCSC::finalize: matrix size not set");

            // Build column counts
            std::vector<Index> col_count(n_, 0);
            for (const auto &t : triplets_)
            {
                // we store CSC by column = t.c
                ++col_count[t.c];
            }

            // Prefix sum -> Ap_
            Ap_.assign(n_ + 1, 0);
            Ap_[0] = 0;
            for (Index c = 0; c < n_; ++c)
                Ap_[c + 1] = Ap_[c] + col_count[c];

            const Index nnz_raw = Ap_[n_];
            Ai_.assign(nnz_raw, 0);
            Ax_.assign(nnz_raw, 0.0);

            // Fill columns using a running pointer
            std::vector<Index> next = Ap_;
            for (const auto &t : triplets_)
            {
                Index p = next[t.c]++;
                Ai_[p] = t.r;
                Ax_[p] = t.v;
            }

            // Now: sort and merge duplicates within each column
            // We'll produce new Ai/Ax and new Ap in one pass.
            std::vector<Index> Ap2(n_ + 1, 0);
            std::vector<Index> Ai2;
            std::vector<double> Ax2;
            Ai2.reserve(nnz_raw);
            Ax2.reserve(nnz_raw);

            Ap2[0] = 0;

            for (Index c = 0; c < n_; ++c)
            {
                Index start = Ap_[c];
                Index end   = Ap_[c + 1];
                Index len   = end - start;

                if (len == 0)
                {
                    Ap2[c + 1] = (Index)Ai2.size();
                    continue;
                }

                // Create a local index list [start,end) and sort by row
                // For speed, sort pairs in-place by creating an array of positions.
                // But simplest: sort a small vector of (row,val).
                tmp_.clear();
                tmp_.reserve((size_t)len);

                for (Index p = start; p < end; ++p)
                    tmp_.push_back({Ai_[p], Ax_[p]});

                std::sort(tmp_.begin(), tmp_.end(),
                          [](const RowVal &a, const RowVal &b) { return a.r < b.r; });

                // Merge duplicates
                Index k = 0;
                while (k < (Index)tmp_.size())
                {
                    Index r = tmp_[k].r;
                    double v = tmp_[k].v;
                    ++k;
                    while (k < (Index)tmp_.size() && tmp_[k].r == r)
                    {
                        v += tmp_[k].v;
                        ++k;
                    }

                    // Optional: drop explicit zeros created by cancellation
                    if (v != 0.0)
                    {
                        Ai2.push_back(r);
                        Ax2.push_back(v);
                    }
                }

                Ap2[c + 1] = (Index)Ai2.size();
            }

            Ap_.swap(Ap2);
            Ai_.swap(Ai2);
            Ax_.swap(Ax2);

            finalized_ = true;

            // You can free triplets to save memory once finalized
            triplets_.clear();
            triplets_.shrink_to_fit();
        }

        // Raw CSC pointers for CHOLMOD
        const Index* Ap() const { require_finalized(); return Ap_.data(); }
        const Index* Ai() const { require_finalized(); return Ai_.data(); }
        const double* Ax() const { require_finalized(); return Ax_.data(); }

    private:
        struct RowVal { Index r; double v; };

        void require_finalized() const
        {
            if (!finalized_) throw std::runtime_error("SparseMatCSC: call finalize() before accessing CSC arrays");
        }

        Index n_ = 0;
        bool store_lower_only_ = true;
        bool finalized_ = false;

        std::vector<Triplet> triplets_;

        // CSC storage
        std::vector<Index> Ap_; // size n+1
        std::vector<Index> Ai_; // size nnz
        std::vector<double> Ax_; // size nnz

        // temp buffer reused during finalize() to avoid re-alloc each column
        std::vector<RowVal> tmp_;
    };
} // namespace linalg