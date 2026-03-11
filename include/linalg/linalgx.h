#pragma once

#include <vector>
#include <algorithm>
#include <utility>

#include "linalg/linalgview.h"

namespace linalg
{
    template <typename Type>
    class Matx : public MatExpr<Matx<Type>, Type>
    {
    public:
        using value_type = Type;

        Matx() : data_(), rows_(0), cols_(0) {}

        Matx(int rows, int cols)
            : data_(rows * cols), rows_(rows), cols_(cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);
        }

        Matx(const Matx &) = default;
        Matx(Matx &&) = default;
        Matx &operator=(const Matx &) = default;
        Matx &operator=(Matx &&) = default;

        static Matx Zero(int rows, int cols)
        {
            Matx out(rows, cols);
            out.setZero();
            return out;
        }

        static Matx Identity(int rows, int cols)
        {
            Matx out(rows, cols);
            out.setZero();
            const int n = (rows < cols) ? rows : cols;
            for (int i = 0; i < n; ++i)
                out(i, i) = Type(1);
            return out;
        }

        template <typename ExprDerived, typename ExprType>
        explicit Matx(const MatExpr<ExprDerived, ExprType> &expr_base)
        {
            const ExprDerived &expr = expr_base.derived();
            rows_ = expr.rows();
            cols_ = expr.cols();
            LINALG_ASSERT(rows_ >= 0);
            LINALG_ASSERT(cols_ >= 0);

            data_.resize(rows_ * cols_);

            for (int c = 0; c < cols_; ++c)
                for (int r = 0; r < rows_; ++r)
                    (*this)(r, c) = Type(expr(r, c));
        }

        template <typename ExprDerived, typename ExprType>
        Matx &operator=(const MatExpr<ExprDerived, ExprType> &expr_base)
        {
            // Safe against aliasing: evaluate into a temporary first
            Matx tmp(expr_base);
            swap(tmp);
            return *this;
        }

        void swap(Matx &other) noexcept
        {
            data_.swap(other.data_);
            std::swap(rows_, other.rows_);
            std::swap(cols_, other.cols_);
        }

        void resize(int rows, int cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);
            rows_ = rows;
            cols_ = cols;
            data_.resize(rows * cols);
        }

        void setZero()
        {
            std::fill(data_.begin(), data_.end(), Type(0));
        }

        void setIdentity()
        {
            LINALG_ASSERT(rows_ == cols_);
            setZero();
            for (int i = 0; i < rows_; ++i)
                (*this)(i, i) = Type(1);
        }

        Type &operator()(int r, int c)
        {
            LINALG_ASSERT(r >= 0 && r < rows_);
            LINALG_ASSERT(c >= 0 && c < cols_);
            return data_[c * rows_ + r]; // column-major
        }

        Type operator()(int r, int c) const
        {
            LINALG_ASSERT(r >= 0 && r < rows_);
            LINALG_ASSERT(c >= 0 && c < cols_);
            return data_[c * rows_ + r]; // column-major
        }

        int rows() const { return rows_; }
        int cols() const { return cols_; }
        int size() const { return rows_ * cols_; }

        Type *data() { return data_.data(); }
        const Type *data() const { return data_.data(); }

        MatView<Type> asView()
        {
            return MatView<Type>(data_.data(), rows_, cols_);
        }

        const MatView<Type> asView() const
        {
            return MatView<Type>(const_cast<Type *>(data_.data()), rows_, cols_);
        }

    private:
        std::vector<Type> data_;
        int rows_;
        int cols_;
    };

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vecx : public VecView<Type, Orient>
    {
    public:
        using Base = VecView<Type, Orient>;
        using value_type = Type;

        using Base::operator=;

        Vecx() : Base(), storage_(), size_(0) {}

        explicit Vecx(int size)
            : Base(), storage_(size), size_(size)
        {
            rebind_();
        }

        Vecx(const Vecx &other)
            : Base(), storage_(other.storage_), size_(other.size_)
        {
            rebind_();
        }

        Vecx(const Base &mat) : Base(mat)
        {
        }

        Vecx &operator=(const Vecx &other)
        {
            if (this == &other)
                return *this;

            storage_ = other.storage_;
            size_ = other.size_;
            rebind_();
            return *this;
        }

        Vecx(Vecx &&other) noexcept
            : Base(), storage_(std::move(other.storage_)), size_(other.size_)
        {
            rebind_();
            other.size_ = 0;
            other.Base::bind(nullptr, 0);
        }

        Vecx &operator=(Vecx &&other) noexcept
        {
            if (this == &other)
                return *this;

            storage_ = std::move(other.storage_);
            size_ = other.size_;
            rebind_();

            other.size_ = 0;
            other.Base::bind(nullptr, 0);
            return *this;
        }

        static Vecx Zero(int size)
        {
            Vecx out(size);
            out.setZero();
            return out;
        }

        void resize(int size)
        {
            LINALG_ASSERT(size >= 0);
            storage_.resize(size);
            size_ = size;
            rebind_();
        }

        int size() const
        {
            return size_;
        }

        Type *data()
        {
            return storage_.data();
        }

        const Type *data() const
        {
            return storage_.data();
        }

    private:
        void rebind_()
        {
            Base::bind(storage_.data(), size_);
        }

        std::vector<Type> storage_;
        int size_;
    };
}