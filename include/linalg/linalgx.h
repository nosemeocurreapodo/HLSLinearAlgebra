
#pragma once

#include <cassert> // for assert
#include <memory>  // for unique_ptr

#include "common.h"
// #include <cmath>

namespace linalg
{
    //============================================================
    // Basic fixed-size matrix class
    //============================================================

    template <typename Type>
    class Matx
    {
    public:
        Matx()
            : data_(nullptr), rows_(0), cols_(0)
        {
        }

        Matx(int rows, int cols)
            : data_(rows * cols > 0 ? std::make_unique<Type[]>(rows * cols) : nullptr), rows_(rows), cols_(cols)
        {
        }

        Matx(const Matx &other)
            : Matx(other.rows_, other.cols_)
        {
            if (data_)
            {
                std::copy(other.data_.get(),
                          other.data_.get() + rows_ * cols_,
                          data_.get());
            }
        }

        Matx &operator=(const Matx &other)
        {
            if (this == &other)
                return *this;

            // Reallocate if size changes
            int newSize = other.rows_ * other.cols_;
            if (newSize != rows_ * cols_)
            {
                data_.reset();
                if (newSize > 0)
                {
                    data_ = std::make_unique<Type[]>(newSize);
                }
            }

            rows_ = other.rows_;
            cols_ = other.cols_;

            if (data_)
            {
                std::copy(other.data_.get(),
                          other.data_.get() + rows_ * cols_,
                          data_.get());
            }

            return *this;
        }

        // Matx(const Matx &) = delete;
        // Matx &operator=(const Matx &) = delete;

        void setZero()
        {
            assert(rows_ * cols_ > 0);

        mat_const_other_loop:
            for (int y = 0; y < rows_; y++)
            {
            mat_const_other_loop_x:
                for (int x = 0; x < cols_; x++)
                {
                    // #pragma HLS UNROLL
                    (*this)(y, x) = Type(0);
                }
            }
        }

        void setIdentity()
        {
            assert(rows_ == cols_);
            assert(rows_ * cols_ > 0);

            setZero();
            for (int i = 0; i < cols_; i++)
                (*this)(i, i) = Type(1);
        }

        static Matx Zero(int rows, int cols)
        {
            Matx result(rows, cols);
            for (int i = 0; i < rows * cols; i++)
                result.data_[i] = Type(0);
            return result;
        }

        static Matx Identity(int rows, int cols)
        {
            assert(rows == cols);

            Matx result = Zero(rows, cols);
            for (int i = 0; i < rows; i++)
                result(i, i) = Type(1);
            return result;
        }

        Matx<Type> transpose() const
        {
            Matx<Type> result = Zero(cols_, rows_);
            for (int r = 0; r < rows_; r++)
                for (int c = 0; c < cols_; c++)
                    result(c, r) = (*this)(r, c);
            return result;
        }

        template <typename Type2>
        Matx<Type> operator*(const Matx<Type2> &rhs) const
        {
            assert(cols_ == rhs.rows());

            Matx<Type> result = Matx<Type>::Zero(rows_, rhs.cols());
            for (int r = 0; r < rows_; r++)
            {
                for (int c = 0; c < rhs.cols(); c++)
                {
                    Type acc = 0.0;
                    for (int k = 0; k < cols_; k++)
                    {
                        acc += (*this)(r, k) * rhs(k, c);
                    }
                    result(r, c) = acc;
                }
            }
            return result;
        }

        Matx operator+(const Matx &other) const
        {
            assert(rows_ == other.rows() && cols_ == other.cols());

            Matx result(rows_, cols_);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    result(r, c) = (*this)(r, c) + other(r, c);
            return result;
        }

        Matx operator-(const Matx &other) const
        {
            assert(rows_ == other.rows() && cols_ == other.cols());

            Matx result(rows_, cols_);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    result(r, c) = (*this)(r, c) - other(r, c);
            return result;
        }

        template <typename OutType, typename InType>
        OutType conv(const Matx<InType> &rhs) const
        {
            OutType result = Type(0);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    result += OutType((*this)(r, c) * rhs(r, c));
            return result;
        }

        template <typename Type2>
        Type dot(const Matx<Type2> &rhs)
        {
            assert(rows_ == rhs.rows() && cols_ == rhs.cols());

            Type result = Type(0);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    result += (*this)(r, c) * rhs(r, c);
            return result;
        }

        void operator+=(const Matx &other)
        {
            assert(rows_ == other.rows() && cols_ == other.cols());

            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    (*this)(r, c) = (*this)(r, c) + other(r, c);
        }

        template <typename Type2>
        void operator*=(const Type2 &s)
        {
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    (*this)(r, c) = (*this)(r, c) * s;
        }

        Matx operator-() const
        {
            Matx result(rows_, cols_);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    result(r, c) = -(*this)(r, c);
            return result;
        }

        Matx sqrt()
        {
            Matx result(rows_, cols_);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    result(r, c) = std::sqrt((*this)(r, c));
            return result;
        }

        // Frobenius norm
        Type norm() const
        {
            Type sum = Type(0);
            for (int c = 0; c < cols_; c++)
                for (int r = 0; r < rows_; r++)
                    sum += (*this)(r, c) * (*this)(r, c);
            return std::sqrt(sum);
        }

        // Element accessors (row, col)
        Type &operator()(int r, int c)
        {
            //  column major
            //   int add = r * cols_ + c;
            //  row major
            int add = c * rows_ + r;

            return get_(add);
        }

        Type operator()(int r, int c) const
        {
            //  column major
            //  int add = r * cols_ + c;
            //  row major
            int add = c * rows_ + r;

            return get_(add);
        }

        Type *data()
        {
            return data_.get();
        }

        const Type *data() const
        {
            return data_.get();
        }

        // Dimension accessors
        int rows() const { return rows_; }
        int cols() const { return cols_; }
        int size() const { return rows_ * cols_; }

    protected:
        Type &get_(int add)
        {
            return data_[add];
        }

        Type get_(int add) const
        {
            return data_[add];
        }

        std::unique_ptr<Type[]> data_;
        int rows_, cols_;
    };

    template <typename Type>
    Matx<Type> operator*(const Matx<Type> &m, Type s)
    {
        Matx<Type> result(m.rows(), m.cols());
        for (int c = 0; c < m.cols(); c++)
            for (int r = 0; r < m.rows(); r++)
                result(r, c) = m(r, c) * s;
        return result;
    }

    template <typename Type>
    Matx<Type> operator*(Type s, const Matx<Type> &m)
    {
        return m * s;
    }

    template <typename Type>
    Matx<Type> operator/(const Matx<Type> &m, Type s)
    {
        Matx<Type> result(m.rows(), m.cols());
        for (int c = 0; c < m.cols(); c++)
            for (int r = 0; r < m.rows(); r++)
                result(r, c) = m(r, c) / s;
        return result;
    }

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vecx : public Matx<Type>
    {
    public:
        using Base = Matx<Type>;

        Vecx() : Base() {}
        Vecx(int size) : Base((Orient == VecOrient::Column ? size : 1),
                              (Orient == VecOrient::Column ? 1 : size))
        {
        }
        Vecx(const Base &mat) : Base(mat)
        {
        }

        static Vecx Zero(int size)
        {
            Vecx result(size);
            for (int i = 0; i < size; i++)
                result(i) = Type(0);
            return result;
        }

        // 1D indexing, orientation-agnostic
        Type &operator()(int i)
        {
#pragma HLS inline
            return this->get_(i);
        }

        Type operator()(int i) const
        {
#pragma HLS inline
            return this->get_(i);
        }

        Type dot(Vecx &rhs)
        {
            Type acc = Type(0);
            for (int i = 0; i < this->size(); i++)
                acc += (*this)(i)*rhs(i);
            return acc;
        }

    private:
    };
}