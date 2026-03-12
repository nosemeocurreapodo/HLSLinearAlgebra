#pragma once

#include <vector>
#include <utility>
#include <type_traits>

#include "linalg/linalgview.h"

namespace linalg
{
    template <typename Type>
    class Matx : public MatView<Type>
    {
    public:
        using Base = MatView<Type>;
        using value_type = Type;

        using Base::operator();
        using Base::addInPlace;
        using Base::cols;
        using Base::copyTo;
        using Base::data;
        using Base::div;
        using Base::divInPlace;
        using Base::dot;
        using Base::matadd;
        using Base::matmul;
        using Base::matsqrt;
        using Base::matsub;
        using Base::negate;
        using Base::norm;
        using Base::rows;
        using Base::scale;
        using Base::scaleInPlace;
        using Base::setIdentity;
        using Base::setZero;
        using Base::size;
        using Base::squaredNorm;
        using Base::subInPlace;

        Matx()
            : Base(), storage_(), rows_(0), cols_(0)
        {
            rebind_();
        }

        Matx(int rows, int cols)
            : Base(), storage_(rows * cols), rows_(rows), cols_(cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);
            rebind_();
        }

        Matx(const Matx &other)
            : Base(), storage_(other.storage_), rows_(other.rows_), cols_(other.cols_)
        {
            rebind_();
        }

        Matx(Matx &&other) noexcept
            : Base(), storage_(std::move(other.storage_)), rows_(other.rows_), cols_(other.cols_)
        {
            rebind_();
            other.rows_ = 0;
            other.cols_ = 0;
            other.Base::bind(nullptr, 0, 0);
        }

        Matx &operator=(const Matx &other)
        {
            if (this == &other)
                return *this;

            storage_ = other.storage_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            rebind_();
            return *this;
        }

        Matx &operator=(Matx &&other) noexcept
        {
            if (this == &other)
                return *this;

            storage_ = std::move(other.storage_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            rebind_();

            other.rows_ = 0;
            other.cols_ = 0;
            other.Base::bind(nullptr, 0, 0);
            return *this;
        }

        template <typename OtherType>
        explicit Matx(const MatView<OtherType> &other)
            : Base(), storage_(other.rows() * other.cols()), rows_(other.rows()), cols_(other.cols())
        {
            rebind_();
            this->copyFrom(other);
        }

        template <typename OtherType>
        Matx &operator=(const MatView<OtherType> &other)
        {
            resize(other.rows(), other.cols());
            this->copyFrom(other);
            return *this;
        }

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

        void resize(int rows, int cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);

            rows_ = rows;
            cols_ = cols;
            storage_.resize(rows * cols);
            rebind_();
        }

        void swap(Matx &other) noexcept
        {
            storage_.swap(other.storage_);
            std::swap(rows_, other.rows_);
            std::swap(cols_, other.cols_);
            rebind_();
            other.rebind_();
        }

        MatView<Type> asView()
        {
            return MatView<Type>(storage_.data(), rows_, cols_);
        }

        const MatView<Type> asView() const
        {
            return MatView<Type>(const_cast<Type *>(storage_.data()), rows_, cols_);
        }

        template <typename OtherType>
        void copyFrom(const MatView<OtherType> &other)
        {
            LINALG_ASSERT(rows_ == other.rows());
            LINALG_ASSERT(cols_ == other.cols());

        copy_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            copy_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = Type(other(r, c));
                }
            }
        }

        Matx transpose()
        {
            Matx out(cols_, rows_);
            this->transposeTo(out);
            return out;
        }

        // ------------------------------------------------------------
        // In-place operators
        // ------------------------------------------------------------

        template <typename OtherType>
        Matx &operator+=(const MatView<OtherType> &rhs)
        {
            this->addInPlace(rhs);
            return *this;
        }

        template <typename OtherType>
        Matx &operator-=(const MatView<OtherType> &rhs)
        {
            this->subInPlace(rhs);
            return *this;
        }

        template <typename ScalarType>
        Matx &operator*=(ScalarType s)
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "Matx::operator*= requires a scalar type");
            this->scaleInPlace(s);
            return *this;
        }

        template <typename ScalarType>
        Matx &operator/=(ScalarType s)
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "Matx::operator/= requires a scalar type");
            this->divInPlace(s);
            return *this;
        }

    private:
        void rebind_()
        {
            Type *ptr = storage_.empty() ? nullptr : storage_.data();
            Base::bind(ptr, rows_, cols_);
        }

        std::vector<Type> storage_;
        int rows_;
        int cols_;
    };

    // ============================================================
    // Matx binary operators
    // ============================================================

    template <typename LType, typename RType>
    Matx<promote_t<LType, RType>>
    operator+(const MatView<LType> &lhs, const MatView<RType> &rhs)
    {
        using OutType = promote_t<LType, RType>;
        LINALG_ASSERT(lhs.rows() == rhs.rows());
        LINALG_ASSERT(lhs.cols() == rhs.cols());

        Matx<OutType> out(lhs.rows(), lhs.cols());
        lhs.matadd(rhs, out);
        return out;
    }

    template <typename LType, typename RType>
    Matx<promote_t<LType, RType>>
    operator-(const MatView<LType> &lhs, const MatView<RType> &rhs)
    {
        using OutType = promote_t<LType, RType>;
        LINALG_ASSERT(lhs.rows() == rhs.rows());
        LINALG_ASSERT(lhs.cols() == rhs.cols());

        Matx<OutType> out(lhs.rows(), lhs.cols());
        lhs.matsub(rhs, out);
        return out;
    }

    template <typename Type>
    Matx<Type> operator-(const MatView<Type> &mat)
    {
        Matx<Type> out(mat.rows(), mat.cols());
        mat.negate(out);
        return out;
    }

    // Matrix-matrix multiply
    template <typename LType, typename RType>
    Matx<promote_t<LType, RType>>
    operator*(const MatView<LType> &lhs, const MatView<RType> &rhs)
    {
        using OutType = promote_t<LType, RType>;
        LINALG_ASSERT(lhs.cols() == rhs.rows());

        Matx<OutType> out(lhs.rows(), rhs.cols());
        lhs.matmul(rhs, out);
        return out;
    }

    // Matrix-scalar
    template <typename Type, typename ScalarType,
              typename std::enable_if<is_scalar<typename std::decay<ScalarType>::type>::value, int>::type = 0>
    Matx<promote_t<Type, typename std::decay<ScalarType>::type>>
    operator*(const MatView<Type> &mat, ScalarType s)
    {
        using OutType = promote_t<Type, typename std::decay<ScalarType>::type>;
        Matx<OutType> out(mat.rows(), mat.cols());
        mat.scale(s, out);
        return out;
    }

    template <typename ScalarType, typename Type,
              typename std::enable_if<is_scalar<typename std::decay<ScalarType>::type>::value, int>::type = 0>
    Matx<promote_t<Type, typename std::decay<ScalarType>::type>>
    operator*(ScalarType s, const MatView<Type> &mat)
    {
        using OutType = promote_t<Type, typename std::decay<ScalarType>::type>;
        Matx<OutType> out(mat.rows(), mat.cols());
        mat.scale(s, out);
        return out;
    }

    template <typename Type, typename ScalarType,
              typename std::enable_if<is_scalar<typename std::decay<ScalarType>::type>::value, int>::type = 0>
    Matx<promote_t<Type, typename std::decay<ScalarType>::type>>
    operator/(const MatView<Type> &mat, ScalarType s)
    {
        using OutType = promote_t<Type, typename std::decay<ScalarType>::type>;
        Matx<OutType> out(mat.rows(), mat.cols());
        mat.div(s, out);
        return out;
    }

    // ============================================================
    // Vecx
    // ============================================================

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vecx : public VecView<Type, Orient>
    {
    public:
        using Base = VecView<Type, Orient>;
        using value_type = Type;

        using Base::operator();
        using Base::copyTo;
        using Base::data;
        using Base::dot;
        using Base::length;
        using Base::norm;
        using Base::setConstant;
        using Base::setZero;
        using Base::size;
        using Base::squaredNorm;

        Vecx()
            : Base(), storage_(), size_(0)
        {
            rebind_();
        }

        explicit Vecx(int size)
            : Base(), storage_(size), size_(size)
        {
            LINALG_ASSERT(size >= 0);
            rebind_();
        }

        Vecx(const Vecx &other)
            : Base(), storage_(other.storage_), size_(other.size_)
        {
            rebind_();
        }

        Vecx(Vecx &&other) noexcept
            : Base(), storage_(std::move(other.storage_)), size_(other.size_)
        {
            rebind_();
            other.size_ = 0;
            other.Base::bind(nullptr, 0);
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

        template <typename OtherType, VecOrient OtherOrient>
        explicit Vecx(const VecView<OtherType, OtherOrient> &other)
            : Base(), storage_(other.size()), size_(other.size())
        {
            static_assert(OtherOrient == Orient, "Vector orientation mismatch");
            rebind_();
            copyFrom(other);
        }

        template <typename OtherType, VecOrient OtherOrient>
        Vecx &operator=(const VecView<OtherType, OtherOrient> &other)
        {
            static_assert(OtherOrient == Orient, "Vector orientation mismatch");
            resize(other.size());
            copyFrom(other);
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

        void swap(Vecx &other) noexcept
        {
            storage_.swap(other.storage_);
            std::swap(size_, other.size_);
            rebind_();
            other.rebind_();
        }

        template <typename OtherType, VecOrient OtherOrient>
        void copyFrom(const VecView<OtherType, OtherOrient> &other)
        {
            static_assert(OtherOrient == Orient, "Vector orientation mismatch");
            LINALG_ASSERT(size_ == other.size());

        copy_vec_loop:
            for (int i = 0; i < size_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i) = Type(other(i));
            }
        }

        // ------------------------------------------------------------
        // In-place operators
        // ------------------------------------------------------------

        template <typename OtherType>
        Vecx &operator+=(const VecView<OtherType, Orient> &rhs)
        {
            LINALG_ASSERT(size_ == rhs.size());

        vadd_ip_loop:
            for (int i = 0; i < size_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i) += Type(rhs(i));
            }
            return *this;
        }

        template <typename OtherType>
        Vecx &operator-=(const VecView<OtherType, Orient> &rhs)
        {
            LINALG_ASSERT(size_ == rhs.size());

        vsub_ip_loop:
            for (int i = 0; i < size_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i) -= Type(rhs(i));
            }
            return *this;
        }

        template <typename ScalarType>
        Vecx &operator*=(ScalarType s)
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "Vecx::operator*= requires a scalar type");

        vscale_ip_loop:
            for (int i = 0; i < size_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i) = Type((*this)(i)*Type(s));
            }
            return *this;
        }

        template <typename ScalarType>
        Vecx &operator/=(ScalarType s)
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "Vecx::operator/= requires a scalar type");

        vdiv_ip_loop:
            for (int i = 0; i < size_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i) = Type((*this)(i) / Type(s));
            }
            return *this;
        }

    private:
        void rebind_()
        {
            Type *ptr = storage_.empty() ? nullptr : storage_.data();
            Base::bind(ptr, size_);
        }

        std::vector<Type> storage_;
        int size_;
    };

    // ============================================================
    // Vecx binary operators
    // ============================================================

    template <typename LType, typename RType, VecOrient Orient>
    Vecx<promote_t<LType, RType>, Orient>
    operator+(const VecView<LType, Orient> &lhs, const VecView<RType, Orient> &rhs)
    {
        using OutType = promote_t<LType, RType>;
        LINALG_ASSERT(lhs.size() == rhs.size());

        Vecx<OutType, Orient> out(lhs.size());

    vadd_loop:
        for (int i = 0; i < lhs.size(); ++i)
        {
#pragma HLS PIPELINE II = 1
            out(i) = OutType(lhs(i)) + OutType(rhs(i));
        }

        return out;
    }

    template <typename LType, typename RType, VecOrient Orient>
    Vecx<promote_t<LType, RType>, Orient>
    operator-(const VecView<LType, Orient> &lhs, const VecView<RType, Orient> &rhs)
    {
        using OutType = promote_t<LType, RType>;
        LINALG_ASSERT(lhs.size() == rhs.size());

        Vecx<OutType, Orient> out(lhs.size());

    vsub_loop:
        for (int i = 0; i < lhs.size(); ++i)
        {
#pragma HLS PIPELINE II = 1
            out(i) = OutType(lhs(i)) - OutType(rhs(i));
        }

        return out;
    }

    template <typename Type, VecOrient Orient>
    Vecx<Type, Orient> operator-(const VecView<Type, Orient> &vec)
    {
        Vecx<Type, Orient> out(vec.size());

    vneg_loop:
        for (int i = 0; i < vec.size(); ++i)
        {
#pragma HLS PIPELINE II = 1
            out(i) = -Type(vec(i));
        }

        return out;
    }

    template <typename Type, typename ScalarType, VecOrient Orient,
              typename std::enable_if<is_scalar<typename std::decay<ScalarType>::type>::value, int>::type = 0>
    Vecx<promote_t<Type, typename std::decay<ScalarType>::type>, Orient>
    operator*(const VecView<Type, Orient> &vec, ScalarType s)
    {
        using OutType = promote_t<Type, typename std::decay<ScalarType>::type>;
        Vecx<OutType, Orient> out(vec.size());

    vscale_loop:
        for (int i = 0; i < vec.size(); ++i)
        {
#pragma HLS PIPELINE II = 1
            out(i) = OutType(vec(i)) * OutType(s);
        }

        return out;
    }

    template <typename ScalarType, typename Type, VecOrient Orient,
              typename std::enable_if<is_scalar<typename std::decay<ScalarType>::type>::value, int>::type = 0>
    Vecx<promote_t<Type, typename std::decay<ScalarType>::type>, Orient>
    operator*(ScalarType s, const VecView<Type, Orient> &vec)
    {
        using OutType = promote_t<Type, typename std::decay<ScalarType>::type>;
        Vecx<OutType, Orient> out(vec.size());

    vscale_loop2:
        for (int i = 0; i < vec.size(); ++i)
        {
#pragma HLS PIPELINE II = 1
            out(i) = OutType(vec(i)) * OutType(s);
        }

        return out;
    }

    template <typename Type, typename ScalarType, VecOrient Orient,
              typename std::enable_if<is_scalar<typename std::decay<ScalarType>::type>::value, int>::type = 0>
    Vecx<promote_t<Type, typename std::decay<ScalarType>::type>, Orient>
    operator/(const VecView<Type, Orient> &vec, ScalarType s)
    {
        using OutType = promote_t<Type, typename std::decay<ScalarType>::type>;
        Vecx<OutType, Orient> out(vec.size());

    vdiv_loop:
        for (int i = 0; i < vec.size(); ++i)
        {
#pragma HLS PIPELINE II = 1
            out(i) = OutType(vec(i)) / OutType(s);
        }

        return out;
    }

} // namespace linalg