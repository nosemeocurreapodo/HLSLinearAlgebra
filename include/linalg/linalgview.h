#pragma once

#ifdef USE_VITIS
#include "hls_math.h"
namespace math = hls;
#else
#include <cmath>
namespace math = std;
#endif

#include <type_traits>
#include <utility>
#include "common.h"

#ifndef __SYNTHESIS__
#include <cassert>
#define LINALG_ASSERT(x) assert(x)
#else
#define LINALG_ASSERT(x) ((void)0)
#endif

namespace linalg
{
    template <typename A, typename B>
    using promote_t = decltype(std::declval<A>() + std::declval<B>());

    template <typename T>
    struct is_scalar : std::false_type
    {
    };

    template <>
    struct is_scalar<float> : std::true_type
    {
    };
    template <>
    struct is_scalar<double> : std::true_type
    {
    };
    template <>
    struct is_scalar<int> : std::true_type
    {
    };
    template <>
    struct is_scalar<unsigned int> : std::true_type
    {
    };
    template <>
    struct is_scalar<long> : std::true_type
    {
    };
    template <>
    struct is_scalar<unsigned long> : std::true_type
    {
    };
    template <>
    struct is_scalar<short> : std::true_type
    {
    };
    template <>
    struct is_scalar<unsigned short> : std::true_type
    {
    };
    template <>
    struct is_scalar<char> : std::true_type
    {
    };
    template <>
    struct is_scalar<unsigned char> : std::true_type
    {
    };

    template <typename Type>
    class MatView
    {
    public:
        using value_type = Type;

        MatView() : data_(nullptr), rows_(0), cols_(0) {}

        MatView(Type *ptr, int rows, int cols)
            : data_(ptr), rows_(rows), cols_(cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);
            LINALG_ASSERT((rows == 0 || cols == 0) || ptr != nullptr);
        }

        void bind(Type *ptr, int rows, int cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);
            LINALG_ASSERT((rows == 0 || cols == 0) || ptr != nullptr);

            data_ = ptr;
            rows_ = rows;
            cols_ = cols;
        }

        Type &operator()(int r, int c)
        {
#pragma HLS INLINE
            LINALG_ASSERT(data_ != nullptr);
            LINALG_ASSERT(r >= 0 && r < rows_);
            LINALG_ASSERT(c >= 0 && c < cols_);
            return data_[c * rows_ + r]; // column-major
        }

        Type operator()(int r, int c) const
        {
#pragma HLS INLINE
            LINALG_ASSERT(data_ != nullptr);
            LINALG_ASSERT(r >= 0 && r < rows_);
            LINALG_ASSERT(c >= 0 && c < cols_);
            return data_[c * rows_ + r]; // column-major
        }

        Type *data()
        {
#pragma HLS INLINE
            return data_;
        }

        const Type *data() const
        {
#pragma HLS INLINE
            return data_;
        }

        int rows() const
        {
#pragma HLS INLINE
            return rows_;
        }

        int cols() const
        {
#pragma HLS INLINE
            return cols_;
        }

        int size() const
        {
#pragma HLS INLINE
            return rows_ * cols_;
        }

        bool isBound() const
        {
#pragma HLS INLINE
            return data_ != nullptr;
        }

        // ------------------------------------------------------------
        // Basic fill / copy
        // ------------------------------------------------------------

        void copyTo(MatView<Type> &out) const
        {
            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());
            LINALG_ASSERT(out.data() != nullptr);

        copy_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            copy_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = (*this)(r, c);
                }
            }
        }

        void setZero()
        {
            LINALG_ASSERT(data_ != nullptr);

        zero_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            zero_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = Type(0);
                }
            }
        }

        void setConstant(const Type &value)
        {
            LINALG_ASSERT(data_ != nullptr);

        const_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            const_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = value;
                }
            }
        }

        void setIdentity()
        {
            LINALG_ASSERT(rows_ == cols_);
            setZero();

        diag_loop:
            for (int i = 0; i < rows_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i, i) = Type(1);
            }
        }

        // ------------------------------------------------------------
        // In-place operations
        // ------------------------------------------------------------

        template <typename OtherType>
        void addInPlace(const MatView<OtherType> &rhs)
        {
            LINALG_ASSERT(rows_ == rhs.rows());
            LINALG_ASSERT(cols_ == rhs.cols());

        add_ip_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            add_ip_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) += Type(rhs(r, c));
                }
            }
        }

        template <typename OtherType>
        void subInPlace(const MatView<OtherType> &rhs)
        {
            LINALG_ASSERT(rows_ == rhs.rows());
            LINALG_ASSERT(cols_ == rhs.cols());

        sub_ip_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            sub_ip_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) -= Type(rhs(r, c));
                }
            }
        }

        template <typename ScalarType>
        void scaleInPlace(ScalarType s)
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "scaleInPlace requires a scalar type");

        scale_ip_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            scale_ip_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = Type((*this)(r, c) * Type(s));
                }
            }
        }

        template <typename ScalarType>
        void divInPlace(ScalarType s)
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "divInPlace requires a scalar type");

        div_ip_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            div_ip_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = Type((*this)(r, c) / Type(s));
                }
            }
        }

        // ------------------------------------------------------------
        // Out-of-place elementwise operations
        // Usage: A.matadd(B, C)  => C = A + B
        // ------------------------------------------------------------

        template <typename OtherType, typename OutType>
        void matadd(const MatView<OtherType> &rhs, MatView<OutType> &out) const
        {
            LINALG_ASSERT(rows_ == rhs.rows());
            LINALG_ASSERT(cols_ == rhs.cols());
            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());

        add_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            add_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = OutType((*this)(r, c)) + OutType(rhs(r, c));
                }
            }
        }

        template <typename OtherType, typename OutType>
        void matsub(const MatView<OtherType> &rhs, MatView<OutType> &out) const
        {
            LINALG_ASSERT(rows_ == rhs.rows());
            LINALG_ASSERT(cols_ == rhs.cols());
            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());

        sub_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            sub_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = OutType((*this)(r, c)) - OutType(rhs(r, c));
                }
            }
        }

        template <typename ScalarType, typename OutType>
        void scale(ScalarType s, MatView<OutType> &out) const
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "scale requires a scalar type");

            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());

        scale_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            scale_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = OutType((*this)(r, c)) * OutType(s);
                }
            }
        }

        template <typename ScalarType, typename OutType>
        void div(ScalarType s, MatView<OutType> &out) const
        {
            static_assert(is_scalar<typename std::decay<ScalarType>::type>::value,
                          "div requires a scalar type");

            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());

        div_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            div_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = OutType((*this)(r, c)) / OutType(s);
                }
            }
        }

        template <typename OutType>
        void negate(MatView<OutType> &out) const
        {
            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());

        neg_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            neg_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = -OutType((*this)(r, c));
                }
            }
        }

        template <typename OutType>
        void matsqrt(MatView<OutType> &out) const
        {
            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(cols_ == out.cols());

        sqrt_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            sqrt_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(r, c) = math::sqrt(OutType((*this)(r, c)));
                }
            }
        }

        template <typename OutType>
        void transposeTo(MatView<OutType> &out) const
        {
            LINALG_ASSERT(out.rows() == cols_);
            LINALG_ASSERT(out.cols() == rows_);

        transpose_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            transpose_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    out(c, r) = OutType((*this)(r, c));
                }
            }
        }

        // ------------------------------------------------------------
        // Matrix multiply
        // Usage: A.matmul(B, C)  => C = A * B
        // ------------------------------------------------------------
        //
        // NOTE:
        // - This is the simple reference implementation.
        // - It avoids temporary full matrices.
        // - It is not yet a tiled/blocked GEMM kernel.
        // - out must not alias this or rhs.
        // ------------------------------------------------------------

        template <typename OtherType, typename OutType>
        void matmul(const MatView<OtherType> &rhs, MatView<OutType> &out) const
        {
            LINALG_ASSERT(cols_ == rhs.rows());
            LINALG_ASSERT(rows_ == out.rows());
            LINALG_ASSERT(rhs.cols() == out.cols());

        mm_loop_c:
            for (int c = 0; c < rhs.cols(); ++c)
            {
            mm_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
                    OutType acc = OutType(0);

                mm_loop_k:
                    for (int k = 0; k < cols_; ++k)
                    {
#pragma HLS PIPELINE II = 1
                        acc += OutType((*this)(r, k)) * OutType(rhs(k, c));
                    }

                    out(r, c) = acc;
                }
            }
        }

        // ------------------------------------------------------------
        // Reductions
        // ------------------------------------------------------------

        template <typename OtherType, typename AccType = promote_t<Type, OtherType>>
        AccType dot(const MatView<OtherType> &rhs) const
        {
            LINALG_ASSERT(rows_ == rhs.rows());
            LINALG_ASSERT(cols_ == rhs.cols());

            AccType acc = AccType(0);

        dot_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            dot_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    acc += AccType((*this)(r, c)) * AccType(rhs(r, c));
                }
            }

            return acc;
        }

        template <typename AccType = Type>
        AccType squaredNorm() const
        {
            AccType acc = AccType(0);

        sqnorm_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            sqnorm_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    AccType v = AccType((*this)(r, c));
                    acc += v * v;
                }
            }

            return acc;
        }

        template <typename AccType = Type>
        AccType norm() const
        {
            return math::sqrt(squaredNorm<AccType>());
        }

    protected:
        Type *data_;
        int rows_;
        int cols_;
    };

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class VecView : public MatView<Type>
    {
    public:
        using Base = MatView<Type>;
        using value_type = Type;

        VecView() : Base() {}

        VecView(Type *ptr, int size)
            : Base(ptr,
                   (Orient == VecOrient::Column ? size : 1),
                   (Orient == VecOrient::Column ? 1 : size))
        {
        }

        void bind(Type *ptr, int size)
        {
            Base::bind(ptr,
                       (Orient == VecOrient::Column ? size : 1),
                       (Orient == VecOrient::Column ? 1 : size));
        }

        int length() const
        {
#pragma HLS INLINE
            return this->size();
        }

        Type &operator()(int i)
        {
#pragma HLS INLINE
            LINALG_ASSERT(this->data_ != nullptr);
            LINALG_ASSERT(i >= 0 && i < this->size());
            return this->data_[i];
        }

        Type operator()(int i) const
        {
#pragma HLS INLINE
            LINALG_ASSERT(this->data_ != nullptr);
            LINALG_ASSERT(i >= 0 && i < this->size());
            return this->data_[i];
        }

        template <typename OtherType, VecOrient OtherOrient, typename AccType = promote_t<Type, OtherType>>
        AccType dot(const VecView<OtherType, OtherOrient> &rhs) const
        {
            LINALG_ASSERT(this->size() == rhs.size());

            AccType acc = AccType(0);

        vec_dot_loop:
            for (int i = 0; i < this->size(); ++i)
            {
#pragma HLS PIPELINE II = 1
                acc += AccType((*this)(i)) * AccType(rhs(i));
            }

            return acc;
        }

        template <typename OutType>
        void add(const VecView<Type, Orient> &rhs, VecView<OutType, Orient> &out) const
        {
            LINALG_ASSERT(this->size() == rhs.size());
            LINALG_ASSERT(this->size() == out.size());

        vadd_loop:
            for (int i = 0; i < this->size(); ++i)
            {
#pragma HLS PIPELINE II = 1
                out(i) = OutType((*this)(i)) + OutType(rhs(i));
            }
        }

        template <typename OutType>
        void sub(const VecView<Type, Orient> &rhs, VecView<OutType, Orient> &out) const
        {
            LINALG_ASSERT(this->size() == rhs.size());
            LINALG_ASSERT(this->size() == out.size());

        vsub_loop:
            for (int i = 0; i < this->size(); ++i)
            {
#pragma HLS PIPELINE II = 1
                out(i) = OutType((*this)(i)) - OutType(rhs(i));
            }
        }
    };

} // namespace linalg