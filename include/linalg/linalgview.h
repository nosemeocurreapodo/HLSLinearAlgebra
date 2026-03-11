#pragma once

#ifdef USE_VITIS
#include "hls_math.h"
namespace math = hls;
#else
#include <cmath>
namespace math = std;
#endif

#include <type_traits>
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
    using promote_t = decltype(A() + B());

    // ============================================================
    // CRTP base for matrix expressions
    // ============================================================
    template <typename Derived, typename ValueType>
    class MatExpr
    {
    public:
        using value_type = ValueType;

        const Derived &derived() const
        {
#pragma HLS INLINE
            return *static_cast<const Derived *>(this);
        }

        int rows() const
        {
#pragma HLS INLINE
            return derived().rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return derived().cols();
        }

        ValueType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return derived()(r, c);
        }
    };

    // ============================================================
    // Basic matrix view over external memory
    // ============================================================
    template <typename Type>
    class MatView : public MatExpr<MatView<Type>, Type>
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

        // Views should usually be copyable
        MatView(const MatView &) = default;
        MatView &operator=(const MatView &other)
        {
            if (this == &other)
                return *this;

            LINALG_ASSERT(rows_ == other.rows_);
            LINALG_ASSERT(cols_ == other.cols_);
            LINALG_ASSERT(data_ != nullptr);
            LINALG_ASSERT(other.data_ != nullptr);

        copy_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            copy_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = other(r, c);
                }
            }

            return *this;
        }

        MatView(MatView &&) = default;
        MatView &operator=(MatView &&) = default;

        void bind(Type *ptr, int rows, int cols)
        {
            LINALG_ASSERT(rows >= 0);
            LINALG_ASSERT(cols >= 0);
            LINALG_ASSERT((rows == 0 || cols == 0) || ptr != nullptr);

            data_ = ptr;
            rows_ = rows;
            cols_ = cols;
        }

        template <typename ExprDerived, typename ExprType>
        MatView &operator=(const MatExpr<ExprDerived, ExprType> &expr_base)
        {
            const ExprDerived &expr = expr_base.derived();

            LINALG_ASSERT(rows_ == expr.rows());
            LINALG_ASSERT(cols_ == expr.cols());
            LINALG_ASSERT(data_ != nullptr);

        expr_assign_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            expr_assign_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) = Type(expr(r, c));
                }
            }

            return *this;
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

        void setIdentity()
        {
            LINALG_ASSERT(rows_ == cols_);
            setZero();

        id_loop:
            for (int i = 0; i < rows_; ++i)
            {
#pragma HLS PIPELINE II = 1
                (*this)(i, i) = Type(1);
            }
        }

        template <typename ExprDerived, typename ExprType>
        void operator+=(const MatExpr<ExprDerived, ExprType> &expr_base)
        {
            const ExprDerived &expr = expr_base.derived();

            LINALG_ASSERT(rows_ == expr.rows());
            LINALG_ASSERT(cols_ == expr.cols());

        plus_eq_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            plus_eq_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) += Type(expr(r, c));
                }
            }
        }

        template <typename ExprDerived, typename ExprType>
        void operator-=(const MatExpr<ExprDerived, ExprType> &expr_base)
        {
            const ExprDerived &expr = expr_base.derived();

            LINALG_ASSERT(rows_ == expr.rows());
            LINALG_ASSERT(cols_ == expr.cols());

        minus_eq_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            minus_eq_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) -= Type(expr(r, c));
                }
            }
        }

        template <typename ScalarType>
        void operator*=(const ScalarType &s)
        {
        mul_eq_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            mul_eq_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) *= Type(s);
                }
            }
        }

        template <typename ScalarType>
        void operator/=(const ScalarType &s)
        {
        div_eq_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            div_eq_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    (*this)(r, c) /= Type(s);
                }
            }
        }

        template <typename ExprDerived, typename ExprType>
        Type dot(const MatExpr<ExprDerived, ExprType> &expr_base) const
        {
            const ExprDerived &expr = expr_base.derived();
            LINALG_ASSERT(rows_ == expr.rows());
            LINALG_ASSERT(cols_ == expr.cols());

            Type acc = Type(0);

        dot_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            dot_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    acc += (*this)(r, c) * Type(expr(r, c));
                }
            }

            return acc;
        }

        Type norm() const
        {
            Type sum = Type(0);

        norm_loop_c:
            for (int c = 0; c < cols_; ++c)
            {
            norm_loop_r:
                for (int r = 0; r < rows_; ++r)
                {
#pragma HLS PIPELINE II = 1
                    Type v = (*this)(r, c);
                    sum += v * v;
                }
            }

            return math::sqrt(sum);
        }

        Type &operator()(int r, int c)
        {
#pragma HLS INLINE
            LINALG_ASSERT(data_ != nullptr);
            LINALG_ASSERT(r >= 0 && r < rows_);
            LINALG_ASSERT(c >= 0 && c < cols_);
            const int add = c * rows_ + r; // column-major
            return data_[add];
        }

        Type operator()(int r, int c) const
        {
#pragma HLS INLINE
            LINALG_ASSERT(data_ != nullptr);
            LINALG_ASSERT(r >= 0 && r < rows_);
            LINALG_ASSERT(c >= 0 && c < cols_);
            const int add = c * rows_ + r; // column-major
            return data_[add];
        }

        Type *data() { return data_; }
        const Type *data() const { return data_; }

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

    protected:
        Type *data_;
        int rows_, cols_;
    };

    // ============================================================
    // Expression types
    // ============================================================

    template <typename LHS, typename RHS, typename OutType>
    class MatAddExpr : public MatExpr<MatAddExpr<LHS, RHS, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        MatAddExpr(const LHS &lhs, const RHS &rhs) : lhs_(lhs), rhs_(rhs)
        {
            LINALG_ASSERT(lhs_.rows() == rhs_.rows());
            LINALG_ASSERT(lhs_.cols() == rhs_.cols());
        }

        int rows() const
        {
#pragma HLS INLINE
            return lhs_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return lhs_.cols();
        }

        OutType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return OutType(lhs_(r, c)) + OutType(rhs_(r, c));
        }

    private:
        const LHS &lhs_;
        const RHS &rhs_;
    };

    template <typename LHS, typename RHS, typename OutType>
    class MatSubExpr : public MatExpr<MatSubExpr<LHS, RHS, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        MatSubExpr(const LHS &lhs, const RHS &rhs) : lhs_(lhs), rhs_(rhs)
        {
            LINALG_ASSERT(lhs_.rows() == rhs_.rows());
            LINALG_ASSERT(lhs_.cols() == rhs_.cols());
        }

        int rows() const
        {
#pragma HLS INLINE
            return lhs_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return lhs_.cols();
        }

        OutType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return OutType(lhs_(r, c)) - OutType(rhs_(r, c));
        }

    private:
        const LHS &lhs_;
        const RHS &rhs_;
    };

    template <typename Expr, typename Scalar, typename OutType>
    class MatScaleExpr : public MatExpr<MatScaleExpr<Expr, Scalar, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        MatScaleExpr(const Expr &expr, Scalar scalar) : expr_(expr), scalar_(scalar) {}

        int rows() const
        {
#pragma HLS INLINE
            return expr_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return expr_.cols();
        }

        OutType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return OutType(expr_(r, c)) * OutType(scalar_);
        }

    private:
        const Expr &expr_;
        Scalar scalar_;
    };

    template <typename Expr, typename Scalar, typename OutType>
    class MatDivExpr : public MatExpr<MatDivExpr<Expr, Scalar, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        MatDivExpr(const Expr &expr, Scalar scalar) : expr_(expr), scalar_(scalar) {}

        int rows() const
        {
#pragma HLS INLINE
            return expr_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return expr_.cols();
        }

        OutType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return OutType(expr_(r, c)) / OutType(scalar_);
        }

    private:
        const Expr &expr_;
        Scalar scalar_;
    };

    template <typename Expr, typename OutType>
    class MatNegExpr : public MatExpr<MatNegExpr<Expr, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        explicit MatNegExpr(const Expr &expr) : expr_(expr) {}

        int rows() const
        {
#pragma HLS INLINE
            return expr_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return expr_.cols();
        }

        OutType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return -OutType(expr_(r, c));
        }

    private:
        const Expr &expr_;
    };

    template <typename Expr, typename OutType>
    class MatSqrtExpr : public MatExpr<MatSqrtExpr<Expr, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        explicit MatSqrtExpr(const Expr &expr) : expr_(expr) {}

        int rows() const
        {
#pragma HLS INLINE
            return expr_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return expr_.cols();
        }

        OutType operator()(int r, int c) const
        {
#pragma HLS INLINE
            return math::sqrt(OutType(expr_(r, c)));
        }

    private:
        const Expr &expr_;
    };

    template <typename LHS, typename RHS, typename OutType>
    class MatMulExpr : public MatExpr<MatMulExpr<LHS, RHS, OutType>, OutType>
    {
    public:
        using value_type = OutType;

        MatMulExpr(const LHS &lhs, const RHS &rhs) : lhs_(lhs), rhs_(rhs)
        {
            LINALG_ASSERT(lhs_.cols() == rhs_.rows());
        }

        int rows() const
        {
#pragma HLS INLINE
            return lhs_.rows();
        }

        int cols() const
        {
#pragma HLS INLINE
            return rhs_.cols();
        }

        OutType operator()(int r, int c) const
        {
            OutType acc = OutType(0);

        mul_k_loop:
            for (int k = 0; k < lhs_.cols(); ++k)
            {
#pragma HLS PIPELINE II = 1
                acc += OutType(lhs_(r, k)) * OutType(rhs_(k, c));
            }

            return acc;
        }

    private:
        const LHS &lhs_;
        const RHS &rhs_;
    };

    // ============================================================
    // Traits
    // ============================================================

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

    // ============================================================
    // Binary operators: expression + expression
    // ============================================================

    template <typename LDerived, typename LType, typename RDerived, typename RType>
    MatAddExpr<LDerived, RDerived, typename promote_t<LType, RType>>
    operator+(const MatExpr<LDerived, LType> &lhs, const MatExpr<RDerived, RType> &rhs)
    {
        return MatAddExpr<LDerived, RDerived, typename promote_t<LType, RType>>(
            lhs.derived(), rhs.derived());
    }

    template <typename LDerived, typename LType, typename RDerived, typename RType>
    MatSubExpr<LDerived, RDerived, typename promote_t<LType, RType>>
    operator-(const MatExpr<LDerived, LType> &lhs, const MatExpr<RDerived, RType> &rhs)
    {
        return MatSubExpr<LDerived, RDerived, typename promote_t<LType, RType>>(
            lhs.derived(), rhs.derived());
    }

    template <typename LDerived, typename LType, typename RDerived, typename RType>
    MatMulExpr<LDerived, RDerived, typename promote_t<LType, RType>>
    operator*(const MatExpr<LDerived, LType> &lhs, const MatExpr<RDerived, RType> &rhs)
    {
        return MatMulExpr<LDerived, RDerived, typename promote_t<LType, RType>>(
            lhs.derived(), rhs.derived());
    }

    template <typename Derived, typename Type>
    MatNegExpr<Derived, Type> operator-(const MatExpr<Derived, Type> &expr)
    {
        return MatNegExpr<Derived, Type>(expr.derived());
    }

    // ============================================================
    // Matrix * scalar
    // ============================================================

    template <typename Derived, typename Type, typename Scalar,
              typename std::enable_if<is_scalar<typename std::decay<Scalar>::type>::value, int>::type = 0>
    MatScaleExpr<Derived, typename std::decay<Scalar>::type, typename promote_t<Type, typename std::decay<Scalar>::type>>
    operator*(const MatExpr<Derived, Type> &expr, Scalar s)
    {
        using S = typename std::decay<Scalar>::type;
        return MatScaleExpr<Derived, S, typename promote_t<Type, S>>(expr.derived(), S(s));
    }

    template <typename Scalar, typename Derived, typename Type,
              typename std::enable_if<is_scalar<typename std::decay<Scalar>::type>::value, int>::type = 0>
    MatScaleExpr<Derived, typename std::decay<Scalar>::type, typename promote_t<Type, typename std::decay<Scalar>::type>>
    operator*(Scalar s, const MatExpr<Derived, Type> &expr)
    {
        using S = typename std::decay<Scalar>::type;
        return MatScaleExpr<Derived, S, typename promote_t<Type, S>>(expr.derived(), S(s));
    }

    // ============================================================
    // Matrix / scalar
    // ============================================================

    template <typename Derived, typename Type, typename Scalar,
              typename std::enable_if<is_scalar<typename std::decay<Scalar>::type>::value, int>::type = 0>
    MatDivExpr<Derived, typename std::decay<Scalar>::type, typename promote_t<Type, typename std::decay<Scalar>::type>>
    operator/(const MatExpr<Derived, Type> &expr, Scalar s)
    {
        using S = typename std::decay<Scalar>::type;
        return MatDivExpr<Derived, S, typename promote_t<Type, S>>(expr.derived(), S(s));
    }

    // ============================================================
    // sqrt(expr)
    // ============================================================

    template <typename Derived, typename Type>
    MatSqrtExpr<Derived, Type> sqrt(const MatExpr<Derived, Type> &expr)
    {
        return MatSqrtExpr<Derived, Type>(expr.derived());
    }

    // ============================================================
    // Vector view
    // ============================================================

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class VecView : public MatView<Type>
    {
    public:
        using Base = MatView<Type>;
        using value_type = Type;

        using Base::operator=;

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

        template <typename OtherType, VecOrient OtherOrient>
        Type dot(const VecView<OtherType, OtherOrient> &rhs) const
        {
            LINALG_ASSERT(this->size() == rhs.size());

            Type acc = Type(0);

        vec_dot_loop:
            for (int i = 0; i < this->size(); ++i)
            {
#pragma HLS PIPELINE II = 1
                acc += (*this)(i)*Type(rhs(i));
            }

            return acc;
        }
    };
}