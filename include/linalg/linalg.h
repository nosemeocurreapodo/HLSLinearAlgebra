
#pragma once

#ifdef USE_VITIS
// #define HLS_PRAGMA(x) _Pragma(#x)
// #define HLS_INLINE HLS_PRAGMA(HLS INLINE)
// #define HLS_UNROLL HLS_PRAGMA(HLS UNROLL)
// #define HLS_PIPELINE HLS_PRAGMA(HLS PIPELINE)
// #define HLS_ARRAY_PARTITION(var, type, dim) HLS_PRAGMA(HLS ARRAY_PARTITION variable = var type = type dim = dim)
#include "hls_math.h"
namespace math = hls;
#else
// #define HLS_PRAGMA(x)
// #define HLS_INLINE
// #define HLS_UNROLL
// #define HLS_PIPELINE
// #define HLS_ARRAY_PARTITION(var, type, dim)
#include <cmath>
namespace math = std;
#endif

#include "common.h"

namespace linalg
{
    //============================================================
    // Basic fixed-size matrix class
    //============================================================

    template <typename Type, int _rows, int _cols>
    class Mat
    {
    public:
        Mat()
        {
        }

        // template <typename OtherType>
        // Mat(const OtherType *data)
        //{
        // mat_const_data_loop:
        //     for (int i = 0; i < _rows * _cols; i++)
        //     {
        //  #pragma HLS UNROLL
        //        data_[i] = data[i];
        //    }
        //}

        template <typename Type2>
        Mat(const Mat<Type2, _rows, _cols> &other)
        {
        mat_const_other_loop:
            for (int y = 0; y < _rows; y++)
            {
            mat_const_other_loop_x:
                for (int x = 0; x < _cols; x++)
                {
                    // #pragma HLS UNROLL
                    (*this)(y, x) = other(y, x);
                }
            }
        }

        void setZero()
        {
        mat_const_other_loop:
            for (int y = 0; y < _rows; y++)
            {
            mat_const_other_loop_x:
                for (int x = 0; x < _cols; x++)
                {
                    // #pragma HLS UNROLL
                    (*this)(y, x) = Type(0);
                }
            }
        }

        void setIdentity()
        {
            setZero();
            for (int i = 0; i < _cols; i++)
                (*this)(i, i) = Type(1);
        }

        static Mat Zero()
        {
            Mat result;
        mat_zero_loop:
            for (int i = 0; i < _rows * _cols; i++)
            {
                // #pragma HLS UNROLL
                result.data_[i] = Type(0);
            }
            return result;
        }

        static Mat Identity()
        {
#ifndef USE_VITIS
            static_assert(_rows == _cols, "Identity only makes sense for square matrices");
#endif
            Mat result = Zero();
        mat_identity_loop_i:
            for (int i = 0; i < _rows; i++)
                // #pragma HLS UNROLL
                result(i, i) = Type(1);
            return result;
        }

        /*
        operator float() const
        {
#ifndef USE_VITIS
            assert(_rows == 1 && _cols == 1);
#endif
            return get_(0, 0);
        }
        */

        // Assignment operator
        template <typename OtherType>
        Mat &operator=(const Mat<OtherType, _rows, _cols> &other)
        {
            // if (this != &other)
            /*
            {
            mat_assign_loop_r:
                for (int r = 0; r < _rows; r++)
                mat_assign_loop_c:
                    for (int c = 0; c < _cols; c++)
                        get_(r, c) = other(r, c);
            }
            */
        mat_assign_loop:
            for (int i = 0; i < _rows * _cols; i++)
            {
                // #pragma HLS UNROLL
                data_[i] = Type(other.data()[i]);
            }

            return *this;
        }

        bool operator==(const Mat &other) const
        {
            bool result = true;
        mat_eq_loop:
            for (int i = 0; i < _rows * _cols; i++)
            {
                // #pragma HLS UNROLL
                result = result && (data_[i] == other.data_[i]);
            }
            return result;
        }

        template <typename Type2, int __rows, int __cols>
        Mat<Type, _rows, __cols> operator*(const Mat<Type2, __rows, __cols> &rhs) const
        {
#ifndef USE_VITIS
            static_assert(_cols == __rows, "Inner dimensions must match for matrix multiplication");
#endif
            // #pragma HLS INLINE

            Mat<Type, _rows, __cols> result; // = Mat<Type, _rows, __cols>::Zero();
        mat_mult_loop_r:
            for (int r = 0; r < _rows; r++)
            {
                // #pragma HLS PIPELINE II = 1
            mat_mult_loop_c:
                for (int c = 0; c < __cols; c++)
                {
                    Type acc = 0.0;
                    // #pragma HLS LOOP_FLATTEN
                mat_mult_loop_k:
                    for (int k = 0; k < _cols; k++)
                    {
                        // #pragma HLS LOOP_FLATTEN
                        acc += (*this)(r, k) * rhs(k, c);
                    }
                    result(r, c) = acc;
                }
            }
            return result;
        }

        Mat operator+(const Mat &other) const
        {
            Mat result;
        mat_add_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_add_loop_r:
                for (int r = 0; r < _rows; r++)
                    result(r, c) = (*this)(r, c) + other(r, c);
            return result;
        }

        void operator+=(const Mat &other)
        {
        mat_add_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_add_loop_r:
                for (int r = 0; r < _rows; r++)
                {
                    Type val = (*this)(r, c) + other(r, c);
                    (*this)(r, c) = val;
                }
        }

        void operator-=(const Mat &other)
        {
        mat_add_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_add_loop_r:
                for (int r = 0; r < _rows; r++)
                {
                    Type val = (*this)(r, c) - other(r, c);
                    (*this)(r, c) = val;
                }
        }

        Mat operator-(const Mat &other) const
        {
            Mat result;
        mat_sub_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_sub_loop_r:
                for (int r = 0; r < _rows; r++)
                    result(r, c) = (*this)(r, c) - other(r, c);
            return result;
        }

        Mat cwiseMin(const Mat &other) const
        {
            Mat result;
        mat_min_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_min_loop_r:
                for (int r = 0; r < _rows; r++)
                    result(r, c) = (*this)(r, c) < other(r, c) ? (*this)(r, c) : other(r, c);
            return result;
        }

        Mat cwiseMax(const Mat &other) const
        {
            Mat result;
        mat_max_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_max_loop_r:
                for (int r = 0; r < _rows; r++)
                    result(r, c) = (*this)(r, c) > other(r, c) ? (*this)(r, c)
                                                               : other(r, c);
            return result;
        }

        Mat<Type, _cols, _rows> transpose() const
        {
            Mat<Type, _cols, _rows> result;
        mat_tran_loop_r:
            for (int r = 0; r < _rows; r++)
            mat_tran_loop_c:
                for (int c = 0; c < _cols; c++)
                    result(c, r) = (*this)(r, c);
            return result;
        }

        template <typename OutType, typename InType>
        OutType conv(const Mat<InType, _rows, _cols> &rhs) const
        {
            OutType result = Type(0);
        mat_conv_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_conv_loop_r:
                for (int r = 0; r < _rows; r++)
                    result += OutType((*this)(r, c) * rhs(r, c));
            return result;
        }

        Mat operator-() const
        {
            Mat result;
        mat_neg_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_neg_loop_r:
                for (int r = 0; r < _rows; r++)
                    result(r, c) = -(*this)(r, c);
            return result;
        }

        // Frobenius norm
        Type norm() const
        {
            Type sum = Type(0);
        mat_norm_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_norm_loop_r:
                for (int r = 0; r < _rows; r++)
                    sum += (*this)(r, c) * (*this)(r, c);
            return math::sqrt(sum);
        }

        Mat sqrt() const
        {
            Mat result;
        mat_sqrt_loop_c:
            for (int c = 0; c < _cols; c++)
            mat_sqrt_loop_r:
                for (int r = 0; r < _rows; r++)
                    result(r, c) = math::sqrt((*this)(r, c));
            return result;
        }

        // Element accessors (row, col)
        Type &operator()(int r, int c)
        {
#pragma HLS inline
            //  column major
            //   int add = r * _cols + c;
            //  row major
            int add = c * _rows + r;

            return get_(add);
        }

        Type operator()(int r, int c) const
        {
#pragma HLS inline
            //  column major
            //   int add = r * _cols + c;
            //  row major
            int add = c * _rows + r;

            return get_(add);
        }

        Type *data()
        {
            return data_;
        }

        const Type *data() const
        {
            return data_;
        }

        // Dimension accessors
        static constexpr int rows() { return _rows; }
        static constexpr int cols() { return _cols; }
        static constexpr int size() { return _rows * _cols; }

    protected:
        Type &get_(int add)
        {
#pragma HLS inline
            return data_[add];
        }

        Type get_(int add) const
        {
#pragma HLS inline
            return data_[add];
        }

        Type data_[_rows * _cols];
    };

    template <typename Type, int rows, int cols>
    Mat<Type, rows, cols> operator*(const Mat<Type, rows, cols> &m, Type s)
    {
        // #pragma HLS INLINE
        Mat<Type, rows, cols> result;
    mat_fmult_loop_c:
        for (int c = 0; c < cols; c++)
        {
        mat_fmult_loop_r:
            for (int r = 0; r < rows; r++)
            {
                // #pragma HLS PIPELINE II = 1

                // #pragma HLS LOOP_FLATTEN
                result(r, c) = m(r, c) * s;
            }
        }
        return result;
    }

    template <typename Type, int rows, int cols>
    Mat<Type, rows, cols> operator*(Type s, const Mat<Type, rows, cols> &m)
    {
        return m * s;
    }

    template <typename Type, int rows, int cols>
    Mat<Type, rows, cols> operator/(const Mat<Type, rows, cols> &m, Type s)
    {
        Mat<Type, rows, cols> result;
    mat_fdiv_loop_c:
        for (int c = 0; c < cols; c++)
        mat_fdiv_loop_r:
            for (int r = 0; r < rows; r++)
                result(r, c) = m(r, c) / s;
        return result;
    }

    template <typename Type, int rows, int cols>
    void operator*=(Mat<Type, rows, cols> &m, Type s)
    {
        // #pragma HLS INLINE
    mat_fmult_loop_c:
        for (int c = 0; c < cols; c++)
        {
        mat_fmult_loop_r:
            for (int r = 0; r < rows; r++)
            {
                // #pragma HLS PIPELINE II = 1

                // #pragma HLS LOOP_FLATTEN
                m(r, c) = m(r, c) * s;
            }
        }
    }

    template <typename Type, int rows, int cols>
    void operator/=(Mat<Type, rows, cols> &m, Type s)
    {
        // #pragma HLS INLINE
    mat_fmult_loop_c:
        for (int c = 0; c < cols; c++)
        {
        mat_fmult_loop_r:
            for (int r = 0; r < rows; r++)
            {
                // #pragma HLS PIPELINE II = 1

                // #pragma HLS LOOP_FLATTEN
                m(r, c) = m(r, c) / s;
            }
        }
    }

    //============================================================
    // Various Vector specializations (just Nx1 Mat)
    //============================================================

    template <typename Type, int Size, VecOrient Orient = VecOrient::Column>
    class Vec : public Mat<
                    Type,
                    (Orient == VecOrient::Column ? Size : 1),
                    (Orient == VecOrient::Column ? 1 : Size)>
    {
    public:
        using Base = Mat<
            Type,
            (Orient == VecOrient::Column ? Size : 1),
            (Orient == VecOrient::Column ? 1 : Size)>;

        Vec() : Base() {}
        Vec(const Base &mat) : Base(mat) {}

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

        Type dot(const Vec &rhs) const
        {
            Type acc = Type(0);
            for (int i = 0; i < Size; i++)
                acc += (*this)(i)*rhs(i);
            return acc;
        }

        Type norm() const
        {
            Type acc = Type(0);
            for (int i = 0; i < Size; i++)
                acc += (*this)(i) * (*this)(i);
            return math::sqrt(acc);
        }
    };
    /*
    template <typename Type>
    class Vec1 : public Mat<Type, 1, 1>
    {
    public:
        Vec1() : Mat<Type, 1, 1>() {}
        Vec1(Type x)
        {
            (*this)(0) = x;
        }
    };
    */
    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vec2 : public Vec<Type, 2, Orient>
    {
    public:
        Vec2() {}
        template <typename Type2>
        Vec2(const Mat<Type2,
                       (Orient == VecOrient::Column ? 2 : 1),
                       (Orient == VecOrient::Column ? 1 : 2)> &mat)
            : Vec<Type, 2>(mat) // call the base-class copy constructor
        {
        }

        template <typename Type1, typename Type2>
        Vec2(Type1 x, Type2 y)
        {
            (*this)(0) = Type(x);
            (*this)(1) = Type(y);
        }
        /*
        template <typename OtherType>
        Vec2 operator*(const OtherType &s) const
        {
            // #pragma HLS inline

            Vec2 result;
            result(0) = (*this)(0) * s;
            result(1) = (*this)(1) * s;
            return result;
        }
        */

        Type cross(const Vec2 &rhs) const
        {
            // #pragma HLS inline

            return (*this)(0) * rhs(1) - (*this)(1) * rhs(0);
        }

    private:
    };

    // template <typename Type>
    // Vec2<Type> operator*(Type s, const Vec2<Type> &m)
    // {
    // #pragma HLS inline

    //    return m * s;
    //}

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vec3 : public Vec<Type, 3, Orient>
    {
    public:
        Vec3() : Vec<Type, 3>() {}
        template <typename Type2>
        Vec3(const Mat<Type2,
                       (Orient == VecOrient::Column ? 3 : 1),
                       (Orient == VecOrient::Column ? 1 : 3)> &mat)
            : Vec<Type, 3>(mat) // call the base-class copy constructor
        {
        }

        template <typename Type1, typename Type2, typename Type3>
        Vec3(Type1 x, Type2 y, Type3 z)
        {
            (*this)(0) = Type(x);
            (*this)(1) = Type(y);
            (*this)(2) = Type(z);
        }

        // bool operator==(const Mat<Type, 3, 1> &other) const
        //{
        //     return ((*this)(0) == other(0, 0) &&
        //             (*this)(1) == other(1, 0) &&
        //             (*this)(2) == other(2, 0));
        // }

        /*
        template <typename OtherType>
        Vec3 operator*(const Mat<OtherType, 3, 1> &other) const
        {
            Vec3 result;
            result(0) = (*this)(0) * other(0);
            result(1) = (*this)(1) * other(1);
            result(2) = (*this)(2) * other(2);
            return result;
        }

        template <typename OtherType>
        Vec3 operator+(const Mat<OtherType, 3, 1> &other) const
        {
            Vec3 result;
            result(0) = (*this)(0) + other(0, 0);
            result(1) = (*this)(1) + other(1, 0);
            result(2) = (*this)(2) + other(2, 0);
            return result;
        }

        template <typename OtherType>
        Vec3 operator-(const Mat<OtherType, 3, 1> &other) const
        {
            Vec3 result;
            result(0) = (*this)(0) - other(0, 0);
            result(1) = (*this)(1) - other(1, 0);
            result(2) = (*this)(2) - other(2, 0);
            return result;
        }

        template <typename OtherType>
        Vec3 operator*=(const OtherType &other) const
        {
            Vec3 result;
            result(0) = (*this)(0) * other;
            result(1) = (*this)(1) * other;
            result(2) = (*this)(2) * other;
            return result;
        }

        template <typename OtherType>
        Vec3 operator/=(const OtherType &other) const
        {
            Vec3 result;
            result(0) = (*this)(0) / other;
            result(1) = (*this)(1) / other;
            result(2) = (*this)(2) / other;
            return result;
        }

        Vec3 operator-() const
        {
            Vec3 result;
            result(0) = -(*this)(0);
            result(1) = -(*this)(1);
            result(2) = -(*this)(2);
            return result;
        }

        // template <typename OtherType>
        Type dot(const Vec3 &other)
        {
            return (*this)(0) * other(0) + (*this)(1) * other(1) + (*this)(2) * other(2);
        }
        */

        template <typename Type2>
        Vec3 cross(const Vec<Type2, 3> &other) const
        {
            Vec3 result;
            result(0) = (*this)(1) * other(2) - (*this)(2) * other(1);
            result(1) = (*this)(2) * other(0) - (*this)(0) * other(2);
            result(2) = (*this)(0) * other(1) - (*this)(1) * other(0);
            return result;
        }

        Vec3 normalized() const
        {
            Type norm = this->norm();
            return Vec3((*this)(0) / norm, (*this)(1) / norm, (*this)(2) / norm);
        }
    };

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vec4 : public Vec<Type, 4>
    {
    public:
        Vec4() : Vec<Type, 4>() {}
        Vec4(const Mat<Type, 4, 1> &mat)
            : Mat<Type, 4, 1>(mat) // call the base-class copy constructor
        {
        }

        Vec4(Type x, Type y, Type z, Type w)
        {
            (*this)(0) = x;
            (*this)(1) = y;
            (*this)(2) = z;
            (*this)(3) = w;
        }

        /*
        Vec4(Vec3<Type> v, Type w)
        {
            (*this)(0) = v(0);
            (*this)(1) = v(1);
            (*this)(2) = v(2);
            (*this)(3) = w;
        }

        Vec2<Type> xy()
        {
            return Vec2<Type>((*this)(0), (*this)(1));
        }

        operator Vec3<Type>() const
        {
            return Vec3<Type>((*this)(0), (*this)(1), (*this)(2));
        }
        */
    };

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vec5 : public Vec<Type, 5>
    {
    public:
        Vec5() : Vec<Type, 5>() {}
        Vec5(Type x, Type y, Type z, Type a, Type b)
        {
            (*this)(0) = x;
            (*this)(1) = y;
            (*this)(2) = z;
            (*this)(3) = a;
            (*this)(4) = b;
        }
        Vec5(const Mat<Type, 5, 1> &vec)
            : Vec<Type, 5>(vec) // call the base-class copy constructor
        {
        }
    };

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vec6 : public Vec<Type, 6>
    {
    public:
        Vec6() : Vec<Type, 6>() {}
        Vec6(Type x, Type y, Type z, Type a, Type b, Type c)
        {
            (*this)(0) = x;
            (*this)(1) = y;
            (*this)(2) = z;
            (*this)(3) = a;
            (*this)(4) = b;
            (*this)(5) = c;
        }
        Vec6(const Mat<Type, 6, 1> &vec)
            : Vec<Type, 6>(vec) // call the base-class copy constructor
        {
        }
    };

    template <typename Type, VecOrient Orient = VecOrient::Column>
    class Vec8 : public Vec<Type, 8>
    {
    public:
        Vec8() : Vec<Type, 8>() {}
        Vec8(Type x, Type y, Type z, Type a, Type b, Type c, Type d, Type e)
        {
            (*this)(0) = x;
            (*this)(1) = y;
            (*this)(2) = z;
            (*this)(3) = a;
            (*this)(4) = b;
            (*this)(5) = c;
            (*this)(6) = d;
            (*this)(7) = e;
        }
    };

    //============================================================
    // Common small matrix specializations
    //============================================================

    template <typename Type>
    class Mat2 : public Mat<Type, 2, 2>
    {
    public:
        Mat2() : Mat<Type, 2, 2>() {}
        Mat2(const Mat<Type, 2, 2> &mat)
            : Mat<Type, 2, 2>(mat) // call the base-class copy constructor
        {
        }
    };

    template <typename Type>
    class Mat3 : public Mat<Type, 3, 3>
    {
    public:
        Mat3() : Mat<Type, 3, 3>()
        {
            // #pragma HLS ARRAY_PARTITION variable = Mat < Type, 3, 3> ::data_ dim = 1 type = complete
        }

        Mat3(const Mat<Type, 3, 3> &mat)
            : Mat<Type, 3, 3>(mat) // call the base-class copy constructor
        {
        }

        template <class OtherType>
        Mat3(const OtherType mat[9])
            : Mat<Type, 3, 3>(mat) // call the base-class copy constructor
        {
        }

        template <class OtherType>
        Vec3<Type> operator*(const Vec3<OtherType> &rhs) const
        {
            // #pragma HLS INLINE
            Vec3<Type> result;
            result(0) = rhs(0) * (*this)(0, 0) + rhs(1) * (*this)(0, 1) + rhs(2) * (*this)(0, 2);
            result(1) = rhs(0) * (*this)(1, 0) + rhs(1) * (*this)(1, 1) + rhs(2) * (*this)(1, 2);
            result(2) = rhs(0) * (*this)(2, 0) + rhs(1) * (*this)(2, 1) + rhs(2) * (*this)(2, 2);
            return result;
        }

        Mat3 operator*(const Mat3 &rhs) const
        {
            // #pragma HLS INLINE
            return Mat<Type, 3, 3>::operator*(rhs);
        }

        // 3×3 determinant
        Type determinant() const
        {
            // #pragma HLS allocation operation instances = add limit = 1
            // #pragma HLS allocation operation instances = sub limit = 1
            // #pragma HLS allocation operation instances = mul limit = 1
            // #pragma HLS allocation operation instances = div limit = 1
            // #pragma HLS allocation operation instances = fadd limit = 1
            // #pragma HLS allocation operation instances = fsub limit = 1
            // #pragma HLS allocation operation instances = fmul limit = 1
            // #pragma HLS allocation operation instances = fdiv limit = 1

            // HLS_INLINE
            //  HLS_PIPELINE
            // #pragma HLS INLINE
            // #pragma HLS PIPELINE II = 1
            Type result = (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) + (*this)(0, 1) * ((*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2)) + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
            // Type aux1 = (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1));
            // Type aux1_1 = (*this)(1, 1) * (*this)(2, 2);
            // Type aux1_2 = (*this)(1, 2) * (*this)(2, 1);
            // Type aux1_3 = aux1_1 - aux1_2;
            // Type aux1 = (*this)(0, 0) * aux1_3;
            // Type aux2 = -(*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0));
            // Type aux2_1 = (*this)(1, 0) * (*this)(2, 2);
            // Type aux2_2 = (*this)(1, 2) * (*this)(2, 0);
            // Type aux2_3 = aux2_2 - aux2_1;
            // Type aux2 = (*this)(0, 1) * aux2_3;
            // Type aux3 = (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
            // Type aux3_1 = (*this)(1, 0) * (*this)(2, 1);
            // Type aux3_2 = (*this)(1, 1) * (*this)(2, 0);
            // Type aux3_3 = aux3_1 - aux3_2;
            // Type aux3 = (*this)(0, 2) * aux3_3;
            // Type aux4 = aux1 + aux2;
            // Type result = aux4 + aux3;
            return result;
        }

        // Inverse of 3×3
        Mat3 inverse() const
        {
            // #pragma HLS PIPELINE II = 1
            // #pragma HLS allocation operation instances = add limit = 1
            // #pragma HLS allocation operation instances = sub limit = 1
            // #pragma HLS allocation operation instances = mul limit = 1
            // #pragma HLS allocation operation instances = div limit = 1
            // #pragma HLS allocation operation instances = fadd limit = 1
            // #pragma HLS allocation operation instances = fsub limit = 1
            // #pragma HLS allocation operation instances = fmul limit = 1
            // #pragma HLS allocation operation instances = fdiv limit = 1

            Mat3 inv;
            // const Mat3<Type> &m = (*this);

            Type det = determinant();
            /*
            if (std::fabs(det) < Type(1e-12))
            {
                throw std::runtime_error("Encountered near-zero determinant in Mat3::inverse()");
            }
            */

            Type invDet = Type(1.0f) / det;

            Type aux00_1 = (*this)(1, 1) * (*this)(2, 2);
            Type aux00_2 = (*this)(1, 2) * (*this)(2, 1);
            inv(0, 0) = (aux00_1 - aux00_2) * invDet;
            inv(0, 1) = ((*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2)) * invDet;
            inv(0, 2) = ((*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1)) * invDet;

            inv(1, 0) = ((*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2)) * invDet;
            inv(1, 1) = ((*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0)) * invDet;
            inv(1, 2) = ((*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2)) * invDet;

            inv(2, 0) = ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0)) * invDet;
            inv(2, 1) = ((*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1)) * invDet;
            inv(2, 2) = ((*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0)) * invDet;

            return inv;
        }
    };

    template <typename Type>
    class Mat4 : public Mat<Type, 4, 4>
    {
    public:
        Mat4() : Mat<Type, 4, 4>() {}
        Mat4(const Mat<Type, 4, 4> &mat)
            : Mat<Type, 4, 4>(mat) // call the base-class copy constructor
        {
            // #pragma HLS ARRAY_PARTITION variable = Mat < Type, 4, 4> ::data_ dim = 1 type = complete
        }

        template <class OtherType>
        Vec4<Type> operator*(const Vec4<OtherType> &rhs) const
        {
            // #pragma HLS inline off

            /*
            Vec4<Type> result;
            result(0) = rhs(0) * (*this)(0, 0) + rhs(1) * (*this)(0, 1) + rhs(2) * (*this)(0, 2) + rhs(3) * (*this)(0, 3);
            result(1) = rhs(0) * (*this)(1, 0) + rhs(1) * (*this)(1, 1) + rhs(2) * (*this)(1, 2) + rhs(3) * (*this)(1, 3);
            result(2) = rhs(0) * (*this)(2, 0) + rhs(1) * (*this)(2, 1) + rhs(2) * (*this)(2, 2) + rhs(3) * (*this)(2, 3);
            result(3) = rhs(0) * (*this)(3, 0) + rhs(1) * (*this)(3, 1) + rhs(2) * (*this)(3, 2) + rhs(3) * (*this)(3, 3);
            return result;
            */

            Vec4<Type> result;
        mat4_mult_loop_c:
            for (int c = 0; c < 4; c++)
            {
                Type acc = Type(0);
                // #pragma HLS PIPELINE off
            mat4_mult_loop_k:
                for (int k = 0; k < 4; k++)
                {
                    // #pragma HLS PIPELINE off
                    acc += (*this)(c, k) * rhs(k);
                }
                result(c) = acc;
            }

            return result;
        }

        Mat4 operator*(const Mat4 &rhs) const
        {
            // #pragma HLS inline
            //  #pragma HLS INLINE
            // return Mat<Type, 4, 4>::operator*(rhs);

            // #pragma HLS INLINE off

            Mat4 result;
        mat4_mult_loop_r:
            for (int r = 0; r < 4; r++)
            {
                // #pragma HLS PIPELINE off
            mat4_mult_loop_c:
                for (int c = 0; c < 4; c++)
                {
                    Type acc = Type(0.0f);
                    // #pragma HLS PIPELINE off
                mat4_mult_loop_k:
                    for (int k = 0; k < 4; k++)
                    {
                        // #pragma HLS PIPELINE off
                        acc += (*this)(r, k) * rhs(k, c);
                    }
                    result(r, c) = acc;
                }
            }
            return result;
        }
    };

    template <typename Type>
    class Mat6 : public Mat<Type, 6, 6>
    {
    public:
        Mat6() : Mat<Type, 6, 6>() {}
        Mat6(const Mat<Type, 6, 6> &mat)
            : Mat<Type, 6, 6>(mat) // call the base-class copy constructor
        {
        }
    };

    template <typename Type>
    class Mat8 : public Mat<Type, 8, 8>
    {
    public:
        Mat8() : Mat<Type, 8, 8>() {}
    };

    //============================================================
    // Basic Quaternion
    //============================================================

    template <typename Type>
    class Quaternion
    {
    public:
        Quaternion() : w_(Type(1)), x_(Type(0)), y_(Type(0)), z_(Type(0)) {}

        template <typename T2>
        Quaternion(T2 w, T2 x, T2 y, T2 z)
            : w_(w), x_(x), y_(y), z_(z)
        {
        }

        Mat3<Type> matrix() const
        {
            Mat3<Type> matrix_;

            // Using the standard formula
            Type xx = Type(2) * x_ * x_;
            Type yy = Type(2) * y_ * y_;
            Type zz = Type(2) * z_ * z_;
            Type xy = Type(2) * x_ * y_;
            Type xz = Type(2) * x_ * z_;
            Type yz = Type(2) * y_ * z_;
            Type wx = Type(2) * w_ * x_;
            Type wy = Type(2) * w_ * y_;
            Type wz = Type(2) * w_ * z_;

            matrix_(0, 0) = Type(1) - (yy + zz);
            matrix_(0, 1) = xy - wz;
            matrix_(0, 2) = xz + wy;

            matrix_(1, 0) = xy + wz;
            matrix_(1, 1) = Type(1) - (xx + zz);
            matrix_(1, 2) = yz - wx;

            matrix_(2, 0) = xz - wy;
            matrix_(2, 1) = yz + wx;
            matrix_(2, 2) = Type(1) - (xx + yy);

            return matrix_;
        }

        // Multiply quaternion by another quaternion
        Quaternion operator*(const Quaternion &q) const
        {
            // (w1, x1, y1, z1) * (w2, x2, y2, z2)
            // = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            //    w1*x2 + x1*w2 + y1*z2 - z1*y2,
            //    w1*y2 - x1*z2 + y1*w2 + z1*x2,
            //    w1*z2 + x1*y2 - y1*x2 + z1*w2)
            return Quaternion(
                w_ * q.w_ - x_ * q.x_ - y_ * q.y_ - z_ * q.z_,
                w_ * q.x_ + x_ * q.w_ + y_ * q.z_ - z_ * q.y_,
                w_ * q.y_ - x_ * q.z_ + y_ * q.w_ + z_ * q.x_,
                w_ * q.z_ + x_ * q.y_ - y_ * q.x_ + z_ * q.w_);
        }

        // Rotate a 3D vector by this quaternion (assumed normalized)
        Vec3<Type> operator*(const Vec3<Type> &v) const
        {
            // Convert v to a pure quaternion with zero real part
            Quaternion vq(Type(0), v(0), v(1), v(2));

            // q * vq
            Quaternion qv = (*this) * vq;
            // conj(q) = (w, -x, -y, -z)
            Quaternion qc(w_, -x_, -y_, -z_);

            // rotated = qv * conj(q)
            Quaternion rotated = qv * qc;

            // The imaginary part is the rotated vector
            return Vec3<Type>(rotated.x_, rotated.y_, rotated.z_);
        }

        // Inverse (for normalized quaternions, inverse = conjugate)
        Quaternion inverse() const
        {
            Type normSq = w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_;
            // if (std::fabs(normSq) < Type(1e-12))
            //     throw std::runtime_error("Near-zero norm in Quaternion::inverse()");

            Type inv = Type(1) / normSq;
            return Quaternion(w_ * inv, -x_ * inv, -y_ * inv, -z_ * inv);
        }

        Type x() const { return x_; }
        Type y() const { return y_; }
        Type z() const { return z_; }
        Type w() const { return w_; }

        Type &x() { return x_; }
        Type &y() { return y_; }
        Type &z() { return z_; }
        Type &w() { return w_; }

    private:
        // Public components for convenience
        Type w_, x_, y_, z_;
    };

    //============================================================
    // SO3 class (3D rotation), stored as a 3×3 matrix
    //============================================================

    template <typename Type>
    class SO3
    {
    public:
        SO3()
        {
            quaternion_ = Quaternion<Type>(Type(1), Type(0), Type(0), Type(0));
        }

        SO3(Type qw, Type qx, Type qy, Type qz)
        {
            quaternion_ = Quaternion<Type>(qw, qx, qy, qz);
        }

        SO3(Quaternion<Type> q)
        {
            quaternion_ = q;
        }

        SO3(const Mat3<Type> &R)
        {
            // Convert rotation matrix to quaternion using Shepperd's method
            Type trace = R(0, 0) + R(1, 1) + R(2, 2);

            if (trace > Type(0))
            {
                Type s = math::sqrt(trace + Type(1)) * Type(2); // s = 4 * qw
                quaternion_.w() = Type(0.25) * s;
                quaternion_.x() = (R(2, 1) - R(1, 2)) / s;
                quaternion_.y() = (R(0, 2) - R(2, 0)) / s;
                quaternion_.z() = (R(1, 0) - R(0, 1)) / s;
            }
            else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2))
            {
                Type s = math::sqrt(Type(1) + R(0, 0) - R(1, 1) - R(2, 2)) * Type(2); // s = 4 * qx
                quaternion_.w() = (R(2, 1) - R(1, 2)) / s;
                quaternion_.x() = Type(0.25) * s;
                quaternion_.y() = (R(0, 1) + R(1, 0)) / s;
                quaternion_.z() = (R(0, 2) + R(2, 0)) / s;
            }
            else if (R(1, 1) > R(2, 2))
            {
                Type s = math::sqrt(Type(1) + R(1, 1) - R(0, 0) - R(2, 2)) * Type(2); // s = 4 * qy
                quaternion_.w() = (R(0, 2) - R(2, 0)) / s;
                quaternion_.x() = (R(0, 1) + R(1, 0)) / s;
                quaternion_.y() = Type(0.25) * s;
                quaternion_.z() = (R(1, 2) + R(2, 1)) / s;
            }
            else
            {
                Type s = math::sqrt(Type(1) + R(2, 2) - R(0, 0) - R(1, 1)) * Type(2); // s = 4 * qz
                quaternion_.w() = (R(1, 0) - R(0, 1)) / s;
                quaternion_.x() = (R(0, 2) + R(2, 0)) / s;
                quaternion_.y() = (R(1, 2) + R(2, 1)) / s;
                quaternion_.z() = Type(0.25) * s;
            }
        }

        SO3(const SO3 &other)
        {
            // matrix_ = other.matrix_;
            quaternion_ = other.quaternion_;
        }

        void setQuaternion(const Quaternion<Type> &q)
        {
            // fromQuaternion(q.w_, q.x_, q.y_, q.z_);
            quaternion_ = q;
        }

        const Quaternion<Type> &unit_quaternion() const
        {
            return quaternion_;
        }

        Quaternion<Type> &unit_quaternion()
        {
            return quaternion_;
        }

        Mat3<Type> matrix() const
        {
            return quaternion_.matrix();
        }

        static SO3 exp(const Vec3<Type> &phi)
        {
            Type angle = phi.norm();
            if (angle < Type(1e-12))
            {
                // Near zero, use approximation: exp(phi) ~ I + wedge(phi)
                Mat3<Type> approx = Mat3<Type>::Identity() + wedge(phi);
                return SO3(approx);
            }

            Vec3<Type> axis = (phi / angle);
            Type s = math::sin(angle);
            Type c = math::cos(angle);

            // Rodrigues' formula: R = I c + (1-c) (axis axis^T) + [axis]_x s
            Mat3<Type> R = Mat3<Type>::Identity() * c + outerProduct(axis, axis) * (Type(1) - c) + wedge(axis) * s;

            return SO3(R);
        }

        Vec3<Type> log() const
        {
            const Quaternion<Type> &q = quaternion_;

            Type qw = q.w();
            Type qx = q.x();
            Type qy = q.y();
            Type qz = q.z();

            // Vector part magnitude = sin(theta/2)
            Type sin_half_theta = math::sqrt(qx * qx + qy * qy + qz * qz);

            // Handle the small-angle case separately to avoid division by zero
            const Type eps = Type(1e-12);

            if (sin_half_theta < eps)
            {
                // For very small angles:
                // q ≈ [1, 0.5 * phi]  =>  phi ≈ 2 * v
                return Vec3<Type>(Type(2) * qx, Type(2) * qy, Type(2) * qz);
            }

            // General case
            // theta = 2 * atan2(||v||, w)
            Type theta = Type(2) * math::atan2(sin_half_theta, qw);

            // Axis = v / sin(theta/2)
            // phi = theta * axis = theta / sin(theta/2) * v
            Type k = theta / sin_half_theta;

            return Vec3<Type>(qx * k, qy * k, qz * k);
        }

        /*
        // Convert quaternion -> 3×3 rotation (assuming unit quaternion)
        void fromQuaternion(Type qw, Type qx, Type qy, Type qz)
        {
            // Using the standard formula
            Type xx = Type(2) * qx * qx;
            Type yy = Type(2) * qy * qy;
            Type zz = Type(2) * qz * qz;
            Type xy = Type(2) * qx * qy;
            Type xz = Type(2) * qx * qz;
            Type yz = Type(2) * qy * qz;
            Type wx = Type(2) * qw * qx;
            Type wy = Type(2) * qw * qy;
            Type wz = Type(2) * qw * qz;

            matrix_(0, 0) = Type(1) - (yy + zz);
            matrix_(0, 1) = xy - wz;
            matrix_(0, 2) = xz + wy;

            matrix_(1, 0) = xy + wz;
            matrix_(1, 1) = Type(1) - (xx + zz);
            matrix_(1, 2) = yz - wx;

            matrix_(2, 0) = xz - wy;
            matrix_(2, 1) = yz + wx;
            matrix_(2, 2) = Type(1) - (xx + yy);
        }
        */

        // Identity
        void setIdentity()
        {
            // matrix_ = Mat3<Type>::Identity();
            quaternion_ = Quaternion<Type>(Type(1), Type(0), Type(0), Type(0));
        }

        // Inverse = transpose for rotation matrix
        SO3<Type> inverse() const
        {
            // return SO3<Type>(matrix_.transpose());
            return SO3<Type>(quaternion_.inverse());
        }

        // Element access
        // Type operator()(int r, int c) const { return matrix_(r, c); }
        // Type &operator()(int r, int c) { return matrix_(r, c); }

        Vec3<Type> operator*(const Vec3<Type> &v) const
        {
            return quaternion_ * v;
        }

        SO3 operator*(const SO3 &rhs) const
        {
            return SO3(quaternion_ * rhs.quaternion_);
        }

        // Get the underlying 3×3
        // const Mat3<Type> &matrix() const { return matrix_; }
        // Mat3<Type> &matrix() { return matrix_; }

    private:
        Quaternion<Type> quaternion_;
    };

    //============================================================
    // Exponential map for so(3) -> SO3
    //    exp: R^3 (axis*angle) -> 3×3 rotation
    //============================================================
    // A helper for the skew-symmetric matrix ("wedge")
    template <typename Type>
    Mat3<Type> wedge(const Vec3<Type> &phi)
    {
        Mat3<Type> w = Mat3<Type>::Zero();
        w(0, 0) = Type(0);
        w(0, 1) = -phi(2);
        w(0, 2) = phi(1);
        w(1, 0) = phi(2);
        w(1, 1) = Type(0);
        w(1, 2) = -phi(0);
        w(2, 0) = -phi(1);
        w(2, 1) = phi(0);
        w(2, 2) = Type(0);
        return w;
    }

    // Outer product of two 3D vectors -> 3×3 matrix
    template <typename Type>
    Mat3<Type> outerProduct(const Vec3<Type> &a, const Vec3<Type> &b)
    {
        Mat3<Type> m; // = Mat3<Type>::Zero();
    mat3_out_loop_r:
        for (int r = 0; r < 3; r++)
        mat_out_loop_c:
            for (int c = 0; c < 3; c++)
                m(r, c) = a(r) * b(c);
        return m;
    }

    //============================================================
    // Left Jacobian of SO3
    //   This is a 3×3 matrix used in many Lie-theory-based derivations
    //============================================================
    template <typename Type>
    Mat3<Type> so3LeftJacobian(const Vec3<Type> &phi)
    {
        Type angle = phi.norm();
        if (angle < Type(1e-12))
        {
            // Approx: J_l(phi) ~ I + 0.5 [phi]_x
            Mat3<Type> result = Mat3<Type>::Identity() + wedge(phi) * Type(0.5);
            return result;
        }

        Vec3<Type> axis = (phi / angle);
        Type s = math::sin(angle);
        Type c = math::cos(angle);

        Mat3<Type> I = Mat3<Type>::Identity();
        Mat3<Type> K = wedge(axis);
        Mat3<Type> aaT = outerProduct(axis, axis);

        // Formula: J_l(phi) = I + ( (1 - c)/angle^2 ) [phi]_x + (angle - s)/angle^3 [phi]_x^2
        // But one commonly used expression is:
        // (s/angle)*I + (1 - s/angle)*aaT + ((1 - c)/angle)*K
        // Both are valid forms. Use whichever is standard for you:

        Mat3<Type> term1 = I * (s / angle);
        Mat3<Type> term2 = aaT * (Type(1) - s / angle);
        Mat3<Type> term3 = K * ((Type(1) - c) / angle);

        return (term1 + term2 + term3);
    }

    //============================================================
    // SE3 class = (SO3 rotation, R^3 translation)
    //============================================================
    template <typename Type>
    class SE3
    {
    public:
        SE3()
        {
            so3_ = SO3<Type>();
            trans_ = Vec3<Type>(Type(0), Type(0), Type(0));
        }

        SE3(const SO3<Type> &r, const Vec3<Type> &t)
        {
            so3_ = r;
            trans_ = t;
        }

        SE3(const Mat3<Type> &R, const Vec3<Type> &t)
        {
            so3_ = SO3<Type>(R);
            trans_ = t;
        }

        SE3(const Mat4<Type> &T)
        {
            Mat3<Type> R;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R(r, c) = T(r, c);
            Vec3<Type> t;
            for (int r = 0; r < 3; r++)
                t(r) = T(r, 3);

            so3_ = SO3<Type>(R);
            trans_ = t;
        }

        SE3(Type *r_data, Type *t_data)
        {
            so3_ = SO3<Type>(r_data[0], r_data[1], r_data[2], r_data[3]);
            trans_ = Vec3<Type>(t_data[0], t_data[1], t_data[2]);
        }

        void setQuaternion(const Quaternion<Type> &q)
        {
            so3_.setQuaternion(q);
        }

        Mat4<Type> matrix() const
        {
            Mat4<Type> mat; // = Mat4<Type>::Zero();
            Mat3<Type> R = so3_.matrix();

        se3_matrix_loop_r:
            for (int r = 0; r < 3; r++)
            {
                mat(r, 3) = trans_(r);
                mat(3, r) = Type(0);
            se3_matrix_loop_c:
                for (int c = 0; c < 3; c++)
                {
                    mat(r, c) = R(r, c);
                }
            }
            mat(3, 3) = Type(1);

            /*
            mat(0, 0) = R(0, 0);
            mat(1, 0) = R(1, 0);
            mat(2, 0) = R(2, 0);
            mat(3, 0) = Type(0);

            mat(0, 1) = R(0, 1);
            mat(1, 1) = R(1, 1);
            mat(2, 1) = R(2, 1);
            mat(3, 1) = Type(0);

            mat(0, 2) = R(0, 2);
            mat(1, 2) = R(1, 2);
            mat(2, 2) = R(2, 2);
            mat(3, 2) = Type(0);

            mat(0, 3) = trans_(0);
            mat(1, 3) = trans_(1);
            mat(2, 3) = trans_(2);
            mat(3, 3) = Type(1);
            */

            return mat;
        }

        Vec3<Type> operator*(const Vec3<Type> &p) const
        {
            return so3_ * p + trans_;
        }

        SE3 operator*(const SE3 &rhs) const
        {
            // [R1|t1] [R2|t2] = [R1R2 | R1 t2 + t1]

            SE3<Type> out;
            out.so3_ = so3_ * rhs.so3_;
            out.trans_ = so3_ * rhs.trans_ + trans_;
            return out;
        }

        SE3<Type> inverse() const
        {
            // Inv( [R|t] ) = [R^T | -R^T t]

            SE3<Type> inv;
            inv.so3_ = so3_.inverse();
            inv.trans_ = inv.so3_ * (trans_ * Type(-1));
            return inv;
        }

        //============================================================
        // Exponential map SE3: R^6 -> SE3
        //   xi = (rho, phi) in R^3 x R^3
        //============================================================
        static SE3 exp(const Vec6<Type> &xi)
        {
            // xi = (rho, phi), each 3D
            Vec3<Type> rho(xi(0), xi(1), xi(2));
            Vec3<Type> phi(xi(3), xi(4), xi(5));

            // Rotation part
            SO3<Type> R = SO3<Type>::exp(phi);

            // Translation part: J_l(phi) * rho
            Mat3<Type> J = so3LeftJacobian(phi);
            Vec3<Type> t = J * rho;

            return SE3(R, t);
        }

        //============================================================
        // Logarithm map SE3: SE3 -> R^6
        //   xi = (rho, phi) in R^3 x R^3
        //   where R = exp(phi^) and t = J_l(phi) * rho
        //============================================================
        Vec6<Type> log() const
        {
            Vec6<Type> xi;

            // Rotation part
            Vec3<Type> phi = so3_.log();

            // Translation part: rho = J_l(phi)^{-1} * t
            Mat3<Type> J = so3LeftJacobian(phi);

            // Simple & general: use matrix inverse (if you don't have a dedicated J^{-1})
            Mat3<Type> J_inv = J.inverse();
            Vec3<Type> rho = J_inv * trans_;

            // Pack into se(3) vector: (rho, phi)
            xi(0) = rho(0);
            xi(1) = rho(1);
            xi(2) = rho(2);
            xi(3) = phi(0);
            xi(4) = phi(1);
            xi(5) = phi(2);

            return xi;
        }

        SO3<Type> &so3() { return so3_; }
        const SO3<Type> &so3() const { return so3_; }
        Vec3<Type> &translation() { return trans_; }
        const Vec3<Type> &translation() const { return trans_; }

    private:
        SO3<Type> so3_;
        Vec3<Type> trans_;
    };

} // namespace linalg