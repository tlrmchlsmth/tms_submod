#ifndef TMS_SUBMOD_VECTOR_H
#define TMS_SUBMOD_VECTOR_H

#include "mkl.h"
#include <assert.h>

template<class DT> class Matrix;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<class DT>
class Vector {

public:
    DT * _values;
    int64_t _len;
    int64_t _stride;

    bool _mem_manage;

//public:
    Vector(int64_t s) : _len(s), _stride(1), _mem_manage(true)
    {
        const int ret = posix_memalign((void **) &_values, 4096, _len * sizeof(DT));
        assert(ret == 0 && "Could not allocate memory");
    }

    Vector(DT* values, int64_t len, int64_t stride, bool mem_manage) :
        _values(values), _len(len), _stride(stride), _mem_manage(mem_manage)
    {
    }

    ~Vector()
    {
        if(_mem_manage) {
            free(_values);
        }
    }

    Vector<DT> subvector(int64_t start, int64_t blksz)
    {
        assert(start < _len && "Vector index out of bounds.");
        auto length = std::min(blksz, _len - start);

        return Vector(_values + start*_stride, length, _stride, false);
    }
    const Vector<DT> subvector(int64_t start, int64_t blksz) const
    {
        assert(start < _len && "Vector index out of bounds.");
        auto length = std::min(blksz, _len - start);

        return Vector(_values + start*_stride, length, _stride, false);
    }

    inline DT& operator() (int64_t index)
    {
        assert(index < _len && "Vector index out of bounds");
        return _values[index * _stride];
    }

    inline DT operator() (int64_t index) const
    {
        assert(index < _len && "Vector index out of bounds.");
        return _values[index * _stride];
    }

    inline int64_t length()
    {
        return _len;
    }

    void set_all(DT alpha){
        for(int i = 0; i < _len; i++) {
            _values[i*_stride] = alpha;
        }
    }
    void zero_out() 
    {
        set_all(0.0);
    }

    template<class RNG, class DIST>
    void fill_rand(RNG &gen, DIST &dist) {
        for(int64_t i = 0; i < _len; i++) {
            (*this)(i) = dist(gen);
        }
    }

    void print() const
    {
        for(int i = 0; i < _len; i++){
            std::cout << _values[i*_stride] << std::endl;
        }
    }

    DT sum() const
    {
        DT sum = 0.0;
        for(int i = 0; i < _len; i++)
            sum += _values[i*_stride];
        return sum;
    }
    DT min() const
    {
        DT min = _values[0*_stride];
        for(int i = 1; i < _len; i++)
            min = std::min(min, _values[i*_stride]);
        return min;
    }
    DT max() const
    {
        DT max = _values[0*_stride];
        for(int i = 1; i < _len; i++)
            max = std::max(max, _values[i*_stride]);
        return max;
    }

    // Routines for later specialization
    DT norm2() const
    {
        std::cout << "norm2 not implemented for datatype" << std::endl;
        exit(1);
    }
    DT dot(const Vector<DT>& other) const
    {
        std::cout << "dot not implemented for datatype" << std::endl;
        exit(1);
    }
    void axpy(const DT alpha, const Vector<DT>& other)
    {
        std::cout << "axpy not implemented for datatype" << std::endl;
        exit(1);
    }
    void axpby(const DT alpha, const Vector<DT>& other, const DT beta)
    {
        std::cout << "axpby not implemented for datatype" << std::endl;
        exit(1);
    }
    void copy(const Vector<DT> from) {
        std::cout << "copy not implemented for datatype" << std::endl;
        exit(1);
    }

    void scale(const DT alpha)
    {
        for(int i = 0; i < _len; i++) {
            this(i) *= alpha;
        }
    }

    DT house_gen() {
        DT chi1 = (*this)(0);
        DT nrm_x2_sqr = 0.0;
        
        for(int64_t i = 1; i < _len; i++){
            nrm_x2_sqr += (*this)(i) * (*this)(i);
        }
        DT nrm_x  = sqrt(chi1*chi1 + nrm_x2_sqr);

        double tau = 0.5;
        if(nrm_x2_sqr == 0) {
            (*this)(0) = -chi1;
            return tau;
        }

        DT alpha = -sgn(chi1) * nrm_x;
        DT mult = 1.0 / (chi1 - alpha);
        
        for(int64_t i = 1; i < _len; i++) {
            (*this)(i) *= mult;
        }

        tau = 1.0 /  (0.5 + 0.5 * nrm_x2_sqr * mult * mult);
        (*this)(0) = alpha;
        return tau;
    }

    //Apply this householder transform to another vector.
    //Need to perform x - tau * v * (v'*x)
    //v[0] is implicitly 1.
    void house_apply(double tau, Vector<DT>& x) const 
    {
        //First perform v'*x
        DT vt_x = x(0);
        for(int i = 1; i < _len; i++) {
            vt_x += (*this)(i) * x(i);
        }

        DT alpha = tau * vt_x;
        x(0) -= alpha;
        for(int i = 1; i < _len, i++) {
            x(i) -= alpha * (*this)(i);
        }
    }

    inline void house_apply(DT tau, Matrix<DT>& X) const 
    {
        _Pragma("omp parallel for")
        for(int j = 0; j < X.width(); j++) {
            //First perform v'*x
            DT vt_x = X(0, j);
            for(int i = 1; i < _len; i++) {
                vt_x += (*this)(i) * X(i, j);
            }

            DT alpha = tau * vt_x;
            X(0, j) -= alpha;
            for(int i = 1; i < _len; i++) {
                X(i,j) -= alpha * (*this)(i);
            }
        }
    }

};

#include "matrix.h"
#include "immintrin.h"
#include "ipps.h"

template<>
inline double Vector<double>::norm2() const
{
   return cblas_dnrm2( _len, _values, _stride);
}
template<>
inline double Vector<double>::dot(const Vector<double>& other) const
{
   return cblas_ddot( _len, _values, _stride, other._values, other._stride);
}
template<>
inline void Vector<double>::scale(const double alpha)
{
    cblas_dscal(_len, alpha, _values, _stride);
}
template<>
inline void Vector<double>::axpy(const double alpha, const Vector<double>& other)
{
    cblas_daxpy(_len, alpha, other._values, other._stride, _values, _stride);
}
template<>
inline void Vector<double>::axpby(const double alpha, const Vector<double>& other, const double beta)
{
    cblas_daxpby(_len, alpha, other._values, other._stride, beta, _values, _stride);
}
template<>
void Vector<double>::copy(const Vector<double> from) {
    cblas_dcopy(_len, from._values, from._stride, _values, _stride);
}

template<>
inline void Vector<double>::house_apply(double tau, Matrix<double>& X) const {
    #pragma omp parallel for
    for(int j = 0; j < X.width(); j++) {
        //BLAS VERSION
//#define BLAS_HOUSE        
#ifdef BLAS_HOUSE
        double vt_x = X(0,j) + cblas_ddot(_len-1, &_values[1], _stride, &X._values[X._rs + j*X._cs], X._rs);
        double alpha = tau * vt_x;
        X(0, j) -= alpha;
        cblas_daxpy(_len-1, -alpha, &_values[1], _stride, &X._values[X._rs + j*X._cs], X._rs);
#else
        //IPP version
        double vt_x;
        ippsDotProd_64f(&_values[1], &X._values[1 + j*X._cs], _len-1, &vt_x);
        vt_x += X(0,j);
        double alpha = tau * vt_x;
        X(0, j) -= alpha;
        ippsAddProductC_64f(&_values[1], -alpha, &X._values[1 + j*X._cs], _len-1);
#endif
    }
}

#endif
