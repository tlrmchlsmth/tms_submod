#ifndef TMS_SUBMOD_VECTOR_H
#define TMS_SUBMOD_VECTOR_H

#include "mkl.h"
#include <assert.h>

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

    Vector<DT> subvector(int64_t start, int64_t length)
    {
        assert(start + length <= _len && "Subvector too long.");
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

};

template<>
double Vector<double>::norm2() const
{
   return cblas_dnrm2( _len, _values, _stride);
}
template<>
double Vector<double>::dot(const Vector<double>& other) const
{
   return cblas_ddot( _len, _values, _stride, other._values, other._stride);
}
template<>
void Vector<double>::scale(const double alpha)
{
    cblas_dscal(_len, alpha, _values, _stride);
}
template<>
void Vector<double>::axpy(const double alpha, const Vector<double>& other)
{
    cblas_daxpy(_len, alpha, other._values, other._stride, _values, _stride);
}
template<>
void Vector<double>::axpby(const double alpha, const Vector<double>& other, const double beta)
{
    cblas_daxpby(_len, alpha, other._values, other._stride, beta, _values, _stride);
}
template<>
void Vector<double>::copy(const Vector<double> from) {
    cblas_dcopy(_len, from._values, from._stride, _values, _stride);
}

#endif
