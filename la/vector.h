#ifndef TMS_SUBMOD_VECTOR_H
#define TMS_SUBMOD_VECTOR_H

#include <list>
#include <assert.h>
#include "mkl.h"

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
    int64_t _base_len;
    bool _mem_manage;

    Vector(int64_t s) : _len(s), _base_len(s), _stride(1), _mem_manage(true)
    {
        auto ret = posix_memalign((void **) &_values, 4096, _len * sizeof(DT));
        if(ret != 0){
            std::cout << "Could not allocate memory for vector of length " << _len *sizeof(DT) / 1e9 << " GB. Exiting..." << std::endl;
            exit(1);
        }
    }

    Vector(DT* values, int64_t len, int64_t base_len, int64_t stride, bool mem_manage) :
        _values(values), _len(len), _base_len(base_len), _stride(stride), _mem_manage(mem_manage)
    { }

    ~Vector()
    {
        if(_mem_manage) {
            free(_values);
        }
    }

    Vector& operator=(Vector&& x) 
    {
        _len = x._len;
        _base_len = x._base_len
        _stride = x._stride;
        _mem_manage = x._mem_manage;
        _values = x._values;
        x._values = NULL;
    }

    Vector(Vector&& x) :
        _values(x._values), _mem_manage(x._mem_manage),
        _len(x._len), _base_len(x._base_len), _stride(x._stride)
    {
        x._values = NULL;
    }

    Vector& operator=(const Vector& x)
    {
        _len = x._len;
        _base_len = x._len;
        _stride = 1;
        _mem_manage = true;

        auto ret = posix_memalign((void **) &_values, 4096, _len * sizeof(DT));
        if(ret != 0){
            std::cout << "Could not allocate memory for vector of length " << _len *sizeof(DT) / 1e9 << " GB. Exiting..." << std::endl;
            exit(1);
        }
        this->copy(x);
    }
    Vector(const Vector& x) :
        _len(x._len), _base_len(x._len), _stride(1), _mem_manage(true)
    {
        auto ret = posix_memalign((void **) &_values, 4096, _len * sizeof(DT));
        if(ret != 0){
            std::cout << "Could not allocate memory for vector of length " << _len *sizeof(DT) / 1e9 << " GB. Exiting..." << std::endl;
            exit(1);
        }
        this->copy(x);
    }


    void realloc(int64_t len) 
    {
        //Can only reallocate "base object"
        assert(_mem_manage == true);
        assert(len >= _len);
        assert(_base_len == _len);
        
        //Allocate new array
        DT* array;
        const int ret = posix_memalign((void **) &array, 4096, len * sizeof(DT));
        if (ret != 0) {
            std::cout << "Could not allocate memory for Vector. Exiting ..." << std::endl;
            exit(1);
        }
        
        //Copy old array over
        Vector<DT> tmp(array, len, len, 1, false);
        auto tmp_partition = tmp.subvector(0,_len);
        tmp_partition.copy(*this);

        //Free old array
        free(_values); 

        //Setup fields
        _values = array;
        _len = len;
        _stride = 1;
        _base_len = len;
    }

    Vector<DT> subvector(int64_t start, int64_t blksz)
    {
        assert(start < _len && "Vector index out of bounds.");
        auto length = std::min(blksz, _len - start);

        return Vector(_values + start*_stride, length, _len, _stride, false);
    }
    const Vector<DT> subvector(int64_t start, int64_t blksz) const
    {
        assert(start < _len && "Vector index out of bounds.");
        auto length = std::min(blksz, _len - start);

        return Vector(_values + start*_stride, length, _len, _stride, false);
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

    inline DT* lea (int64_t index)
    {
        assert(index < _len && "Vector index out of bounds.");
        return &_values[index * _stride];
    }

    inline int64_t length() const
    {
        return _len;
    }

    void set_all(DT alpha){
        for(int i = 0; i < _len; i++) {
            _values[i*_stride] = alpha;
        }
    }

    template<class RNG, class DIST>
    void fill_rand(RNG &gen, DIST &dist) {
        for(int64_t i = 0; i < _len; i++) {
            (*this)(i) = dist(gen);
        }
    }

    void fill_rand() {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::normal_distribution<> normal(0.0, 1.0);
        this->fill_rand(gen, normal);
    }

    void print(std::string name) const
    {
        std::cout << name << std::endl;
        for(int i = 0; i < _len; i++){
            std::cout << _values[i*_stride] << std::endl;
        }
        std::cout << std::endl;
    }

    void print() const
    {
        for(int i = 0; i < _len; i++){
            std::cout << _values[i*_stride] << std::endl;
        }
        std::cout << std::endl;
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
        return _values[index_of_min()*_stride];
    }
    DT max() const
    {
        return _values[index_of_max()*_stride];
    }
    int64_t index_of_max() const
    {
        int64_t index = 0;
        DT max = _values[0*_stride];
        for(int i = 1; i < _len; i++) {
            if(_values[i*_stride] > max) {
                max = _values[i*_stride];
                index = i;
            }
        }
        return index;
    }
    int64_t index_of_min() const
    {
        int64_t index = 0;
        DT min = _values[0*_stride];
        for(int i = 1; i < _len; i++) {
            if(_values[i*_stride] < min) {
                min = _values[i*_stride];
                index = i;
            }
        }
        return index;
    }
    DT abs_min() const
    {
        return std::abs(_values[index_of_abs_min()*_stride]);
    }
    DT abs_max() const
    {
        return std::abs(_values[index_of_abs_max()*_stride]);
    }
    int64_t index_of_abs_min() const
    {
        int64_t index = 0;
        DT abs_min = _values[0*_stride];
        for(int i = 1; i < _len; i++) {
            if(std::abs(_values[i*_stride]) < abs_min) {
                abs_min = std::abs(_values[i*_stride]);
                index = i;
            }
        }

        return index;
    }

    int64_t index_of_abs_max() const
    {
        int64_t index = 0;
        DT abs_max = _values[0*_stride];
        for(int i = 1; i < _len; i++) {
            if(std::abs(_values[i*_stride]) > abs_max) {
                abs_max = std::abs(_values[i*_stride]);
                index = i;
            }
        }

        return index;
    }

    bool has_nan() const
    {
        for(int i = 0; i < _len; i++) {
            DT alpha = (*this)(i);
            if(alpha != alpha)
                return true;
        }
        return false;
    }

    void enlarge(int64_t l_inc) {
       assert(_len + l_inc <= _base_len);
       _len += l_inc; 
    }

    void remove_elems(std::list<int64_t> indices) {
        int64_t n_removed = 1;
        for(auto iter = indices.begin(); iter != indices.end(); iter++) {
            int64_t block_begin = *iter - (n_removed - 1);
            int64_t block_end = _len - n_removed;
            if(std::next(iter,1) != indices.end()) {
                block_end = *std::next(iter,1) - n_removed;
            }

            for(int64_t i = block_begin; i < block_end; i++) {
                (*this)(i) = (*this)(i+n_removed);
            }
            n_removed++;
        }
        _len -= indices.size();
    }

    void remove_elem(int64_t index) {
        if(index != _len - 1) {
            Vector<DT> t(_len - index - 1);
            const auto a2 = subvector(index+1, _len - index - 1);
            t.copy(a2);
            auto a1 = subvector(index, _len - index - 1);
            a1.copy(t);
        }
        _len--;
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
    void copy(const Vector<DT>& from) {
        assert(_len == from._len);
        for(int i = 0; i < _len; i++) {
            (*this)(i) = from(i);
        }
    }

    void scale(const DT alpha)
    {
        for(int i = 0; i < _len; i++) {
            (*this)(i) *= alpha;
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
        } else {
            DT alpha = -sgn(chi1) * nrm_x;
            DT mult = 1.0 / (chi1 - alpha);
            
            for(int64_t i = 1; i < _len; i++) {
                (*this)(i) *= mult;
            }

            tau = 1.0 /  (0.5 + 0.5 * nrm_x2_sqr * mult * mult);
            (*this)(0) = alpha;
        }

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

//
// Double precision vector ops
//
template<>
inline double Vector<double>::norm2() const
{
    double nrm = cblas_dnrm2( _len, _values, _stride);
    return nrm;
}
template<>
inline double Vector<double>::dot(const Vector<double>& other) const
{
    assert(_len == other.length());
    double alpha = cblas_ddot( _len, _values, _stride, other._values, other._stride);
    return alpha;
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
void Vector<double>::copy(const Vector<double>& from) {
    cblas_dcopy(_len, from._values, from._stride, _values, _stride);
}

//
// Single precision vector ops
//
template<>
inline float Vector<float>::norm2() const
{
    float nrm = cblas_snrm2( _len, _values, _stride);
    return nrm;
}
template<>
inline float Vector<float>::dot(const Vector<float>& other) const
{
    float alpha = cblas_sdot( _len, _values, _stride, other._values, other._stride);
    return alpha;
}
template<>
inline void Vector<float>::scale(const float alpha)
{
    cblas_sscal(_len, alpha, _values, _stride);
}
template<>
inline void Vector<float>::axpy(const float alpha, const Vector<float>& other)
{
    cblas_saxpy(_len, alpha, other._values, other._stride, _values, _stride);
}
template<>
inline void Vector<float>::axpby(const float alpha, const Vector<float>& other, const float beta)
{
    cblas_saxpby(_len, alpha, other._values, other._stride, beta, _values, _stride);
}

template<>
void Vector<float>::copy(const Vector<float>& from) {
    cblas_scopy(_len, from._values, from._stride, _values, _stride);
}

#endif
