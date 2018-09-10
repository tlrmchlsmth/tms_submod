#ifndef TMS_SUBMOD_VECTOR_H
#define TMS_SUBMOD_VECTOR_H

#include <assert.h>
#include "mkl.h"
#include "perf_log.h"
#include "perf/perf.h"

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

    PerfLog* log;

    bool _mem_manage;

//public:
    Vector(int64_t s) : _len(s), _stride(1), _mem_manage(true), log(NULL)
    {
        auto ret = posix_memalign((void **) &_values, 4096, _len * sizeof(DT));
        if(ret != 0){
            std::cout << "Could not allocate memory for vector of length " << _len *sizeof(DT) / 1e9 << " GB. Exiting..." << std::endl;
            exit(1);
        }
    }

    Vector(DT* values, int64_t len, int64_t stride, bool mem_manage) :
        _values(values), _len(len), _stride(stride), _mem_manage(mem_manage), log(NULL)
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

    inline DT* lea (int64_t index)
    {
        assert(index < _len && "Vector index out of bounds.");
        return &_values[index * _stride];
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
        int64_t start = rdtsc();

        DT sum = 0.0;
        for(int i = 0; i < _len; i++)
            sum += _values[i*_stride];

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
            log->log("VECTOR FLOPS", _len);
            log->log("VECTOR BYTES", sizeof(DT)*_len);
        }

        return sum;
    }
    DT min() const
    {
        int64_t start = rdtsc();

        DT min = _values[0*_stride];
        for(int i = 1; i < _len; i++)
            min = std::min(min, _values[i*_stride]);

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
            log->log("VECTOR BYTES", sizeof(DT)*_len);
        }

        return min;
    }
    DT max() const
    {
        int64_t start = rdtsc();

        DT max = _values[0*_stride];
        for(int i = 1; i < _len; i++)
            max = std::max(max, _values[i*_stride]);

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
            log->log("VECTOR BYTES", sizeof(DT)*_len);
        }
        return max;
    }

    bool has_nan() const
    {
        for(int i = 0; i < _len; i++) {
            if(std::isnan((*this)(i)))
                return true;
        }
        return false;
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
        int64_t start = rdtsc();

        for(int i = 0; i < _len; i++) {
            this(i) *= alpha;
        }

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
            log->log("VECTOR BYTES", 2*sizeof(DT)*_len);
            log->log("VECTOR FLOPS", _len);
        }
    }

    DT house_gen() {
        int64_t start = rdtsc();

        DT chi1 = (*this)(0);
        DT nrm_x2_sqr = 0.0;
        
        for(int64_t i = 1; i < _len; i++){
            nrm_x2_sqr += (*this)(i) * (*this)(i);
        }
        DT nrm_x  = sqrt(chi1*chi1 + nrm_x2_sqr);

        double tau = 0.5;
        if(nrm_x2_sqr == 0) {
            (*this)(0) = -chi1;
            if(log) {
                log->log("VECTOR BYTES", sizeof(DT)*_len);
                log->log("VECTOR FLOPS", 2*_len);
            }
        } else {
            DT alpha = -sgn(chi1) * nrm_x;
            DT mult = 1.0 / (chi1 - alpha);
            
            for(int64_t i = 1; i < _len; i++) {
                (*this)(i) *= mult;
            }

            tau = 1.0 /  (0.5 + 0.5 * nrm_x2_sqr * mult * mult);
            (*this)(0) = alpha;

            if(log) {
                log->log("VECTOR FLOPS", 3*_len);
                log->log("VECTOR BYTES", 3*sizeof(DT)*_len);
            }
        }

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
        }
        return tau;
    }

    //Apply this householder transform to another vector.
    //Need to perform x - tau * v * (v'*x)
    //v[0] is implicitly 1.
    void house_apply(double tau, Vector<DT>& x) const 
    {
        int64_t start = rdtsc();

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

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
            log->log("VECTOR FLOPS", 4*_len);
            log->log("VECTOR BYTES", 3*sizeof(DT)*_len);
        }
    }

    inline void house_apply(DT tau, Matrix<DT>& X) const 
    {
        int64_t start = rdtsc();

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

        if(log) {
            log->log("VECTOR TIME", rdtsc() - start);
            log->log("VECTOR FLOPS", 4*_len * X.width());
            log->log("VECTOR BYTES", sizeof(DT)*(_len + 2*X.width() * X.height()));
        }
    }

};

#include "matrix.h"
#include "immintrin.h"
#include "ipps.h"

template<>
inline double Vector<double>::norm2() const
{
    int64_t start = rdtsc();

    double nrm = cblas_dnrm2( _len, _values, _stride);

    if(log) {
        log->log("VECTOR TIME", rdtsc() - start);
        log->log("VECTOR FLOPS", 2*_len);
        log->log("VECTOR BYTES", sizeof(double)*_len);
    }

    return nrm;
}
template<>
inline double Vector<double>::dot(const Vector<double>& other) const
{
    int64_t start = rdtsc();

    double alpha = cblas_ddot( _len, _values, _stride, other._values, other._stride);

    if(log) {
        log->log("VECTOR TIME", rdtsc() - start);
        log->log("VECTOR FLOPS", 2*_len);
        log->log("VECTOR BYTES", sizeof(double)*_len);
    }

    return alpha;
}
template<>
inline void Vector<double>::scale(const double alpha)
{
    int64_t start = rdtsc();

    cblas_dscal(_len, alpha, _values, _stride);

    if(log) {
        log->log("VECTOR TIME", rdtsc() - start);
        log->log("VECTOR FLOPS", _len);
        log->log("VECTOR BYTES", 2*sizeof(double)*_len);
    }
}
template<>
inline void Vector<double>::axpy(const double alpha, const Vector<double>& other)
{
    int64_t start = rdtsc();

    cblas_daxpy(_len, alpha, other._values, other._stride, _values, _stride);

    if(log) {
        log->log("VECTOR TIME", rdtsc() - start);
        log->log("VECTOR FLOPS", 2*_len);
        log->log("VECTOR BYTES", 3*sizeof(double)*_len);
    }
}
template<>
inline void Vector<double>::axpby(const double alpha, const Vector<double>& other, const double beta)
{
    int64_t start = rdtsc();

    cblas_daxpby(_len, alpha, other._values, other._stride, beta, _values, _stride);

    if(log) {
        log->log("VECTOR TIME", rdtsc() - start);
        log->log("VECTOR FLOPS", 3*_len);
        log->log("VECTOR BYTES", 3*sizeof(double)*_len);
    }
}
template<>
void Vector<double>::copy(const Vector<double> from) {
    int64_t start = rdtsc();

    cblas_dcopy(_len, from._values, from._stride, _values, _stride);

    if(log != NULL) {
        log->log("VECTOR TIME", rdtsc() - start);
        log->log("VECTOR BYTES", 2*sizeof(double)*_len);
    }
}

#endif
