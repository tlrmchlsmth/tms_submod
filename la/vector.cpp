#include <iostream>

#include "vector.h"
#include "matrix.h"
#include "immintrin.h"
#include "ipps.h"

//
// Double precision vector ops
//
template<>
double Vector<double>::norm2() const
{
    double nrm = cblas_dnrm2( _len, _values, _stride);
    return nrm;
}

template<>
double Vector<double>::dot(const Vector<double>& other) const
{
    assert(_len == other.length());
    double alpha = cblas_ddot( _len, _values, _stride, other._values, other._stride);
    return alpha;
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
void Vector<double>::copy(const Vector<double>& from) 
{
    cblas_dcopy(_len, from._values, from._stride, _values, _stride);
}

//
// Single precision vector ops
//
template<>
float Vector<float>::norm2() const
{
    float nrm = cblas_snrm2( _len, _values, _stride);
    return nrm;
}

template<>
float Vector<float>::dot(const Vector<float>& other) const
{
    float alpha = cblas_sdot( _len, _values, _stride, other._values, other._stride);
    return alpha;
}

template<>
void Vector<float>::scale(const float alpha)
{
    cblas_sscal(_len, alpha, _values, _stride);
}

template<>
void Vector<float>::axpy(const float alpha, const Vector<float>& other)
{
    cblas_saxpy(_len, alpha, other._values, other._stride, _values, _stride);
}

template<>
void Vector<float>::axpby(const float alpha, const Vector<float>& other, const float beta)
{
    cblas_saxpby(_len, alpha, other._values, other._stride, beta, _values, _stride);
}

template<>
void Vector<float>::copy(const Vector<float>& from) 
{
    cblas_scopy(_len, from._values, from._stride, _values, _stride);
}

