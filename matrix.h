#ifndef TMS_SUBMOD_MATRIX_H
#define TMS_SUBMOD_MATRIX_H

#include "mkl.h"
#include <random>

template<class DT>
class Matrix {

//protected:
public:
    DT * _values;
    int64_t _m;
    int64_t _n;
    int64_t _rs;
    int64_t _cs;

    int64_t _base_m;
    int64_t _base_n;

    bool _mem_manage;

//public:

    //
    // Constructors
    //
    Matrix(int64_t m, int64_t n) : _m(m), _n(n), _rs(1), _cs(m), _mem_manage(true), _base_m(m), _base_n(n)
    {
        //TODO: Pad columns so each column is aligned
        const int ret = posix_memalign((void **) &_values, 4096, _m * _n * sizeof(DT));
        if (ret == 0) {
        } else {
            std::cout << "Could not allocate memory for Matrix. Exiting ..." << std::endl;
            exit(1);
        }
    }

    Matrix(DT* values, int64_t m, int64_t n, int64_t rs, int64_t cs, int64_t base_m, int64_t base_n, bool mem_manage) :
        _values(values), _m(m), _n(n), _rs(rs), _cs(cs), _base_m(base_m), _base_n(base_n), _mem_manage(mem_manage)
    {
    }
    ~Matrix()
    {
        if(_mem_manage){
            free(_values);
        }
    }

    //
    // Acquiring submatrices, subvectors
    //
    Matrix<DT> submatrix(int64_t row, int64_t col, int64_t height, int64_t width)
    {
        return Matrix<DT>(&_values[row*_rs + col*_cs], height, width, _rs, _cs, _base_m, _base_n, false);
    }
    Vector<DT> subrow(int64_t row, int64_t col, int64_t width)
    {
        return Vector<DT>(&_values[row*_rs, col*_cs], width, _cs, false);
    }
    Vector<DT> subrow(int64_t row)
    {
        return Vector<DT>(&_values[row*_rs], _n, _cs, false);
    }
    Vector<DT> subcol(int64_t row, int64_t col, int64_t height)
    {
        return Vector<DT>(&_values[row*_rs, col*_cs], height, _rs, false);
    }
    Vector<DT> subcol(int64_t col)
    {
        return Vector<DT>(&_values[col*_cs], _m, _rs, false);
    }

    template<class RNG, class DIST>
    void fill_rand(RNG &gen, DIST &dist) {
        for(int64_t i = 0; i < _m; i++) {
            for(int64_t j = 0; j < _n; j++) {
                (*this)(i,j) = dist(gen);
            }
        }
    }

    void copy(const Matrix<DT>& other) 
    {
        assert(_m = other._m && _n == other._n);
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    //
    // Simple routines
    //
    inline void transpose()
    {
        std::swap(_m, _n);
        std::swap(_rs, _cs);
        std::swap(_base_m, _base_n);
    }

    Matrix<DT> transposed()
    {
        return Matrix<DT>(_values, _n, _m, _cs, _rs, _base_n, _base_m, false);
    }

    inline int64_t height() { return _m; }
    inline int64_t width() { return _n; }

    inline 
    DT& operator() (int64_t row, int64_t col)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds");
        return _values[row * _rs + col * _cs];
    }

    inline 
    DT operator() (int64_t row, int64_t col) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds");
        return _values[row * _rs + col * _cs];
    }

    void print() 
    {
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                std::cout << (*this)(i,j) << "\t";
            }
            std::cout << std::endl;
        }
    }

    void enlarge_m(int64_t m_inc)
    {
        assert(_m + m_inc <= _base_m && "Cannot add row to matrix.");
        _m += m_inc;
    }

    void enlarge_n(int64_t n_inc)
    {
        assert(_n + n_inc <= _base_n && "Cannot add colum to matrix.");
        _n += n_inc;
    }

    //
    // BLAS, LAPACK routines
    //
    void gemv(DT alpha, const Vector<DT>& x, DT beta, Vector<DT>& y)
    {
        std::cout << "gemv not implemented for datatype" << std::endl;
        exit(1);
    }
    void trsv(CBLAS_UPLO uplo, Vector<DT>& x)
    {
        std::cout << "trsv not implemented for datatype" << std::endl;
        exit(1);
    }
    void qr(Vector<DT>& tau)
    {
        std::cout << "QR factorization not implemented for datatype" << std::endl;
        exit(1);
    }
    void gemm(DT alpha, const Matrix<DT>& A, const Matrix<DT>& B, DT beta)
    {
        std::cout << "GEMM not implemented for datatype" << std::endl;
        exit(1);

    }
};

template<>
void Matrix<double>::gemv(double alpha, const Vector<double>& x, double beta, Vector<double>& y)
{
    assert(_m == y._len && _n == x._len && "Nonconformal gemv.");

    if(_rs == 1) {
        cblas_dgemv(CblasColMajor, CblasNoTrans, _m, _n, alpha, _values, _cs, 
                x._values, x._stride, 
                beta, y._values, y._stride);
    } else if(_cs == 1) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, _m, _n, alpha, _values, _rs, 
                x._values, x._stride, 
                beta, y._values, y._stride);
    } else {
        std::cout << "Only row or column major GEMV supported. Exiting..." << std::endl;
        exit(1);
    }
}
template<>
void Matrix<double>::trsv(CBLAS_UPLO uplo, Vector<double>& x)
{
    assert(_m == _n && _m == x._len && "Nonconformal trsm.");
    
    if(_rs == 1) {
        cblas_dtrsv(CblasColMajor, uplo, CblasNoTrans, CblasNonUnit, _m, _values, _cs,
                x._values, x._stride);
    } else if(_cs == 1) {
        cblas_dtrsv(CblasRowMajor, uplo, CblasNoTrans, CblasNonUnit, _m, _values, _rs,
                x._values, x._stride);
    } else {
        std::cout << "Only row or column major GEMV supported. Exiting..." << std::endl;
        exit(1);
    }
}

template<>
void Matrix<double>::qr(Vector<double>& tau)
{
    assert(tau._stride == 1 && tau._len >= std::min(_m, _n) && "Cannot perform qr.");

    if(_rs == 1) {
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, _m, _n, _values, _cs, tau._values);
    } else if(_cs == 1) {
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, _m, _n, _values, _rs, tau._values);
    } else {
        std::cout << "Only row or column major QR supported. Exiting..." << std::endl;
        exit(1);
    }
}

template<>
void Matrix<double>::gemm(double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta)
{
    assert(_m == A._m && _n == B._n && A._n == B._m && "Nonconformal gemm");
    
    if((_rs  != 1 && _cs != 1) || (A._rs != 1 && A._cs != 1) || (B._rs != 1 && B._cs != 1)) {
        std::cout << "GEMM requires row or column major. Exiting..." << std::endl;
        exit(1);
    }

    auto ATrans = CblasNoTrans;
    auto BTrans = CblasNoTrans;
    int64_t lda = A._rs * A._cs;
    int64_t ldb = B._rs * B._cs;

    if(_rs == 1) {
        if(A._rs != 1) ATrans = CblasTrans;
        if(B._rs != 1) BTrans = CblasTrans;

        cblas_dgemm(CblasColMajor, ATrans, BTrans, _m, _n, A._n,
                alpha, A._values, lda, B._values, ldb,
                beta, _values, _cs);
    } else if(_cs == 1) {
        if(A._cs != 1) ATrans = CblasTrans;
        if(B._cs != 1) BTrans = CblasTrans;

        cblas_dgemm(CblasRowMajor, ATrans, BTrans, _m, _n, A._n,
                alpha, A._values, lda, B._values, ldb,
                beta, _values, _rs);
    } else {
        std::cout << "Only row or column major QR supported. Exiting..." << std::endl;
        exit(1);
    }
}


#endif
