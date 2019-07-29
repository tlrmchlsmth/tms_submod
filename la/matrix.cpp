#include <iostream>

#include "vector.h"
#include "matrix.h"

template<>
void Matrix<double>::mvm(double alpha, const Vector<double>& x, double beta, Vector<double>& y) const
{
    assert(_m == y._len && _n == x._len && "Nonconformal mvm.");

    int64_t start, end;
    start = rdtsc();

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

    end = rdtsc();
    PerfLog::get().log_total("MVM FLOPS", 2 * _m * _n);
    PerfLog::get().log_total("MVM TIME", end - start);
    PerfLog::get().log_total("MVM BYTES", sizeof(double) * (_m * _n + 2*_m + _n));
}
template<>
void Matrix<double>::trsv(CBLAS_UPLO uplo, Vector<double>& x) const
{
    assert(_m == _n && _m == x._len && "Nonconformal trsv.");

    int64_t start, end;
    start = rdtsc();
    
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

    end = rdtsc();
    PerfLog::get().log_total("TRSV FLOPS", _n*_n);
    PerfLog::get().log_total("TRSV TIME", end - start);
    PerfLog::get().log_total("TRSV BYTES", sizeof(double) * (_n*_n / 2 + 2*_n));
}

template<>
void Matrix<double>::trsm(CBLAS_UPLO uplo, CBLAS_SIDE side, Matrix<double>& X) const
{
    if(side == CblasLeft) assert(_m == X.height() && "Nonconformal trsm");
    else assert (_m == X.width() && "Nonconformal trsm");
    assert(_m == _n && "Nonconformal trsm.");
    assert((_rs == 1  || _cs == 1) && (X._rs == 1 || X._cs == 1));

    int64_t start, end;
    start = rdtsc();

    int64_t ldx = X._cs * X._rs;
    CBLAS_UPLO uplo_trans = CblasUpper;
    if(uplo == CblasUpper) uplo_trans = CblasLower;

    if(_rs == 1) {
        if(X._rs != 1) {
            cblas_dtrsm(CblasRowMajor, side, uplo_trans, CblasTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _cs,
                    X._values, ldx);
        }
        else {
            cblas_dtrsm(CblasColMajor, side, uplo, CblasNoTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _cs,
                    X._values, ldx);
        }
    } else {
        if(X._cs != 1) {
            cblas_dtrsm(CblasColMajor, side, uplo_trans, CblasTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _rs,
                    X._values, ldx);
        }
        else {
            cblas_dtrsm(CblasRowMajor, side, uplo, CblasNoTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _rs,
                    X._values, ldx);
        }
    }

    end = rdtsc();

    PerfLog::get().log_total("TRSM FLOPS", _n*_n*X.width());
    PerfLog::get().log_total("TRSM TIME", end - start);
    PerfLog::get().log_total("TRSM BYTES", sizeof(double) * (_n*_n/2 + 2*_n*X.width()));
}

template<>
void Matrix<double>::mmm(double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta)
{
    assert(_m == A._m && _n == B._n && A._n == B._m && "Nonconformal gemm");
    assert((_rs == 1 || _cs == 1) && (A._rs == 1 || A._cs == 1) && (B._rs == 1 || B._cs == 1)); 

    auto ATrans = CblasNoTrans;
    auto BTrans = CblasNoTrans;
    int64_t lda = A._rs * A._cs;
    int64_t ldb = B._rs * B._cs;

    int64_t start = rdtsc();

    if(_rs == 1) {
        if(A._rs != 1) ATrans = CblasTrans;
        if(B._rs != 1) BTrans = CblasTrans;

        cblas_dgemm(CblasColMajor, ATrans, BTrans, _m, _n, A._n,
                alpha, A._values, lda, B._values, ldb,
                beta, _values, _cs);
    } else {
        if(A._cs != 1) ATrans = CblasTrans;
        if(B._cs != 1) BTrans = CblasTrans;

        cblas_dgemm(CblasRowMajor, ATrans, BTrans, _m, _n, A._n,
                alpha, A._values, lda, B._values, ldb,
                beta, _values, _rs);
    }

    PerfLog::get().log_total("MMM FLOPS", 2 * _m * _n * A._n);
    PerfLog::get().log_total("MMM TIME", rdtsc() - start);
    PerfLog::get().log_total("MMM BYTES", sizeof(double) * (2*_m *_n + _m * A._n + A._n * _n)); 
}

template<>
void Matrix<double>::syrk(CBLAS_UPLO uplo, double alpha, const Matrix<double>& A, double beta)
{
    assert(_m == A._m && _m == _n && "Nonconformal syrk");
    assert((_rs == 1 || _cs == 1) && (A._rs == 1 || A._cs == 1));

    auto ATrans = CblasNoTrans;
    int64_t lda = A._rs * A._cs;

    int64_t start = rdtsc();
    if(_rs == 1) {
        if(A._rs != 1) ATrans = CblasTrans;
        cblas_dsyrk(CblasColMajor, uplo, ATrans, _n, A._n, 
            alpha, A._values, lda,
            beta, _values, _cs);
    } else {
        if(A._cs != 1) ATrans = CblasTrans;
        cblas_dsyrk(CblasRowMajor, uplo, ATrans, _n, A._n, 
            alpha, A._values, lda,
            beta, _values, _rs);
    }

    int64_t end = rdtsc();
    PerfLog::get().log_total("SYRK FLOPS", _n*_n*A._n);
    PerfLog::get().log_total("SYRK TIME", end - start);
}

template<>
void Matrix<double>::qr(Vector<double>& t)
{
    assert(t._stride == 1 && t._len >= std::min(_m, _n) && "Cannot perform qr.");
    assert(_cs == 1 || _rs == 1 && "Only row or column major qr supported");

    int64_t start, end;
    start = rdtsc();

    if(_rs == 1) {
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, _m, _n, _values, _cs, t._values);
    } else /*if(_cs == 1)*/ {
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, _m, _n, _values, _rs, t._values);
    }

    end = rdtsc();
    PerfLog::get().log_total("QR FLOPS", 2*_m*(_n*_n - 2*_n/3));
    PerfLog::get().log_total("QR TIME", end - start);
}

template<>
void Matrix<double>::chol(char uplo)
{
    assert(_m == _n);
    assert(_rs == 1 || _cs == 1);
    int inf;
    int n = _n;
    if(_rs == 1) {
        int lda = _cs;
        dpotrf_(&uplo, &n, _values, &lda, &inf);
    } else {
        char uplo_trans = 'L';
        if(uplo == 'L') uplo_trans = 'U';
        int lda = _rs;

        dpotrf_(&uplo_trans, &n, _values, &lda, &inf);
    }
#if 0
    if(_rs == 1) {
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, _m,  _values, _cs);
    } else /*if(_cs == 1)*/ {
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, _m,  _values, _rs);
    }
#endif
}
template<>
void Matrix<float>::chol(char uplo)
{
    assert(_m == _n);
    assert(_rs == 1 || _cs == 1);
    int inf;
    int n = _n;
    if(_rs == 1) {
        int lda = _cs;
        spotrf_(&uplo, &n, _values, &lda, &inf);
    } else {
        char uplo_trans = 'L';
        if(uplo == 'L') uplo_trans = 'U';
        int lda = _rs;

        spotrf_(&uplo_trans, &n, _values, &lda, &inf);
    }
#if 0
    if(_rs == 1) {
        LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, _m,  _values, _cs);
    } else /*if(_cs == 1)*/ {
        LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, _m,  _values, _rs);
    }
#endif
}


//Use tpqrt to annihilate rectangle above the triangle.
//Then stores the reflectors in that block
//TODO: handle row-major
template<>
void Matrix<double>::tpqr(Matrix<double>& B, Matrix<double>& T, int64_t l_in, int64_t nb_in)
{
    int32_t m = B.height();
    int32_t n = B.width();

    assert(_m == _n && _n == B.width() && _n == T.width() && T.height() >= nb_in && "Nonconformal tpqrt");
    assert(_cs == 1 || _rs == 1 && "Only row or column major qr supported");
    assert(_rs == 1 && "Only column major qr supported");
    assert(T._rs == 1); assert(B._rs == 1);

    int32_t l = l_in;
    int32_t nb = std::min(n, (int32_t)nb_in);

    int64_t start = rdtsc();

    LAPACKE_dtpqrt(LAPACK_COL_MAJOR, m, n, l, nb,
            _values, this->_cs, B._values, B._cs, T._values, T._cs);

    int64_t end = rdtsc();

    PerfLog::get().log_total("TPQR TIME", end - start);
}

template<>
void Matrix<double>::apply_tpq(Matrix<double>& A, Matrix<double>& B, const Matrix<double>& T, int64_t l_in, int64_t nb_in, Matrix<double>& ws) const
{
    int m = B.height();
    int n = B.width();
    int k = A.height();
    int l = l_in;
    
    // A is k-by-n, B is m-by-n and V is m-by-k.
    assert(_m == m && _n == k && A.width() == n && T.height() >= nb_in && T.width() == k && "Nonconformal apply tpq");
    assert((_cs == 1 || _rs == 1) && "Only row or column major qr supported");
    assert(_rs == 1 && "Only column major qr supported");


    //Apply from the left
    int nb = std::min(nb_in, A.height());
   
    //TODO: make sure ws has enough size 
    //TODO 2: handle row stride
    assert(_rs == 1); 


    char side = 'L';
    char trans = 'T'; //WHY??
    int ldv = _cs;
    int ldt = T._cs;
    int lda = A._cs;
    int ldb = B._cs;
    int info;

    int64_t start, end;
    start = rdtsc();

    dtpmqrt_(&side, &trans, &m, &n, &k, &l, &nb,
            _values, &ldv, T._values, &ldt,
            A._values, &lda, B._values, &ldb,
            ws._values, &info);

    end = rdtsc();
    PerfLog::get().log_total("APPLY TPQR FLOPS", k*k*n + 4*m*k);
    PerfLog::get().log_total("APPLY TPQR TIME", end - start);
}

template<>
void Matrix<float>::mvm(float alpha, const Vector<float>& x, float beta, Vector<float>& y) const
{
    assert(_m == y._len && _n == x._len && "Nonconformal mvm.");

    int64_t start, end;
    start = rdtsc();

    if(_rs == 1) {
        cblas_sgemv(CblasColMajor, CblasNoTrans, _m, _n, alpha, _values, _cs, 
                x._values, x._stride, 
                beta, y._values, y._stride);
    } else if(_cs == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, _m, _n, alpha, _values, _rs, 
                x._values, x._stride, 
                beta, y._values, y._stride);
    } else {
        std::cout << "Only row or column major GEMV supported. Exiting..." << std::endl;
        exit(1);
    }

    end = rdtsc();
    PerfLog::get().log_total("MVM FLOPS", 2 * _m * _n);
    PerfLog::get().log_total("MVM TIME", end - start);
    PerfLog::get().log_total("MVM BYTES", sizeof(float) * (_m * _n + 2*_m + _n));
}
template<>
void Matrix<float>::trsv(CBLAS_UPLO uplo, Vector<float>& x) const
{
    assert(_m == _n && _m == x._len && "Nonconformal trsv.");

    int64_t start, end;
    start = rdtsc();
    
    if(_rs == 1) {
        cblas_strsv(CblasColMajor, uplo, CblasNoTrans, CblasNonUnit, _m, _values, _cs,
                x._values, x._stride);
    } else if(_cs == 1) {
        cblas_strsv(CblasRowMajor, uplo, CblasNoTrans, CblasNonUnit, _m, _values, _rs,
                x._values, x._stride);
    } else {
        std::cout << "Only row or column major GEMV supported. Exiting..." << std::endl;
        exit(1);
    }

    end = rdtsc();
    PerfLog::get().log_total("TRSV FLOPS", _m * _n);
    PerfLog::get().log_total("TRSV TIME", end - start);
    PerfLog::get().log_total("TRSV BYTES", sizeof(float) * (_m * _n / 2 + 2*_m + _n));
}

template<>
void Matrix<float>::trsm(CBLAS_UPLO uplo, CBLAS_SIDE side, Matrix<float>& X) const
{
    if(side == CblasLeft) assert(_m == X.height() && "Nonconformal trsm");
    else assert (_m == X.width() && "Nonconformal trsm");
    assert(_m == _n && "Nonconformal trsm.");
    assert((_rs == 1  || _cs == 1) && (X._rs == 1 || X._cs == 1));

    int64_t start, end;
    start = rdtsc();

    int64_t ldx = X._cs * X._rs;
    CBLAS_UPLO uplo_trans = CblasUpper;
    if(uplo == CblasUpper) uplo_trans = CblasLower;

    if(_rs == 1) {
        if(X._rs != 1) {
            cblas_strsm(CblasRowMajor, side, uplo_trans, CblasTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _cs,
                    X._values, ldx);
        }
        else {
            cblas_strsm(CblasColMajor, side, uplo, CblasNoTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _cs,
                    X._values, ldx);
        }
    } else {
        if(X._cs != 1) {
            cblas_strsm(CblasColMajor, side, uplo_trans, CblasTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _rs,
                    X._values, ldx);
        }
        else {
            cblas_strsm(CblasRowMajor, side, uplo, CblasNoTrans, CblasNonUnit, 
                    X.height(), X.width(), 1.0,
                    _values, _rs,
                    X._values, ldx);
        }
    }

    end = rdtsc();
    PerfLog::get().log_total("TRSM FLOPS", _n*_n*X.width());
    PerfLog::get().log_total("TRSM TIME", end - start);
    PerfLog::get().log_total("TRSM BYTES", sizeof(float) * (_n*_n/2 + 2*_n*X.width()));
}

template<>
void Matrix<float>::mmm(float alpha, const Matrix<float>& A, const Matrix<float>& B, float beta)
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

    int64_t start, end;
    start = rdtsc();
    if(_rs == 1) {
        if(A._rs != 1) ATrans = CblasTrans;
        if(B._rs != 1) BTrans = CblasTrans;

        cblas_sgemm(CblasColMajor, ATrans, BTrans, _m, _n, A._n,
                alpha, A._values, lda, B._values, ldb,
                beta, _values, _cs);
    } else if(_cs == 1) {
        if(A._cs != 1) ATrans = CblasTrans;
        if(B._cs != 1) BTrans = CblasTrans;

        cblas_sgemm(CblasRowMajor, ATrans, BTrans, _m, _n, A._n,
                alpha, A._values, lda, B._values, ldb,
                beta, _values, _rs);
    } else {
        std::cout << "Only row or column major QR supported. Exiting..." << std::endl;
        exit(1);
    }

    end = rdtsc();
    PerfLog::get().log_total("MMM FLOPS", 2 * _m * _n * A._n);
    PerfLog::get().log_total("MMM TIME", end - start);
    PerfLog::get().log_total("MMM BYTES", sizeof(float) * (2*_m *_n + _m * A._n + A._n * _n)); 
}

template<>
void Matrix<float>::syrk(CBLAS_UPLO uplo, float alpha, const Matrix<float>& A, float beta)
{
    assert(_m == A._m && _m == _n && "Nonconformal syrk");
    assert((_rs == 1 || _cs == 1) && (A._rs == 1 || A._cs == 1));

    auto ATrans = CblasNoTrans;
    int64_t lda = A._rs * A._cs;

    int64_t start = rdtsc();
    if(_rs == 1) {
        if(A._rs != 1) ATrans = CblasTrans;
        cblas_ssyrk(CblasColMajor, uplo, ATrans, _n, A._n, 
            alpha, A._values, lda,
            beta, _values, _cs);
    } else {
        if(A._cs != 1) ATrans = CblasTrans;
        cblas_ssyrk(CblasRowMajor, uplo, ATrans, _n, A._n, 
            alpha, A._values, lda,
            beta, _values, _rs);
    }
    int64_t end = rdtsc();
    PerfLog::get().log_total("SYRK FLOPS", _n*_n*A._n);
    PerfLog::get().log_total("SYRK TIME", end - start);
}

template<>
void Matrix<float>::qr(Vector<float>& t)
{
    assert(t._stride == 1 && t._len >= std::min(_m, _n) && "Cannot perform qr.");
    assert(_cs == 1 || _rs == 1 && "Only row or column major qr supported");

    int64_t start, end;
    start = rdtsc();

    if(_rs == 1) {
        LAPACKE_sgeqrf(LAPACK_COL_MAJOR, _m, _n, _values, _cs, t._values);
    } else /*if(_cs == 1)*/ {
        LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, _m, _n, _values, _rs, t._values);
    }

    end = rdtsc();
    PerfLog::get().log_total("QR FLOPS", 2*_m*(_n*_n - 2*_n/3));
    PerfLog::get().log_total("QR TIME", end - start);
}


//Use tpqrt to annihilate rectangle above the triangle.
//Then stores the reflectors in that block
//TODO: handle row-major
template<>
void Matrix<float>::tpqr(Matrix<float>& B, Matrix<float>& T, int64_t l_in, int64_t nb_in)
{
    assert(_m == _n && _n == B.width() && _n == T.width() && T.height() >= nb_in && "Nonconformal tpqrt");
    assert(_cs == 1 || _rs == 1 && "Only row or column major qr supported");
    assert(_rs == 1 && "Only column major qr supported");
    assert(T._rs == 1); assert(B._rs == 1);

    int32_t m = B.height();
    int32_t n = B.width();
    int32_t l = l_in;
    int32_t nb = std::min(n, (int32_t)nb_in);

    int64_t start = rdtsc();

    LAPACKE_stpqrt(LAPACK_COL_MAJOR, m, n, l, nb,
            _values, this->_cs, B._values, B._cs, T._values, T._cs);

    int64_t end = rdtsc();

    PerfLog::get().log_total("TPQR TIME", end - start);
}

template<>
void Matrix<float>::apply_tpq(Matrix<float>& A, Matrix<float>& B, const Matrix<float>& T, int64_t l_in, int64_t nb_in, Matrix<float>& ws) const
{
    int m = B.height();
    int n = B.width();
    int k = A.height();
    int l = l_in;
    
    // A is k-by-n, B is m-by-n and V is m-by-k.
    assert(_m == m && _n == k && A.width() == n && T.height() >= nb_in && T.width() == k && "Nonconformal apply tpq");
    assert((_cs == 1 || _rs == 1) && "Only row or column major qr supported");
    assert(_rs == 1 && "Only column major qr supported");


    //Apply from the left
    int nb = std::min(nb_in, A.height());
   
    //TODO: make sure ws has enough size 
    //TODO 2: handle row stride
    assert(_rs == 1); 


    char side = 'L';
    char trans = 'T'; //WHY??
    int ldv = _cs;
    int ldt = T._cs;
    int lda = A._cs;
    int ldb = B._cs;
    int info;

    int64_t start, end;
    start = rdtsc();

    stpmqrt_(&side, &trans, &m, &n, &k, &l, &nb,
            _values, &ldv, T._values, &ldt,
            A._values, &lda, B._values, &ldb,
            ws._values, &info);

    end = rdtsc();

    PerfLog::get().log_total("APPLY TPQR FLOPS", k*k*n + 4*m*k);
    PerfLog::get().log_total("APPLY TPQR TIME", end - start);
}

