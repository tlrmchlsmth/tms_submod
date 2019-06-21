#ifndef TMS_SUBMOD_INC_QR_MATRIX_H
#define TMS_SUBMOD_INC_QR_MATRIX_H

// This is simply a wrapper around two buffers that are used to store R such that R^T R = S^T S
// Thus it stores one triangular matrix, supporting adding and removing columns,
// It contains routines and temporary workspace to maintain this relationship
// It currently uses twice as much workspace as is necessary.

// Future work is to store the other buffer in the lower portion of the first,
// taking up half as much workspace

#include <list>
#include "matrix.h"
#include "list_matrix.h"

#define NB 32
#define MAX_COLS_AT_ONCE 32

template<class DT>
class IncQRMatrix {
public:
    int64_t _n;
    int64_t _base_n;

    bool _am_upper_tri; 
    bool _am_a;
    Matrix<DT> _a;
    Matrix<DT> _b;

    //Workspace for orthogonal transformations
    Matrix<DT> _T;
    Matrix<DT> _V;
    Matrix<DT> _ws; 

    // Constructors
    // Supports resizing, so n is the initial n
    IncQRMatrix(int64_t n) : _n(n), _base_n(n), _am_a(true), _am_upper_tri(true),
        _a(Matrix<DT>(_n,_n)), _b(Matrix<DT>(_n,_n)), _ws(Matrix<DT>(2*NB, _n)), _T(2*NB, _n), _V(MAX_COLS_AT_ONCE, _n)
    {
       _b.set_all(0.0); 
       _a.set_all(0.0); 
    }

    IncQRMatrix(int64_t n, int64_t base_n, bool am_upper_tri, bool am_a,
            Matrix<DT> a, Matrix<DT> b, Matrix<DT> T, Matrix<DT> V, Matrix<DT> ws) :
        _n(n), _base_n(base_n), _am_upper_tri(am_upper_tri), _am_a(am_a),
        _a(a), _b(b), _T(T), _V(V), _ws(ws)
    { }

    void transpose()
    {
        _a.transpose();
        _b.transpose();
        _am_upper_tri = ! _am_upper_tri;
    }
    
    inline int64_t height() const { return _n; }

    inline int64_t width() const { return _n; }

    void print() {
        if(_am_a) {
            _a.print();
        } else {
            _b.print();
        }
    }

    void print(std::string s) {
        if(_am_a) {
            _a.print(s);
        } else {
            _b.print(s);
        }
    }

    inline DT& operator() (int64_t row, int64_t col)
    {
        if(_am_a) {
            return _a(row,col);
        } else {
            return _b(row,col);
        }
    }
    inline DT operator() (int64_t row, int64_t col) const
    {
        if(_am_a) {
            return _a(row,col);
        } else {
            return _b(row,col);
        }
    }

    inline DT* lea (int64_t row, int64_t col) 
    {
        if(_am_a) {
            return _a.lea(row,col);
        } else {
            return _b.lea(row,col);
        }
    }
    inline const DT* lea (int64_t row, int64_t col) const
    {
        if(_am_a) {
            return _a.lea(row,col);
        } else {
            return _b.lea(row,col);
        }
    }

    inline Matrix<DT>& current_matrix() {
        if(_am_a) {
            assert(_a._m == _n);
            assert(_a._n == _n);
            return _a;
        } else {
            assert(_b._m == _n);
            assert(_b._n == _n);
            return _b;
        }
    }

    inline const Matrix<DT> current_matrix() const {
        if(_am_a) {
            assert(_a._m == _n);
            assert(_a._n == _n);
            return _a;
        } else {
            assert(_b._m == _n);
            assert(_b._n == _n);
            return _b;
        }
    }

    //
    // Acquiring submatrices, subvectors
    //
    inline IncQRMatrix<DT> submatrix(int64_t diag_start, int64_t nc)
    {
        assert(diag_start < _n && "Matrix index out of bounds.");
        auto n = std::min(nc, _n - diag_start);

        auto toret = IncQRMatrix(n, _base_n, _am_upper_tri, _am_a,
            _a.submatrix(diag_start, diag_start, n, n), 
            _b.submatrix(diag_start, diag_start, n, n), 
            _T.submatrix(0, 0, _T.height(), _T.width()),
            _V.submatrix(0, 0, _V.height(), _T.width()),
            _ws.submatrix(0, 0, _ws.height(), _T.width()));
            
        return toret;
    }

    inline Vector<DT> subrow(int64_t row, int64_t col, int64_t nc)
    {
        if(_am_a) {
            return _a.subrow(row,col,nc);
        } else {
            return _b.subcol(row,col,nc);
        }
    }
    inline Vector<DT> subrow(int64_t row)
    {
        if(_am_a) {
            return _a.subrow(row);
        } else {
            return _b.subcol(row);
        }
    }
    inline Vector<DT> subcol(int64_t row, int64_t col, int64_t mc)
    {
        if(_am_a) {
            return _a.subcol(row,col,mc);
        } else {
            return _b.subrow(row,col,mc);
        }
    }
    inline Vector<DT> subcol(int64_t col)
    {
        if(_am_a) {
            return _a.subrow(row,col,nc);
        } else {
            return _b.subcol(row,col,nc);
        }
    }

    inline const IncQRMatrix<DT> submatrix(int64_t diag_start, int64_t nc) const
    {
        assert(diag_start < _n && "Matrix index out of bounds.");
        auto n = std::min(nc, _n - diag_start);

        auto toret = IncQRMatrix(n, _base_n, _am_upper_tri, _am_a,
            _a.submatrix(diag_start, diag_start, n, n), 
            _b.submatrix(diag_start, diag_start, n, n), 
            _T.submatrix(0, 0, _T.height(), _T.width()),
            _V.submatrix(0, 0, _V.height(), _T.width()),
            _ws.submatrix(0, 0, _ws.height(), _T.width()));
            
        return toret;
    }

    inline const Vector<DT> subrow(int64_t row, int64_t col, int64_t nc) const
    {
        if(_am_a) {
            return _a.subrow(row,col,nc);
        } else {
            return _b.subcol(row,col,nc);
        }
    }
    inline const Vector<DT> subrow(int64_t row) const
    {
        if(_am_a) {
            return _a.subrow(row);
        } else {
            return _b.subcol(row);
        }
    }
    inline const Vector<DT> subcol(int64_t row, int64_t col, int64_t mc) const
    {
        if(_am_a) {
            return _a.subcol(row,col,mc);
        } else {
            return _b.subrow(row,col,mc);
        }
    }
    inline const Vector<DT> subcol(int64_t col) const
    {
        if(_am_a) {
            return _a.subrow(row,col,nc);
        } else {
            return _b.subcol(row,col,nc);
        }
    }

    inline void enlarge_n(int64_t n_inc)
    {
        assert(_n + n_inc <= _base_n && "Cannot enlarge matrix.");
        _n += n_inc;

        assert(    _a.width()  == _a.height() 
                && _a.height() == _b.width() 
                && _b.width()  == _b.height() && "BEFORE ENLARGE N");

        _b.enlarge_n(n_inc);
        _b.enlarge_m(n_inc);

        _a.enlarge_n(n_inc);
        _a.enlarge_m(n_inc);
    }


    void remove_col_inc_qr(int64_t col) {
        std::list<int64_t> cols_to_remove;
        cols_to_remove.push_back(col);
        remove_cols_inc_qr(cols_to_remove);
    }
    void remove_cols_inc_qr(const std::list<int64_t>& cols_to_remove)
    {
        if(cols_to_remove.size() == 0) return;
        assert(_a._n == _a._m);
        assert(_a._m == _b._m);
        assert(_a._n == _b._n);

        int64_t task_size = 128;
        int64_t n_remove = cols_to_remove.size();

        if(_am_a) {
            _a.remove_cols_incremental_qr_tasks_kressner(_b, cols_to_remove, _T, _V, task_size, NB, _ws);
            _a.enlarge_n(-n_remove);
            _a.enlarge_m(-n_remove);

            _am_a = false;
        } else {
            _b.remove_cols_incremental_qr_tasks_kressner(_a, cols_to_remove, _T, _V, task_size, NB, _ws);
            _b.enlarge_n(-n_remove);
            _b.enlarge_m(-n_remove);

            _am_a = true;
        }

        //resize object
        _n -= n_remove;

        assert(_a._n == _a._m);
        assert(_a._m == _b._m);
        assert(_a._n == _b._n);
    }

    //If the column s is to be added to S, update this matrix to maintain factorization
    void add_col_inc_qr(const Matrix<DT>& S, const Vector<DT>& s) 
    {
        assert(_am_upper_tri);

        DT* r0_buffer;
        int64_t stride;
        if(_am_a) {
            r0_buffer = _a._values + _n * _a._cs;
            stride = _a._rs;
        } else {
            r0_buffer = _b._values + _n * _b._cs;
            stride = _b._rs;
        }
        
        // Let [r0 rho1]^T be the vector to add to r
        // r0 = R' \ (S' * s)
        Vector<DT> r0(r0_buffer, _n, _n, stride, false);
        auto ST = S.transposed();
        ST.mvm(1.0, s, 0.0, r0);
        this->transpose(); this->trsv(r0); this->transpose();

        // rho1^2 = s' * s - r0' * r0;
        DT rho1 = sqrt(s.dot(s) - r0.dot(r0));
        this->enlarge_n(1);
        (*this)(_n-1, _n-1) = rho1;
    }

    //If the column s is to be added to S, update this matrix to maintain factorization
    void add_col_inc_qr(const ColListMatrix<DT>& S, const Vector<DT>& s) 
    {
        assert(_am_upper_tri);

        DT* r0_buffer;
        int64_t stride;
        if(_am_a) {
            r0_buffer = _a._values + _n * _a._cs;
            stride = _a._rs;
        } else {
            r0_buffer = _b._values + _n * _b._cs;
            stride = _b._rs;
        }
        
        // Let [r0 rho1]^T be the vector to add to r
        // r0 = R' \ (S' * s)
        Vector<DT> r0(r0_buffer, _n, _n, stride, false);
        S.transposed_mvm(1.0, s, 0.0, r0);
        this->transpose(); this->trsv(r0); this->transpose();
        
        //Now r0 = Rv for some v
        Vector<DT> v(this->width());
        v.copy(r0);
        this.trsv(v);
        v.print("v");

        // rho1^2 = s' * s - r0' * r0;
        DT rho1 = sqrt(std::abs(s.dot(s) - r0.dot(r0)));
        this->enlarge_n(1);
        (*this)(_n-1, _n-1) = rho1;
    }

    //If the column s is to be added to S, update this matrix to maintain factorization
    int64_t  add_cols_inc_qr(const Matrix<DT>& S, const Matrix<DT>& S_hat) 
    {
        assert(_am_upper_tri);
        DT* R_TR_buffer;
        DT* R_BR_buffer;
        int64_t rs, cs; 
        if(_am_a) {
            R_TR_buffer = _a._values + _n * _a._cs;
            R_BR_buffer = _a._values + _n * (_a._cs + _a._rs);
            rs = _a._rs;
            cs = _a._cs;
        } else {
            R_TR_buffer = _b._values + _n * _b._cs;
            R_BR_buffer = _b._values + _n * (_b._cs + _b._rs);
            rs = _b._rs;
            cs = _b._cs;
        }
        
        // Let [r0 rho1]^T be the vector to add to r
        // r0 = R' \ (S' * s)
        Matrix<DT> R_TR(R_TR_buffer, _n, S_hat.width(), rs, cs, _base_n, _base_n, false);
        Matrix<DT> R_BR(R_BR_buffer, S_hat.width(), S_hat.width(), rs, cs, _base_n, _base_n, false);
        assert(_n == S.width()); 
        if(S.width() == 0) {
            R_BR.set_all(0.0);
        }

        auto ST = S.transposed();
        R_TR.mmm(1.0, ST, S_hat, 0.0);
        this->transpose(); this->trsm(R_TR); this->transpose();

        // R_BR' * R_BR = S' * S - R0' * R0;
        R_BR.syrk(CblasUpper, 1.0, S_hat.transposed(), 0.0);
        R_BR.syrk(CblasUpper, -1.0, R_TR.transposed(), 1.0);

        //Now do a left-looking cholesky factorization
        //If we run into a zero or negative along the diagonal, just throw the column away
        int64_t n_valid = 0;
        for(int64_t j = 0; j < R_BR.width(); j++) {
            auto R00 = R_BR.submatrix(0,0,j,j);
            auto r01 = R_BR.subcol(0,j,j);
            R00.transposed().trsv(CblasLower, r01);
            DT rho11_sqr = R_BR(j,j) - r01.dot(r01);
            if(rho11_sqr > 0.0) {
                R_BR(j,j) = sqrt(rho11_sqr);
                n_valid++;
            } else {
                assert(j != 0);
                //Delete the jth row and column
                for(int64_t jj = j; jj < R_BR.width()-1; jj++) {
                    auto dest_tr = R_TR.subcol(jj);
                    auto src_tr = R_TR.subcol(jj+1);
                    dest_tr.copy(src_tr);

                    auto dest_br = R_BR.subcol(0,jj,j);
                    auto src_br = R_BR.subcol(0,jj+1,j);
                    dest_br.copy(src_br);
                    R_BR(j,jj) = R_BR(jj+1,jj+1);
                }
                R_BR.enlarge_m(-1);
                R_BR.enlarge_n(-1);
            }
        }

        this->enlarge_n(n_valid);
        return n_valid;
    }

    void trsm(Matrix<DT>& X) 
    {
        CBLAS_UPLO uplo = CblasLower;
        if( _am_upper_tri ) uplo = CblasUpper;

        if(_am_a) {
            _a.trsm(uplo, CblasLeft, X);
        } else {
            _b.trsm(uplo, CblasLeft, X);
        }

    }

    void trsv(Vector<DT>& x)
    {
        CBLAS_UPLO uplo = CblasLower;
        if( _am_upper_tri ) uplo = CblasUpper;

        if( _am_a ) {
            _a.trsv(uplo, x);
        } else {
            _b.trsv(uplo, x);
        }
    }
};

template<class DT>
DT check_STS_eq_RTR(const Matrix<DT>& S, const IncQRMatrix<DT>& R_in)
{
    assert(R_in._n > 0);
    auto R = R_in.current_matrix();

    Vector<DT> y(S.width());
    y.fill_rand();

    Vector<DT> Sy(S.height());
    Vector<DT> STSy(S.width());
    auto ST = S.transposed();
    S.mvm(1.0, y, 0.0, Sy);
    ST.mvm(1.0, Sy, 0.0, STSy);

    Vector<DT> Ry(R.height());
    Vector<DT> RTRy(R.width());
    auto RT = R.transposed();
    R.set_subdiagonal(0.0);

    R.mvm(1.0, y, 0.0, Ry);
    RT.mvm(1.0, Ry, 0.0, RTRy);

    RTRy.axpy(-1.0, STSy);
    return RTRy.norm2();
}
#endif
