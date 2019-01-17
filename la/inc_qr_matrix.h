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
        _a(Matrix<DT>(_n,_n)), _b(Matrix<DT>(_n,_n)), _ws(Matrix<DT>(NB, _n)), _T(NB, _n), _V(MAX_COLS_AT_ONCE, _n)
    { }

    IncQRMatrix(const IncQRMatrix<DT>& parent, int64_t diag_offset, int64_t nc) : 
        _n(std::min(nc,_n - diag_offset)),
        _base_n(parent._base_n),
        _am_a(parent._am_a), _am_upper_tri(parent._am_upper_tri),
        _a(parent._a.submatrix(diag_offset, diag_offset, _n, _n)),
        _b(parent._b.submatrix(diag_offset, diag_offset, _n, _n)),
        _ws(parent._ws.submatrix(0, 0, NB, _base_n)),
        _T(parent._T.submatrix(0, 0, NB, _base_n)),
        _V(parent._V.submatrix(0, 0, MAX_COLS_AT_ONCE, _base_n))

    {
        assert(diag_offset < _n && "Matrix index out of bounds.");
    }

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

    inline Matrix<DT> current_matrix() {
        if( _am_a ) {
            assert(_a._m == _n);
            assert(_a._n == _n);
            return _a.submatrix(0,0,_n,_n);
        } else {
            assert(_b._m == _n);
            assert(_b._n == _n);
            return _b.submatrix(0,0,_n,_n);
        }
    }

    inline const Matrix<DT> current_matrix() const {
        if( _am_a ) {
            assert(_a._m == _n);
            assert(_a._n == _n);
            return _a.submatrix(0,0,_n,_n);
        } else {
            assert(_b._m == _n);
            assert(_b._n == _n);
            return _b.submatrix(0,0,_n,_n);
        }
    }

    //
    // Acquiring submatrices, subvectors
    //
    inline IncQRMatrix<DT> submatrix(int64_t diag_start, int64_t nc)
    {
        return IncQRMatrix<DT>(*this, diag_start, nc);
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
        return IncQRMatrix<DT>(this, diag_start, nc);
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
        assert(_n + n_inc <= _base_n && "Cannot add colum to matrix.");
        _n += n_inc;

        assert(    _a.width() == _a.height() 
                && _a.height() == _b.width() 
                && _b.width() == _b.height() && "BEFORE ENLARGE N");

        _b.enlarge_n(n_inc);
        _b.enlarge_m(n_inc);

        _a.enlarge_n(n_inc);
        _a.enlarge_m(n_inc);

        assert(    _a.width() == _a.height() 
                && _a.height() == _b.width() 
                && _b.width() == _b.height());
        
        //_ws.enlarge_n(n_inc); 
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
        DT rho1 = sqrt(std::abs(s.dot(s) - r0.dot(r0)));
        this->enlarge_n(1);
        (*this)(_n-1, _n-1) = rho1;
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
#endif
