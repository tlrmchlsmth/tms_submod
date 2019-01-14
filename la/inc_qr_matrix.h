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
    bool _am_buffer_a;
    Matrix<DT> _buffer_a;
    Matrix<DT> _buffer_b;

    //Workspace for orthogonal transformations
    Matrix<DT> _T;
    Matrix<DT> _V;
    Matrix<DT> _ws; 

    // Constructors
    // Supports resizing, so n is the initial n
    IncQRMatrix(int64_t n) : _n(n), _base_n(n), _am_buffer_a(true), _am_upper_tri(true),
        _buffer_a(Matrix<DT>(_n,_n)), _buffer_b(Matrix<DT>(_n,_n)), _ws(Matrix<DT>(NB, _n)), _T(NB, _n), _V(MAX_COLS_AT_ONCE, _n)
    { }

    IncQRMatrix(const IncQRMatrix<DT>& parent, int64_t diag_offset, int64_t nc) : 
        _n(std::min(nc,_n - diag_offset)),
        _base_n(parent._base_n),
        _am_buffer_a(parent._am_buffer_a), _am_upper_tri(parent._am_upper_tri),
        _buffer_a(parent._buffer_a.submatrix(diag_offset, diag_offset, _n, _n)),
        _buffer_b(parent._buffer_b.submatrix(diag_offset, diag_offset, _n, _n)),
        _ws(parent._ws.submatrix(0, 0, NB, _base_n)),
        _T(parent._T.submatrix(0, 0, NB, _base_n)),
        _V(parent._V.submatrix(0, 0, MAX_COLS_AT_ONCE, _base_n))

    {
        assert(diag_offset < _n && "Matrix index out of bounds.");
    }

    void transpose()
    {
        _buffer_a.transpose();
        _buffer_b.transpose();
        _am_upper_tri = ! _am_upper_tri;
    }
    
    inline int64_t height() const { return _n; }

    inline int64_t width() const { return _n; }

    void print() {
        if(_am_buffer_a) {
            _buffer_a.print();
        } else {
            _buffer_b.print();
        }
    }

    void print(std::string s) {
        if(_am_buffer_a) {
            _buffer_a.print(s);
        } else {
            _buffer_b.print(s);
        }
    }

    inline DT& operator() (int64_t row, int64_t col)
    {
        if(_am_buffer_a) {
            return _buffer_a(row,col);
        } else {
            return _buffer_b(row,col);
        }
    }
    inline DT operator() (int64_t row, int64_t col) const
    {
        if(_am_buffer_a) {
            return _buffer_a(row,col);
        } else {
            return _buffer_b(row,col);
        }
    }

    inline DT* lea (int64_t row, int64_t col) 
    {
        if(_am_buffer_a) {
            return _buffer_a.lea(row,col);
        } else {
            return _buffer_b.lea(row,col);
        }
    }
    inline const DT* lea (int64_t row, int64_t col) const
    {
        if(_am_buffer_a) {
            return _buffer_a.lea(row,col);
        } else {
            return _buffer_b.lea(row,col);
        }
    }

    inline Matrix<DT> current_matrix() {
        if( _am_buffer_a ) {
            assert(_buffer_a._m == _n);
            assert(_buffer_a._n == _n);
            return _buffer_a.submatrix(0,0,_n,_n);
        } else {
            assert(_buffer_b._m == _n);
            assert(_buffer_b._n == _n);
            return _buffer_b.submatrix(0,0,_n,_n);
        }
    }

    inline const Matrix<DT> current_matrix() const {
        if( _am_buffer_a ) {
            assert(_buffer_a._m == _n);
            assert(_buffer_a._n == _n);
            return _buffer_a.submatrix(0,0,_n,_n);
        } else {
            assert(_buffer_b._m == _n);
            assert(_buffer_b._n == _n);
            return _buffer_b.submatrix(0,0,_n,_n);
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
        if(_am_buffer_a) {
            return _buffer_a.subrow(row,col,nc);
        } else {
            return _buffer_b.subcol(row,col,nc);
        }
    }
    inline Vector<DT> subrow(int64_t row)
    {
        if(_am_buffer_a) {
            return _buffer_a.subrow(row);
        } else {
            return _buffer_b.subcol(row);
        }
    }
    inline Vector<DT> subcol(int64_t row, int64_t col, int64_t mc)
    {
        if(_am_buffer_a) {
            return _buffer_a.subcol(row,col,mc);
        } else {
            return _buffer_b.subrow(row,col,mc);
        }
    }
    inline Vector<DT> subcol(int64_t col)
    {
        if(_am_buffer_a) {
            return _buffer_a.subrow(row,col,nc);
        } else {
            return _buffer_b.subcol(row,col,nc);
        }
    }

    inline const IncQRMatrix<DT> submatrix(int64_t diag_start, int64_t nc) const
    {
        return IncQRMatrix<DT>(this, diag_start, nc);
    }
    inline const Vector<DT> subrow(int64_t row, int64_t col, int64_t nc) const
    {
        if(_am_buffer_a) {
            return _buffer_a.subrow(row,col,nc);
        } else {
            return _buffer_b.subcol(row,col,nc);
        }
    }
    inline const Vector<DT> subrow(int64_t row) const
    {
        if(_am_buffer_a) {
            return _buffer_a.subrow(row);
        } else {
            return _buffer_b.subcol(row);
        }
    }
    inline const Vector<DT> subcol(int64_t row, int64_t col, int64_t mc) const
    {
        if(_am_buffer_a) {
            return _buffer_a.subcol(row,col,mc);
        } else {
            return _buffer_b.subrow(row,col,mc);
        }
    }
    inline const Vector<DT> subcol(int64_t col) const
    {
        if(_am_buffer_a) {
            return _buffer_a.subrow(row,col,nc);
        } else {
            return _buffer_b.subcol(row,col,nc);
        }
    }

    inline void enlarge_n(int64_t n_inc)
    {
        assert(_n + n_inc <= _base_n && "Cannot add colum to matrix.");
        _n += n_inc;

        assert(    _buffer_a.width() == _buffer_a.height() 
                && _buffer_a.height() == _buffer_b.width() 
                && _buffer_b.width() == _buffer_b.height() && "BEFORE ENLARGE N");

        _buffer_b.enlarge_n(n_inc);
        _buffer_b.enlarge_m(n_inc);

        _buffer_a.enlarge_n(n_inc);
        _buffer_a.enlarge_m(n_inc);

        assert(    _buffer_a.width() == _buffer_a.height() 
                && _buffer_a.height() == _buffer_b.width() 
                && _buffer_b.width() == _buffer_b.height());
        
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
        assert(_buffer_a._n == _buffer_a._m);
        assert(_buffer_a._m == _buffer_b._m);
        assert(_buffer_a._n == _buffer_b._n);

        int64_t task_size = 128;
        int64_t n_remove = cols_to_remove.size();

        if(_am_buffer_a) {
            _buffer_a.remove_cols_incremental_qr_tasks_kressner(_buffer_b, cols_to_remove, _T, _V, task_size, NB, _ws);
            _buffer_a.enlarge_n(-n_remove);
            _buffer_a.enlarge_m(-n_remove);

            _am_buffer_a = false;
        } else {
            _buffer_b.remove_cols_incremental_qr_tasks_kressner(_buffer_a, cols_to_remove, _T, _V, task_size, NB, _ws);
            _buffer_b.enlarge_n(-n_remove);
            _buffer_b.enlarge_m(-n_remove);

            _am_buffer_a = true;
        }

        //resize object
        _n -= n_remove;
//        _ws.enlarge_n(-n_remove); 

        assert(_buffer_a._n == _buffer_a._m);
        assert(_buffer_a._m == _buffer_b._m);
        assert(_buffer_a._n == _buffer_b._n);
    }

    //If the column s is to be added to S, update this matrix to maintain factorization
    void add_col_inc_qr(const Matrix<DT>& S, const Vector<DT>& s) 
    {
        assert(_am_upper_tri);

        DT* r0_buffer;
        int64_t stride;
        if(_am_buffer_a) {
            r0_buffer = _buffer_a._values + _n * _buffer_a._cs;
            stride = _buffer_a._rs;
        } else {
            r0_buffer = _buffer_b._values + _n * _buffer_b._cs;
            stride = _buffer_b._rs;
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

        if( _am_buffer_a ) {
            _buffer_a.trsv(uplo, x);
        } else {
            _buffer_b.trsv(uplo, x);
        }
    }
};
#endif
