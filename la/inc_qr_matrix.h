#ifndef TMS_SUBMOD_IC_QR_MATRIX_H
#define TMS_SUBMOD_IC_QR_MATRIX_H

//
// Stores one triangular matrix, supporting adding and removing columns,
// and applying orthogonal transformations to maintain factorization.
// Other half of the matrix is workspace.
//
// Flips between a column stored upper triangular matrix and a row stored lower triangular transposed matrix
// This facilitates out of place column removal and updates
//

#include <list>
#include "matrix.h"

template<class DT>
void remove_cols_permute_rows_out_of_place(Matrix<DT>& src, Matrix<DT>& dest, const std::list<int64_t>& cols_to_remove, Matrix<DT>& V)
{
    if(cols_to_remove.size() == 0) {
        dest.copy_upper_tri(src);    
    }
    //Copy initial triangle
    auto src_tri = src.submatrix(0, 0, cols_to_remove.front(), cols_to_remove.front());
    auto dest_tri = dest.submatrix(0, 0, cols_to_remove.front(), cols_to_remove.front());
    dest_tri.copy_upper_tri(src_tri);

    int64_t cols_removed = 1;
    for(auto j_iter = cols_to_remove.begin(); j_iter != cols_to_remove.end(); j_iter++) {
        int64_t source_j_begin = *j_iter + 1;
        if(source_j_begin == src.width()) break;

        int64_t source_j_end = src.width();
        if(std::next(j_iter,1) != cols_to_remove.end()) {
            source_j_end = *std::next(j_iter,1);
        }
        int64_t dest_j_begin = source_j_begin - cols_removed;

        //Size of the triangular matrix along diagonal
        int64_t trap_n = source_j_end - source_j_begin;

        //Early exit conditions
        if(source_j_begin == source_j_end) {
            cols_removed++;
            continue;
        }

        //Copy blocks above diagonal, copy rows to annihilate
        int64_t rows_permuted = 0;
        int64_t source_i_begin = 0;
        
        for(auto i_iter = cols_to_remove.begin(); i_iter != std::next(j_iter,1); i_iter++) {
            int64_t dest_i_begin = source_i_begin - rows_permuted;

            //Copy the row to annihilate into the V matrix
            auto r = src.subrow(*i_iter, source_j_begin, trap_n);
            auto v = V.subrow(rows_permuted, dest_j_begin, trap_n);
            v.copy(r);
            
            //Copy block
            auto src_blk = src.submatrix(source_i_begin, source_j_begin, *i_iter - source_i_begin, trap_n);
            auto dest_blk = dest.submatrix(dest_i_begin, dest_j_begin, *i_iter - source_i_begin, trap_n);
            dest_blk.copy(src_blk);

            source_i_begin = *i_iter + 1;
            rows_permuted++;
        }

        //Copy triangle along diagonal
        auto src_tri = src.submatrix(source_j_begin, source_j_begin, trap_n, trap_n);
        auto dest_tri = dest.submatrix(dest_j_begin, dest_j_begin, trap_n, trap_n);
        dest_tri.copy_upper_tri(src_tri);

        cols_removed++;
    }
}
//Remove columns and rows beforehand, and use tpqr to annihilate rows, task parallel version
//ws must be at least nb by n
template<class DT>
void remove_cols_incremental_qr_tasks_kressner(Matrix<DT>& src, Matrix<DT>& dest, const std::list<int64_t>& cols_to_remove, Matrix<DT>& T, Matrix<DT>& V, int64_t task_size, int64_t nb, Matrix<DT>& ws)
{
    //Delete columns and permute rows into the V matrix
    remove_cols_permute_rows_out_of_place(src, dest, cols_to_remove, V);

    //Partition the matrix according to the positions of the columns to be removed
    int64_t n_removed = 1;
    for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
        //trap_begin and trap_end represent the start and end of the trapezoid after shifting it.
        int64_t trap_begin = (*iter) + 1 - n_removed;
        int64_t trap_end = dest.width();
        if(std::next(iter,1) != cols_to_remove.end())
            trap_end = *std::next(iter,1) - n_removed;
        int64_t trap_n = trap_end - trap_begin;

        //Early exit conditions
        if(trap_begin == trap_end) {
            n_removed++;
            continue;
        }
        
        #pragma omp parallel
        #pragma omp single
        {
            for(int64_t i = 0; i < trap_n; i += task_size) {
                int64_t block_m = std::min(task_size, trap_n - i);
                int64_t block_begin = trap_begin + i;

                auto V1 = V.submatrix(0, block_begin, n_removed, block_m);
                auto T1 = T.submatrix(0, block_begin, T.height(), block_m);

                //Factorization
                #pragma omp task depend(inout: i)
                {
                    auto R11 = dest.submatrix(block_begin, block_begin, block_m, block_m);
                    auto ws1 = ws.submatrix(0, block_begin, ws.height(), block_m);
                    R11.tpqr(V1, T1, 0, nb, ws1);
                }

                //Apply Q to the rest of the matrix
                for(int64_t j = block_begin + block_m; j < dest.width(); j += task_size) {
                    int64_t block_n = std::min(task_size, dest.width() - j);
                    
                    #pragma omp task depend(in:i) //depend(inout:j)
                    {
                        auto R12 = dest.submatrix(block_begin, j, block_m, block_n);
                        auto V2 = V.submatrix(0, j, n_removed, block_n);
                        auto ws2 = ws.submatrix(0, j, ws.height(), block_n);
                        V1.apply_tpq(R12, V2, T1, 0, nb, ws2);
                    }
                }
            }
        }
        n_removed++;
    }
}

template<class DT>
class IncQRMatrix {
public:
//    DT * _values;
    int64_t _n;
    int64_t _base_n;
//    bool _mem_manage;

    bool _am_upper_tri; 
    bool _am_half_a;
    Matrix<DT> _half_a; //Either a column-stored upper triangular matrix or a row-stored lower triangular matrix
    Matrix<DT> _half_b; //Either a row-stored upper triangular matrix or a column-stored lower triangular matrix

    //Workspace for orthogonal transformations
    int64_t _nb = 16;
    int64_t _max_cols_at_once = 128;
    Matrix<DT> _ws; 

    //
    // Constructors
    // Supports resizeing, so n is the initial n
    IncQRMatrix(int64_t n) : _n(n), _base_n(n), _am_half_a(true), _am_upper_tri(true),
        _half_a(Matrix<DT>(_n,_n)), _half_b(Matrix<DT>(_n,_n)), _ws(Matrix<DT>(2*_nb + _max_cols_at_once, n))
    {
/*        const int ret = posix_memalign((void **) &_values, 4096, (_n+1) * _n * sizeof(DT));
        _half_a._values = &_values[0];
        _half_b._values = &_values[1];
        
        _half_a = Matrix<DT>(&_values[0], _n, _n, 1, _n+1, _n, _n, false, NULL);
        _half_b = Matrix<DT>(&_values[1], _n, _n, _n+1, 1, _n, _n, false, NULL);
*/
    }

    IncQRMatrix(const IncQRMatrix<DT>& parent, int64_t diag_offset, int64_t nc) : 
        _n(std::min(nc,_n - diag_offset)),
        _base_n(parent._base_n),
        _am_half_a(parent._am_half_a), _am_upper_tri(parent._am_upper_tri),
        _half_a(parent._half_a.submatrix(diag_offset, diag_offset, _n, _n)),
        _half_b(parent._half_b.submatrix(diag_offset, diag_offset, _n, _n)),
        _ws(parent._ws.submatrix(0, diag_offset, parent._ws.height(), _n))

    {
        assert(diag_offset < _n && "Matrix index out of bounds.");
//        auto size = std::min(nc, _n - diag_offset);
//        _n = size;
    }

//    ~IncQRMatrix()
//    {
//        if(_mem_manage){
//            free(_values);
//        }
//    }

    void transpose()
    {
        _half_a.transpose();
        _half_b.transpose();
        _am_upper_tri = ! _am_upper_tri;
    }
    
    inline int64_t height() const { return _n; }

    inline int64_t width() const { return _n; }

    void print() {
        if(_am_half_a) {
            _half_a.print();
        } else {
            _half_b.print();
        }
    }

    void print(std::string s) {
        if(_am_half_a) {
            _half_a.print(s);
        } else {
            _half_b.print(s);
        }
    }

    inline DT& operator() (int64_t row, int64_t col)
    {
        if(_am_half_a) {
            return _half_a(row,col);
        } else {
            return _half_b(row,col);
        }
    }
    inline DT operator() (int64_t row, int64_t col) const
    {
        if(_am_half_a) {
            return _half_a(row,col);
        } else {
            return _half_b(row,col);
        }
    }

    inline DT* lea (int64_t row, int64_t col) 
    {
        if(_am_half_a) {
            return _half_a.lea(row,col);
        } else {
            return _half_b.lea(row,col);
        }
    }
    inline const DT* lea (int64_t row, int64_t col) const
    {
        if(_am_half_a) {
            return _half_a.lea(row,col);
        } else {
            return _half_b.lea(row,col);
        }
    }

    inline Matrix<DT> current_matrix() {
        if( _am_half_a ) {
            assert(_half_a._m == _n);
            assert(_half_a._n == _n);
            return _half_a.submatrix(0,0,_n,_n);
        } else {
            assert(_half_b._m == _n);
            assert(_half_b._n == _n);
            return _half_b.submatrix(0,0,_n,_n);
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
        if(_am_half_a) {
            return _half_a.subrow(row,col,nc);
        } else {
            return _half_b.subcol(row,col,nc);
        }
    }
    inline Vector<DT> subrow(int64_t row)
    {
        if(_am_half_a) {
            return _half_a.subrow(row);
        } else {
            return _half_b.subcol(row);
        }
    }
    inline Vector<DT> subcol(int64_t row, int64_t col, int64_t mc)
    {
        if(_am_half_a) {
            return _half_a.subcol(row,col,mc);
        } else {
            return _half_b.subrow(row,col,mc);
        }
    }
    inline Vector<DT> subcol(int64_t col)
    {
        if(_am_half_a) {
            return _half_a.subrow(row,col,nc);
        } else {
            return _half_b.subcol(row,col,nc);
        }
    }

    inline const IncQRMatrix<DT> submatrix(int64_t diag_start, int64_t nc) const
    {
        return IncQRMatrix<DT>(this, diag_start, nc);
    }
    inline const Vector<DT> subrow(int64_t row, int64_t col, int64_t nc) const
    {
        if(_am_half_a) {
            return _half_a.subrow(row,col,nc);
        } else {
            return _half_b.subcol(row,col,nc);
        }
    }
    inline const Vector<DT> subrow(int64_t row) const
    {
        if(_am_half_a) {
            return _half_a.subrow(row);
        } else {
            return _half_b.subcol(row);
        }
    }
    inline const Vector<DT> subcol(int64_t row, int64_t col, int64_t mc) const
    {
        if(_am_half_a) {
            return _half_a.subcol(row,col,mc);
        } else {
            return _half_b.subrow(row,col,mc);
        }
    }
    inline const Vector<DT> subcol(int64_t col) const
    {
        if(_am_half_a) {
            return _half_a.subrow(row,col,nc);
        } else {
            return _half_b.subcol(row,col,nc);
        }
    }

    inline void enlarge_n(int64_t n_inc)
    {
        assert(_n + n_inc <= _base_n && "Cannot add colum to matrix.");
        _n += n_inc;

        assert(    _half_a.width() == _half_a.height() 
                && _half_a.height() == _half_b.width() 
                && _half_b.width() == _half_b.height() && "BEFORE ENLARGE N");

        _half_b.enlarge_n(n_inc);
        _half_b.enlarge_m(n_inc);

        _half_a.enlarge_n(n_inc);
        _half_a.enlarge_m(n_inc);

        assert(    _half_a.width() == _half_a.height() 
                && _half_a.height() == _half_b.width() 
                && _half_b.width() == _half_b.height());
        
        _ws.enlarge_n(n_inc); 
    }


    void remove_cols_inc_qr(const std::list<int64_t>& cols_to_remove)
    {
        if(cols_to_remove.size() == 0) return;

        int64_t task_size = 32;

        Matrix<DT> T  = _ws.submatrix(0,0,_nb,_n);
        Matrix<DT> ws = _ws.submatrix(_nb,0,_nb,_n);
        Matrix<DT> V  = _ws.submatrix(2*_nb,0,cols_to_remove.size(),_n);

        if(_am_half_a) {
            remove_cols_incremental_qr_tasks_kressner(_half_a, _half_b, cols_to_remove, T, V, task_size, _nb, ws);
        } else {
            remove_cols_incremental_qr_tasks_kressner(_half_b, _half_a, cols_to_remove, T, V, task_size, _nb, ws);
        }
        this->enlarge_n(-cols_to_remove.size());
        _am_half_a = !_am_half_a;
    }

    //If the column s is to be added to S, update this matrix to maintain factorization
    bool add_col_inc_qr(const Matrix<DT>& S, const Vector<DT>& s) 
    {
        assert(_am_upper_tri);

        DT* r0_buffer;
        int64_t stride;
        if(_am_half_a) {
            r0_buffer = _half_a._values + _n * _half_a._cs;
            stride = _half_a._rs;
        } else {
            r0_buffer = _half_b._values + _n * _half_b._cs;
            stride = _half_b._rs;
        }
        
        // Let [r0 rho1]^T be the vector to add to r
        // r0 = R' \ (S' * s)
        Vector<DT> r0(r0_buffer, _n, stride, false);
        auto ST = S.transposed();
        ST.mvm(1.0, s, 0.0, r0);
        this->transpose(); this->trsv(r0); this->transpose();

        // rho1^2 = s' * s - r0' * r0;
        DT rho1 = sqrt(std::abs(s.dot(s) - r0.dot(r0)));
        if(rho1 < 1e-12) {
            return false;
        } else {
            this->enlarge_n(1);
            (*this)(_n-1, _n-1) = rho1;
            return true;
        }
    }

    void trsv(Vector<DT>& x)
    {
        CBLAS_UPLO uplo = CblasLower;
        if( _am_upper_tri ) uplo = CblasUpper;

        if( _am_half_a ) {
            _half_a.trsv(uplo, x);
        } else {
            _half_b.trsv(uplo, x);
        }
    }
};
#endif
