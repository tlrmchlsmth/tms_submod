#ifndef TMS_SUBMOD_MATRIX_H
#define TMS_SUBMOD_MATRIX_H

#include <random>
#include <list>

#include "mkl.h"
#include <assert.h>
#include <iomanip>

template<class DT> class Vector;

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
    inline Matrix<DT> submatrix(int64_t row, int64_t col, int64_t mc, int64_t nc)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        auto width  = std::min(nc, _n - col);

        return Matrix<DT>(&_values[row*_rs + col*_cs], height, width, _rs, _cs, _base_m, _base_n, false);
    }
    inline Vector<DT> subrow(int64_t row, int64_t col, int64_t nc)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto width  = std::min(nc, _n - col);
        return Vector<DT>(&_values[row*_rs + col*_cs], width, _cs, false);
    }
    inline Vector<DT> subrow(int64_t row)
    {
        assert(row < _m && "Matrix index out of bounds.");
        return Vector<DT>(&_values[row*_rs], _n, _cs, false);
    }
    inline Vector<DT> subcol(int64_t row, int64_t col, int64_t mc)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        return Vector<DT>(&_values[row*_rs + col*_cs], height, _rs, false);
    }
    inline Vector<DT> subcol(int64_t col)
    {
        assert(col < _n && "Matrix index out of bounds.");
        return Vector<DT>(&_values[col*_cs], _m, _rs, false);
    }

    inline const Matrix<DT> submatrix(int64_t row, int64_t col, int64_t mc, int64_t nc) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        auto width  = std::min(nc, _n - col);

        return Matrix<DT>(&_values[row*_rs + col*_cs], height, width, _rs, _cs, _base_m, _base_n, false);
    }
    inline const Vector<DT> subrow(int64_t row, int64_t col, int64_t nc) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto width  = std::min(nc, _n - col);
        return Vector<DT>(&_values[row*_rs + col*_cs], width, _cs, false);
    }
    inline const Vector<DT> subrow(int64_t row) const
    {
        assert(row < _m && "Matrix index out of bounds.");
        return Vector<DT>(&_values[row*_rs], _n, _cs, false);
    }
    inline const Vector<DT> subcol(int64_t row, int64_t col, int64_t mc) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        return Vector<DT>(&_values[row*_rs + col*_cs], height, _rs, false);
    }
    inline const Vector<DT> subcol(int64_t col) const
    {
        assert(col < _n && "Matrix index out of bounds.");
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

    void set_subdiagonal(DT alpha) {
        for(int64_t i = 0; i < _m; i++) {
            for(int64_t j = 0; j < i && j < _n; j++) {
                (*this)(i,j) = alpha;
            }
        }
    }

    void copy(const Matrix<DT>& other) 
    {
        assert(_m == other._m && _n == other._n);
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    void axpby(DT alpha, const Matrix<DT>& other, DT beta) 
    {
        assert(_m == other._m && _n == other._n);
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                (*this)(i,j) = beta * (*this)(i,j) + alpha * other(i,j);
            }
        }
    }

    DT fro_norm() const
    {
        DT fro_nrm = 0.0;
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                fro_nrm += (*this)(i,j)*(*this)(i,j);
            }
        }
        return sqrt(fro_nrm);
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
        //assert(row < _m && col < _n && "Matrix index out of bounds");
        return _values[row * _rs + col * _cs];
    }

    inline 
    DT operator() (int64_t row, int64_t col) const
    {
        //assert(row < _m && col < _n && "Matrix index out of bounds");
        return _values[row * _rs + col * _cs];
    }

    void print() 
    {
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                std::cout << std::left << std::setw(12) << (*this)(i,j);
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
    void mvm(DT alpha, const Vector<DT>& x, DT beta, Vector<DT>& y)
    {
        std::cout << "Gemv not implemented for datatype" << std::endl;
        exit(1);
    }
    void trsv(CBLAS_UPLO uplo, Vector<DT>& x)
    {
        std::cout << "Trsv not implemented for datatype" << std::endl;
        exit(1);
    }
    void mmm(DT alpha, const Matrix<DT>& A, const Matrix<DT>& B, DT beta)
    {
        std::cout << "Gemm not implemented for datatype" << std::endl;
        exit(1);

    }

    void qr(Vector<DT>& t)
    {
        std::cout << "QR factorization not implemented for datatype" << std::endl;
        exit(1);
    }

    void apply_q(Vector<DT>& t, Matrix<DT>& a) const
    {
        std::cout << "Applying Q from qr factorization not implemented for datatype" << std::endl;
        exit(1);
    }

    //
    // Routines that should be LAPACK routines but are not.
    //
    inline void trap_qr(Vector<DT>& t, int64_t l)
    {
        assert(_n + l >= _m && "Nonconformal trap_qr");
        assert( t._len >= std::min(_m, _n) && "Cannot perform trap qr.");
        
        for(int64_t i = 0; i < _n; i++) {
            Vector<DT> xi = this->subcol(i, i, l);
            t(i) = xi.house_gen();
            
            if(i+1 < _n){
                Matrix A2 = this->submatrix(i, i+1, l, _n-i-1);
                xi.house_apply(t(i), A2);
            }
        }
    }

    inline void apply_trap_q(Matrix<DT>& A, const Vector<DT>& t, int64_t l) const
    {
        assert(_n + l >= _m && "Nonconformal apply_trap_q");
        assert(t._len >= std::min(_m, _n) && "Cannot apply q from trap qr.");
        assert(_cs == 1 || _rs == 1 && "Only row or column major trap_qr supported");

        for(int64_t i = 0; i < _n; i++) {
            const Vector<DT> xi = (*this).subcol(i, i, l);
            Matrix<DT> Ai = A.submatrix(i, 0, l, A.width());
            xi.house_apply(t(i), Ai);
        }
    }

    void remove_cols(std::list<int64_t>& cols_to_remove)
    {
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            int64_t block_begin = *iter - (n_removed - 1);
            int64_t block_end = _n - n_removed;
            if(std::next(iter,1) != cols_to_remove.end()) {
                block_end = *std::next(iter,1) - n_removed;
            }

            for(int64_t i = block_begin; i < block_end; i++) {
                auto col_to_overwrite = this->subcol(i);
                const auto col_to_copy = this->subcol(i + n_removed);
                col_to_overwrite.copy(col_to_copy);
            }

            n_removed++;
        }

        this->enlarge_n(-cols_to_remove.size());
    }

    void remove_cols_trap(std::list<int64_t>& cols_to_remove)
    {
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            int64_t block_begin = *iter - (n_removed - 1);
            int64_t block_end = _n - n_removed;
            if(std::next(iter,1) != cols_to_remove.end()) {
                block_end = *std::next(iter,1) - n_removed;
            }

            for(int64_t i = block_begin; i < block_end; i++) {
                auto col_to_overwrite = this->subcol(0, i, std::min(i + n_removed + 1, _m));
                const auto col_to_copy = this->subcol(0, i + n_removed, std::min(i + n_removed + 1, _m));
                col_to_overwrite.copy(col_to_copy);
            }

            n_removed++;
        }

        this->enlarge_n(-cols_to_remove.size());
    }

    void remove_cols_incremental_qr(std::list<int64_t>& cols_to_remove, Vector<DT>& t)
    {
        //First remove the columns.
        this->remove_cols_trap(cols_to_remove);
        
        //Second pass.
        //Use trapezoidal QR factorization to clean up elements below the diagonal
        //Partition the matrix by panels based on the columns there were removed
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            int64_t block_begin = *iter - (n_removed - 1);
            if(block_begin == _n) break;

            int64_t block_end = _n;
            if(std::next(iter,1) != cols_to_remove.end()) {
                block_end = *std::next(iter,1) - n_removed;
            }

            int64_t m = block_end - block_begin + n_removed;
            int64_t n = block_end - block_begin;

            auto R11 = this->submatrix(block_begin, block_begin, m, n);
            auto t1  = t.subvector(block_begin, n);
            R11.trap_qr(t1, n_removed + 1);

            if(block_end < _n) {
                auto R12 = this->submatrix(block_begin, block_end, m, _n - block_end);
                R11.apply_trap_q(R12, t1, n_removed + 1);
            }

            n_removed++;
        }
        this->enlarge_m(-cols_to_remove.size());
    }

    inline void shift_trapezoid(int64_t dest_m_coord, int64_t dest_n_coord, int64_t nc, int64_t h, int64_t x_dist) 
    {
        if(_rs == 1) {
            int64_t end_n = std::min(_n - x_dist, dest_n_coord + nc);
            for(int64_t j = dest_n_coord; j < end_n; j++) {
                int64_t end_m = std::min(_m, dest_m_coord + j + h);
                #pragma omp parallel for
                for(int64_t i = dest_m_coord; i < end_m; i++) {
                    _values[i*_rs + j*_cs] = _values[i*_rs + (j + x_dist)*_cs];
                }
            }
        } else {
            int64_t end_m = std::min(_m, dest_m_coord + nc + h);
            int64_t end_n = std::min(_n - x_dist, dest_n_coord + nc);
            #pragma omp parallel for
            for(int64_t i = dest_m_coord; i < end_m; i++) {
                int64_t start_n = std::min(i-dest_m_coord - h, (int64_t)0);
                for(int64_t j = dest_n_coord + start_n; j < end_n; j++) {
                    _values[i*_rs + j*_cs] = _values[i*_rs + (j + x_dist)*_cs];
                }
            }
        }
    }

    inline void shift_dense(int64_t dest_m_coord, int64_t dest_n_coord, int64_t mc, int64_t nc, int64_t x_dist) 
    {
        int64_t end_n = std::min(_n - x_dist, dest_n_coord + nc);
        int64_t end_m = std::min(_m, dest_m_coord + mc);
        if(_rs == 1) {
            for(int64_t j = dest_n_coord; j < end_n; j++) {
                #pragma omp parallel for
                for(int64_t i = dest_m_coord; i < end_m; i++) {
                    _values[i*_rs + j*_cs] = _values[i*_rs + (j + x_dist)*_cs];
                }
            }
        } else {
            #pragma omp parallel for
            for(int64_t i = dest_m_coord; i < end_m; i++) {
                for(int64_t j = dest_n_coord; j < end_n; j++) {
                    _values[i*_rs + j*_cs] = _values[i*_rs + (j + x_dist)*_cs];
                }
            }
        }
    }

    void blocked_remove_cols_incremental_qr(std::list<int64_t>& cols_to_remove, Vector<DT>& t, int64_t nb)
    {
        //Fused column deletion and blocked application of orthogonal transformations
        //to remove columns from this matrix, and then annihilate elements below the diagonal.

        //First partition the matrix according to the positions of the columns to be removed
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {

            //trap_begin and trap_end represent the start and end of the trapezoid after shifting it.
            int64_t source_begin = *iter + 1;
            if(source_begin == _n) break;

            int64_t source_end = _n;
            if(std::next(iter,1) != cols_to_remove.end()) {
                source_end = *std::next(iter,1);
            }
            int64_t dest_begin = source_begin - n_removed;
            int64_t dest_end = source_end - n_removed;
            
            //Dimensions of the trapezoidal matrix
            int64_t trap_m = source_end;
            int64_t trap_n = source_end - source_begin;
            int64_t trap_h = n_removed + 1;

            //Need to shift the dense block above the diagonal for this trapezoid.
            this->shift_dense(0, dest_begin, dest_begin + trap_h-1, trap_n, n_removed);

            //Here we have a trapezoidal matrix.
            //Perform blocked trapezoidal QR factorization on the current matrix.
            for(int i = 0; i < trap_n; i += nb) {
                //Size of the data block along the diagonal
                int64_t block_m = std::min(nb + trap_h - 1, trap_n - i + trap_h - 1);
                int64_t block_n = std::min(nb, trap_n - i);

                //Shift this trapezoidal block to the left by n_removed blocks
                int64_t offset = trap_h - 1;
                this->shift_trapezoid(dest_begin + i + offset, dest_begin + i, block_n, trap_h, n_removed);

                //Perform a trapezoidal QR facorization on the first block along the diagonal.
                auto R11 = this->submatrix(dest_begin + i, dest_begin + i, block_m, block_n);
                auto t1  = t.subvector(dest_begin + i, block_n);
                R11.trap_qr(t1, trap_h);

                //Apply Q to the rest of the current (shifted) trapezoidal matrix
                for(int64_t j = i + nb; j < trap_n; j += nb) {
                    block_n = std::min(nb, trap_n - j);
                    //Shift this dense block to the left by n_removed blocks
                    this->shift_dense(dest_begin + i + offset, dest_begin + j, block_m - offset, block_n, n_removed);

                    //Apply Q to this block
                    auto R12 = this->submatrix(dest_begin + i, dest_begin + j, block_m, block_n);
                    R11.apply_trap_q(R12, t1, trap_h);
                }

                //Apply this QR factorization to the rest of the (unshifted) matrix
                //We do some extra work here because we haven't deleted the columns yet
                for(int64_t j = source_end; j < _n; j += nb) {
                    block_n = std::min(nb, _n - j);
                    auto R12 = this->submatrix(dest_begin + i, j, block_m, block_n);
                    R11.apply_trap_q(R12, t1, trap_h);
                }
            }

            n_removed++;
        }

        this->enlarge_n(-cols_to_remove.size());
        this->enlarge_m(-cols_to_remove.size());
    }
};

template<>
void Matrix<double>::mvm(double alpha, const Vector<double>& x, double beta, Vector<double>& y)
{
    assert(_m == y._len && _n == x._len && "Nonconformal mvm.");

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
void Matrix<double>::mmm(double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta)
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

template<>
void Matrix<double>::qr(Vector<double>& tau)
{
    assert(tau._stride == 1 && tau._len >= std::min(_m, _n) && "Cannot perform qr.");
    assert(_cs == 1 || _rs == 1 && "Only row or column major qr supported");

    if(_rs == 1) {
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, _m, _n, _values, _cs, tau._values);
    } else /*if(_cs == 1)*/ {
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, _m, _n, _values, _rs, tau._values);
    } 
}





#endif
