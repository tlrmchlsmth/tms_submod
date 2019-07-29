#ifndef TMS_SUBMOD_MATRIX_H
#define TMS_SUBMOD_MATRIX_H
#include <random>
#include <list>

#include "mkl.h"
#include <assert.h>
#include <iomanip>
#include "../perf/perf.h"
#include "../perf_log.h"
#include "../util.h"

template<class DT> class Vector;

template<class DT>
class Matrix 
{

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
    Matrix(int64_t m, int64_t n) : _m(m), _n(n), _rs(1), _cs(m), _base_m(m), _base_n(n), _mem_manage(true) 
    {
        const int ret = posix_memalign((void **) &_values, 4096, _m * _n * sizeof(DT));
        if (ret != 0) {
            std::cout << "Could not allocate memory for Matrix. Exiting ..." << std::endl;
            exit(1);
        }
    }

    Matrix(DT* values, int64_t m, int64_t n, int64_t rs, int64_t cs, int64_t base_m, int64_t base_n, bool mem_manage) :
        _values(values), _m(m), _n(n), _rs(rs), _cs(cs), _base_m(base_m), _base_n(base_n), _mem_manage(mem_manage)
    { }

    Matrix(DT* values, int64_t m, int64_t n, int64_t rs, int64_t cs) :
        _values(values), _m(m), _n(n), _rs(rs), _cs(cs), _base_m(m), _base_n(n), _mem_manage(false)
    { }

    ~Matrix()
    {
        if(_mem_manage){
            free(_values);
        }
    }

    Matrix& operator=(Matrix&& A) {
        _m = A._m;
        _m = A._n;
        _rs = A._rs;
        _cs = A._cs;
        _base_m = A._m;
        _base_n = A._n;
        _mem_manage = A._mem_manage;
        _values = A._values;
        A._values = nullptr;
    };

    Matrix(Matrix&& A) : 
        _values(A._values),
        _m(A._m), _n(A._n),
        _rs(A._rs), _cs(A._cs),
        _base_m(A._base_m), _base_n(A._base_n),
        _mem_manage(A._mem_manage)
    {
        A._values = nullptr;
    }

    Matrix& operator=(const Matrix& A)
    {
        _m = A._m;
        _m = A._n;
        _base_m = _m;
        _base_n = _n;
        if(A._cs == 1) {
            _cs = 1;
            _rs = _n;
        } else {
            _rs = 1;
            _cs = _m;
        }
        _mem_manage = true;

        const int ret = posix_memalign((void **) &_values, 4096, _m * _n * sizeof(DT));
        if (ret != 0) {
            std::cout << "Could not allocate memory for Matrix. Exiting ..." << std::endl;
            exit(1);
        }

        this->copy(A);
    }

    Matrix(const Matrix& A) :
         _m(A._m), _n(A._n), _base_m(_m), _base_n(_n), _mem_manage(true)
    {
        if(A._cs == 1) {
            _cs = 1;
            _rs = _n;
        } else {
            _rs = 1;
            _cs = _m;
        }
        _mem_manage = true;

        const int ret = posix_memalign((void **) &_values, 4096, _m * _n * sizeof(DT));
        if (ret != 0) {
            std::cout << "Could not allocate memory for Matrix. Exiting ..." << std::endl;
            exit(1);
        }

        this->copy(A);
    }

    
    void realloc(int64_t m, int64_t n) {
        //Can only reallocate "base object"
        assert(_mem_manage == true);
        assert(m >= _m && n >= _n);
        assert(_base_m >= _m && _base_n >= _n);
        
        //Allocate new array
        DT* array;
        const int ret = posix_memalign((void **) &array, 4096, m * n * sizeof(DT));
        if (ret != 0) {
            std::cout << "Could not allocate memory for Matrix. Exiting ..." << std::endl;
            exit(1);
        }
        
        //Copy old array over
        Matrix<DT> tmp(array, m, n, 1, m, m, n, false);
        auto tmp_partition = tmp.submatrix(0,0,_m,_n);
        tmp_partition.copy(*this);

        //Free old array
        free(_values); 

        //Setup fields
        _values = array;
        _m = m;
        _n = n;
        _rs = 1;
        _cs = m;
        _base_m = m;
        _base_n = n;
    }

    int64_t height() const 
    { 
        return _m; 
    }
    int64_t width() const 
    { 
        return _n; 
    }

    DT& operator() (int64_t row, int64_t col)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds");
        return _values[row * _rs + col * _cs];
    }
    
    DT operator() (int64_t row, int64_t col) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds");
        return _values[row * _rs + col * _cs];
    }

    DT* lea (int64_t row, int64_t col) 
    {
        assert(row < _m && col < _n && "Matrix index out of bounds");
        return &_values[row * _rs + col * _cs];
    }

    const DT* lea (int64_t row, int64_t col) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds");
        return &_values[row * _rs + col * _cs];
    }

    //
    // Acquiring submatrices, subvectors
    //
    Matrix<DT> submatrix(int64_t row, int64_t col, int64_t mc, int64_t nc)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        auto width  = std::min(nc, _n - col);

        return Matrix<DT>(lea(row,col), height, width, _rs, _cs, _m, _n, false);
    }

    Vector<DT> subrow(int64_t row, int64_t col, int64_t nc)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto width  = std::min(nc, _n - col);
        return Vector<DT>(&_values[row*_rs + col*_cs], width, _n, _cs, false);
    }

    Vector<DT> subrow(int64_t row)
    {
        assert(row < _m && "Matrix index out of bounds.");
        return Vector<DT>(&_values[row*_rs], _n, _n, _cs, false);
    }

    Vector<DT> subcol(int64_t row, int64_t col, int64_t mc)
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        return Vector<DT>(&_values[row*_rs + col*_cs], height, _m, _rs, false);
    }

    Vector<DT> subcol(int64_t col)
    {
        assert(col < _n && "Matrix index out of bounds.");
        return Vector<DT>(&_values[col*_cs], _m, _m, _rs, false);
    }

    const Matrix<DT> submatrix(int64_t row, int64_t col, int64_t mc, int64_t nc) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        auto width  = std::min(nc, _n - col);
        return Matrix<DT>(&_values[row*_rs + col*_cs], height, width, _rs, _cs, _m, _n, false);
    }

    const Vector<DT> subrow(int64_t row, int64_t col, int64_t nc) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto width  = std::min(nc, _n - col);
        return Vector<DT>(&_values[row*_rs + col*_cs], width, _n, _cs, false);
    }

    const Vector<DT> subrow(int64_t row) const
    {
        assert(row < _m && "Matrix index out of bounds.");
        return Vector<DT>(&_values[row*_rs], _n, _n, _cs, false);
    }

    const Vector<DT> subcol(int64_t row, int64_t col, int64_t mc) const
    {
        assert(row < _m && col < _n && "Matrix index out of bounds.");
        auto height = std::min(mc, _m - row);
        return Vector<DT>(&_values[row*_rs + col*_cs], height, _m, _rs, false);
    }

    const Vector<DT> subcol(int64_t col) const
    {
        assert(col < _n && "Matrix index out of bounds.");
        return Vector<DT>(&_values[col*_cs], _m, _m, _rs, false);
    }

    //
    // Routines for setting elements of the matrix
    //
    void set_all(DT alpha) {
        #pragma omp parallel for
        for(int64_t j = 0; j < _n; j++) {
            for(int64_t i = 0; i < _m; i++) {
                (*this)(i,j) = alpha;
            }
        }
    }

    template<class RNG, class DIST>
    void fill_rand(RNG &gen, DIST &dist) {
        for(int64_t j = 0; j < _n; j++) {
            for(int64_t i = 0; i < _m; i++) {
                (*this)(i,j) = dist(gen);
            }
        }
    }
    void fill_rand() {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::normal_distribution<> normal(0.0, 1.0);
        this->fill_rand(gen, normal);
    }

    void set_diagonal(DT alpha) {
        #pragma omp parallel for
        for(int64_t i = 0; i < std::min(_m, _n); i++) {
            (*this)(i,i) = alpha;
        }
    }

    void set_subdiagonal(DT alpha) {
        #pragma omp parallel for
        for(int64_t i = 0; i < _m; i++) {
            for(int64_t j = 0; j < i && j < _n; j++) {
                (*this)(i,j) = alpha;
            }
        }
    }

    void copy(const Matrix<DT>& other) 
    {
        assert(_m == other._m && _n == other._n);
        #pragma omp parallel for
        for(int j = 0; j < _n; j++) {
            for(int i = 0; i < _m; i++) {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    void copy_permute_rc(const Matrix<DT>& other, const std::vector<int64_t>& p) 
    {
        assert(_m == other._m && _n == other._n);

        #pragma omp parallel for
        for(int j = 0; j < _n; j++) {
            for(int i = 0; i < _m; i++) {
                (*this)(i,j) = other(p[i],p[j]);
            }
        }
    }

    void copy_upper_tri(const Matrix<DT>& other) 
    {
        assert(_m == other._m && _n == other._n);
        assert(_m == _n);

        #pragma omp parallel for
        for(int j = 0; j < _n; j++) {
            for(int i = 0; i <= j; i++) {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    //
    // Simple computational routines
    //
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

    void transpose()
    {
        std::swap(_m, _n);
        std::swap(_rs, _cs);
        std::swap(_base_m, _base_n);
    }

    Matrix<DT> transposed()
    {
        return Matrix<DT>(_values, _n, _m, _cs, _rs, _base_n, _base_m, false);
    }
    const Matrix<DT> transposed() const
    {
        return Matrix<DT>(_values, _n, _m, _cs, _rs, _base_n, _base_m, false);
    }

    void print(std::string name) const
    {
        std::cout << name << std::endl;
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                std::cout << std::left << std::setprecision(5) << std::setw(12) << (*this)(i,j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void print() const
    {
        for(int i = 0; i < _m; i++) {
            for(int j = 0; j < _n; j++) {
                std::cout << std::left << std::setprecision(5) << std::setw(12) << (*this)(i,j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
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
    void mvm(DT, const Vector<DT>&, DT, Vector<DT>&) const
    {
        std::cout << "Gemv not implemented for datatype" << std::endl;
        exit(1);
    }
    void trsv(CBLAS_UPLO, Vector<DT>&) const
    {
        std::cout << "Trsv not implemented for datatype" << std::endl;
        exit(1);
    }

    void trsm(CBLAS_UPLO, CBLAS_SIDE, Matrix<DT>&) const
    {
        std::cout << "Trsv not implemented for datatype" << std::endl;
        exit(1);
    }
    
    void mmm(DT, const Matrix<DT>&, const Matrix<DT>&, DT)
    {
        std::cout << "Gemm not implemented for datatype" << std::endl;
        exit(1);

    }

    void syrk(CBLAS_UPLO, DT, const Matrix<DT>&, DT)
    {
        std::cout << "Syrk not implemented for datatype" << std::endl;
        exit(1);

    }

    void qr(Vector<DT>&)
    {
        std::cout << "QR factorization not implemented for datatype" << std::endl;
        exit(1);
    }
    void chol(char)
    {
        std::cout << "Cholesky factorization not implemented for datatype" << std::endl;
        exit(1);
    }

    void apply_q(Vector<DT>&, Matrix<DT>&) const
    {
        std::cout << "Applying Q from qr factorization not implemented for datatype" << std::endl;
        exit(1);
    }

    void tpqr(Matrix<DT>&, Matrix<DT>&, int64_t, int64_t)
    {
        std::cout << "TPQR factorization not implemented for datatype" << std::endl;
        exit(1);
    }

    void apply_tpq(Matrix<DT>&, Matrix<DT>&, const Matrix<DT>&, int64_t, int64_t, Matrix<DT>&) const
    {
        std::cout << "Applying TPQ not implemented for datatype" << std::endl;
        exit(1);
    }
    
    //
    // Routines related to removing columns of a matrix
    //
    void trap_qr(Vector<DT>& t, int64_t l)
    {
        assert(_n + l >= _m && "Nonconformal trap_qr");
        assert( t._len >= std::min(_m, _n) && "Cannot perform trap qr.");
        
        for(int64_t i = 0; i < _n; i++) {
            t(i) = house_gen(l, &_values[i*_rs + i*_cs], _rs);
            
            if(i+1 < _n){
                 house_apply(l, _n-i-1, &_values[i*_rs + i*_cs], _rs, t(i), &_values[i*_rs + (i+1)*_cs], _rs, _cs);
            }
        }
    }

    void apply_trap_q(Matrix<DT>& A, const Vector<DT>& t, int64_t l) const
    {
        assert(_n + l >= _m && "Nonconformal apply_trap_q");
        assert(t._len >= std::min(_m, _n) && "Cannot apply q from trap qr.");
        assert(_cs == 1 || _rs == 1 && "Only row or column major trap_qr supported");

        for(int64_t i = 0; i < _n; i++) {
            house_apply(l, A.width(), &_values[i*_rs + i*_cs], _rs, t(i), &A._values[i*A._rs], A._rs, A._cs);
        }
    }

    void remove_col(int64_t col) {
        std::list<int64_t> tmp;
        tmp.push_back(col);
        remove_cols(tmp);
    }
    void remove_cols(const std::list<int64_t>& cols_to_remove)
    {
        int64_t start, end;
        start = rdtsc();

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

        end = rdtsc();
        PerfLog::get().log_total("REMOVE COLS BYTES", 2 * _m * _n);
        PerfLog::get().log_total("REMOVE COLS TIME", end - start);
    }

    void remove_cols_trap(const std::list<int64_t>& cols_to_remove)
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

    void remove_cols_trap_oop(Matrix<DT>& dest, const std::list<int64_t>& cols_to_remove) const
    {
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            int64_t block_begin = *iter - (n_removed - 1);
            int64_t block_end = _n - n_removed;
            if(std::next(iter,1) != cols_to_remove.end()) {
                block_end = *std::next(iter,1) - n_removed;
            }

            for(int64_t i = block_begin; i < block_end; i++) {
                auto col_to_overwrite = dest.subcol(0, i, std::min(i + n_removed + 1, _m));
                const auto col_to_copy = this->subcol(0, i + n_removed, std::min(i + n_removed + 1, _m));
                col_to_overwrite.copy(col_to_copy);
            }

            n_removed++;
        }

        dest.enlarge_n(-cols_to_remove.size());
    }


    void shift_trapezoid_up(int64_t dest_m_coord, int64_t dest_n_coord, int64_t nc, int64_t h, int64_t y_dist) 
    {
        if(_rs == 1) {
            int64_t end_n = std::min(_n, dest_n_coord + nc);
            for(int64_t j = dest_n_coord; j < end_n; j++) {
                int64_t end_m = std::min(_m - y_dist, dest_m_coord + j + h);
                #pragma omp parallel for
                for(int64_t i = dest_m_coord; i < end_m; i++) {
                    _values[i*_rs + j*_cs] = _values[(i+y_dist)*_rs + j*_cs];
                }
            }
        } else {
            int64_t end_m = std::min(_m - y_dist, dest_m_coord + nc + h);
            int64_t end_n = std::min(_n, dest_n_coord + nc);
            #pragma omp parallel for
            for(int64_t i = dest_m_coord; i < end_m; i++) {
                int64_t start_n = std::min(i - dest_m_coord - h, (int64_t) 0);
                for(int64_t j = dest_n_coord + start_n; j < end_n; j++) {
                    _values[i*_rs + j*_cs] = _values[(i + y_dist)*_rs + j*_cs];
                }
            }
        }
    }

    void shift_trapezoid_left(int64_t dest_m_coord, int64_t dest_n_coord, int64_t nc, int64_t h, int64_t x_dist) 
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

    void shift_triangle_left(int64_t dest_m_coord, int64_t dest_n_coord, int64_t nc, int64_t x_dist) 
    {
        DT* src =  lea(dest_m_coord, dest_n_coord + x_dist);
        DT* dest = lea(dest_m_coord, dest_n_coord);

        if(_rs == 1) {
            int64_t N = std::min(nc, _n - x_dist - dest_n_coord);
            for(int64_t j = 0; j < N; j++) {
                int64_t M = std::min(j+1, _m - dest_m_coord);
                #pragma omp parallel for
                for(int64_t i = 0; i < M; i++) {
                    dest[i*_rs + j*_cs] = src[i*_rs + j*_cs];
                    src[i*_rs + j*_cs] = 0.0;
                }
            }
        } else {
            int64_t M = std::min(nc, _m - dest_m_coord);
            #pragma omp parallel for
            for(int64_t i = 0; i < M; i++) {
                int64_t N = std::min(nc, _n - dest_n_coord - x_dist);
                for(int64_t j = i; j < N; j++) {
                    dest[i*_rs + j*_cs] = src[i*_rs + j*_cs];
                }
            }
        }
    }

    void shift_triangle_up(int64_t dest_m_coord, int64_t dest_n_coord, int64_t nc, int64_t y_dist) 
    {
        DT* src =  lea(dest_m_coord + y_dist, dest_n_coord);
        DT* dest = lea(dest_m_coord, dest_n_coord);

        if(_rs == 1) {
            int64_t N = std::min(nc, _n - dest_n_coord);

            #pragma omp parallel for
            for(int64_t j = 0; j < N; j++) {
                int64_t M = std::min(j+1, _m - dest_m_coord - y_dist);
                for(int64_t i = 0; i < M; i++) {
                    dest[i*_rs + j*_cs] = src[i*_rs + j*_cs];

                    //if(i+y_dist >= M) src[i*_rs + j*_cs] = 0.0;
                    src[i*_rs + j*_cs] = 0.0;
                }
            }
        } else {
            int64_t M = std::min(nc, _m - dest_m_coord - y_dist);
            for(int64_t i = 0; i < M; i++) {
                int64_t N = std::min(nc, _n - dest_n_coord);
                #pragma omp parallel for
                for(int64_t j = i; j < N; j++) {
                    dest[i*_rs + j*_cs] = src[i*_rs + j*_cs];
                }
            }
        }
    }

    void shift_dense_left(int64_t dest_m_coord, int64_t dest_n_coord, int64_t mc, int64_t nc, int64_t x_dist) 
    {
        int64_t end_n = std::min(_n - x_dist, dest_n_coord + nc);
        int64_t end_m = std::min(_m, dest_m_coord + mc);

        if(_rs == 1) {
            for(int64_t j = dest_n_coord; j < end_n; j++) {
                #pragma omp parallel for
                for(int64_t i = dest_m_coord; i < end_m; i++) {
                    _values[i*_rs + j*_cs] = _values[i*_rs + (j + x_dist)*_cs];

                    _values[i*_rs + (j + x_dist)*_cs] = 0;
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

    void shift_dense_up(int64_t dest_m_coord, int64_t dest_n_coord, int64_t mc, int64_t nc, int64_t y_dist) 
    {
        int64_t end_n = std::min(_n, dest_n_coord + nc);
        int64_t end_m = std::min(_m - y_dist, dest_m_coord + mc);
        if(_rs == 1) {
            #pragma omp parallel for
            for(int64_t j = dest_n_coord; j < end_n; j++) {
                for(int64_t i = dest_m_coord; i < end_m; i++) {
                    _values[i*_rs + j*_cs] = _values[(i+y_dist)*_rs + j*_cs];

                    _values[(i+ y_dist)*_rs + j*_cs] = 0;
                }
            }
        } else {
            for(int64_t i = dest_m_coord; i < end_m; i++) {
                #pragma omp parallel for
                for(int64_t j = dest_n_coord; j < end_n; j++) {
                    _values[i*_rs + j*_cs] = _values[(i + y_dist)*_rs + j*_cs];
                }
            }
        }
    }


    //Delete columns of the matrix and permute rows (to be annihilated) into V
    void remove_cols_permute_rows_in_place(const std::list<int64_t>& cols_to_remove, Matrix<DT>& V)
    {
        int64_t cols_removed = 1;
        for(auto j_iter = cols_to_remove.begin(); j_iter != cols_to_remove.end(); j_iter++) {
            //trap_begin and trap_end represent the start and end of the trapezoid after shifting it.
            int64_t source_j_begin = *j_iter + 1;
            if(source_j_begin == _n) break;

            int64_t source_j_end = _n;
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
                auto r = this->subrow(*i_iter, source_j_begin, trap_n);
                auto v = V.subrow(rows_permuted, dest_j_begin, trap_n);
                v.copy(r);
                
                //Shift block up and to the left                
                this->shift_dense_left(source_i_begin, dest_j_begin, *i_iter - source_i_begin, trap_n, cols_removed);
                if(rows_permuted > 0)
                    this->shift_dense_up(dest_i_begin, dest_j_begin, *i_iter - source_i_begin, trap_n, rows_permuted);

                source_i_begin = *i_iter + 1;
                rows_permuted++;
            }

            //Shift triangle along diagonal up and to the left
            this->shift_triangle_up(dest_j_begin, source_j_begin, trap_n, cols_removed);
            this->shift_triangle_left(dest_j_begin, dest_j_begin, trap_n, cols_removed);

            cols_removed++;
        }

        this->enlarge_n(-cols_to_remove.size());
        this->enlarge_m(-cols_to_remove.size());
    }

    void remove_cols_permute_rows_out_of_place(Matrix<DT>& dest, const std::list<int64_t>& cols_to_remove, Matrix<DT>& V) const
    {
        if(cols_to_remove.size() == 0) {
            dest.copy_upper_tri(*this);    
        }
        //Copy initial triangle
        auto src_tri = this->submatrix(0, 0, cols_to_remove.front(), cols_to_remove.front());
        auto dest_tri = dest.submatrix(0, 0, cols_to_remove.front(), cols_to_remove.front());
        dest_tri.copy_upper_tri(src_tri);

        int64_t cols_removed = 1;
        for(auto j_iter = cols_to_remove.begin(); j_iter != cols_to_remove.end(); j_iter++) {
            int64_t source_j_begin = *j_iter + 1;
            if(source_j_begin == _n) break;

            int64_t source_j_end = _n;
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
                auto r = this->subrow(*i_iter, source_j_begin, trap_n);
                auto v = V.subrow(rows_permuted, dest_j_begin, trap_n);
                v.copy(r);
                
                //Copy block
                auto src_blk = this->submatrix(source_i_begin, source_j_begin, *i_iter - source_i_begin, trap_n);
                auto dest_blk = dest.submatrix(dest_i_begin, dest_j_begin, *i_iter - source_i_begin, trap_n);
                dest_blk.copy(src_blk);

                source_i_begin = *i_iter + 1;
                rows_permuted++;
            }

            //Copy triangle along diagonal
            auto src_tri = this->submatrix(source_j_begin, source_j_begin, trap_n, trap_n);
            auto dest_tri = dest.submatrix(dest_j_begin, dest_j_begin, trap_n, trap_n);
            dest_tri.copy_upper_tri(src_tri);

            cols_removed++;
        }

        dest.enlarge_n(-cols_to_remove.size());
        dest.enlarge_m(-cols_to_remove.size());
    }

    //Remove columns and rows beforehand, and use tpqr to annihilate rows, task parallel version
    //ws must be at least nb by n
    void remove_cols_incremental_qr_tasks_kressner(Matrix<DT>& dest, const std::list<int64_t>& cols_to_remove, Matrix<DT>& T, Matrix<DT>& V, int64_t task_size, int64_t nb, Matrix<DT>& ws) const
    {
        int64_t start, end;
        start = rdtsc();

        //Delete columns and permute rows into the V matrix
        this->remove_cols_permute_rows_out_of_place(dest, cols_to_remove, V);

        //Partition the matrix according to the positions of the columns to be removed
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            //trap_begin and trap_end represent the start and end of the trapezoid after shifting it.
            int64_t trap_begin = *iter + 1 - n_removed;
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
            #pragma omp single nowait
            {
                for(int64_t i = 0; i < trap_n; i += task_size) {
                    int64_t block_m = std::min(task_size, trap_n - i);
                    int64_t block_begin = trap_begin + i;


                    //Factorization
                    #pragma omp task depend(inout: i)
                    {
                        auto V1 = V.submatrix(0, block_begin, n_removed, block_m);
                        auto T1 = T.submatrix(0, block_begin, nb, block_m);

                        auto R11 = dest.submatrix(block_begin, block_begin, block_m, block_m);
                        auto ws1 = ws.submatrix(0, block_begin, nb, block_m);
                        R11.tpqr(V1, T1, 0, nb);
                    }

                    //Apply Q to the rest of the matrix
                    for(int64_t j = block_begin + block_m; j < dest.width(); j += task_size) {
                        int64_t block_n = std::min(task_size, dest.width() - j);
                        
                        #pragma omp task depend(in:i) //depend(inout:j)
                        {
                            auto V1 = V.submatrix(0, block_begin, n_removed, block_m);
                            auto T1 = T.submatrix(0, block_begin, nb, block_m);

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

        end = rdtsc();
        end++;
        PerfLog::get().log_total("REMOVE COLS QR BYTES", 2 * _m * _n);
        PerfLog::get().log_total("REMOVE COLS QR TIME", end - start);
    }

    void remove_cols_incremental_qr_kressner(Matrix<DT>& dest, const std::list<int64_t>& cols_to_remove, Matrix<DT>& T, Matrix<DT>& Vin, int64_t nb, Matrix<DT>& ws) const
    {
        int64_t start, end;
        start = rdtsc();
   
        Matrix<DT> V = Vin.submatrix(0,0,Vin.height(), this->width() - cols_to_remove.size());

        //Delete columns and permute rows into the V matrix
        this->remove_cols_permute_rows_out_of_place(dest, cols_to_remove, V);


        //First partition the matrix according to the positions of the columns to be removed
        int64_t n_removed = 1;
        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            //trap_begin and trap_end represent the start and end of the trapezoid after shifting it.
            int64_t trap_begin = *iter + 1 - n_removed;
            int64_t trap_end = dest.width();
            if(std::next(iter,1) != cols_to_remove.end())
                trap_end = *std::next(iter,1) - n_removed;
            int64_t trap_n = trap_end - trap_begin;

            //Early exit conditions
            if(trap_begin == trap_end) {
                n_removed++;
                continue;
            }

            //TPQR along the diagonal
            auto V1 = V.submatrix(0, trap_begin, n_removed, trap_n);
            auto R11 = dest.submatrix(trap_begin, trap_begin, trap_n, trap_n);
            auto T1 = T.submatrix(0, trap_begin, T.height(), trap_n);
            R11.tpqr(V1, T1, 0, nb);

            //Apply Q to the rest of matrix
            int64_t trail_begin = trap_end;
            if (trail_begin < dest.width()) {
                auto R12 = dest.submatrix(trap_begin, trail_begin, trap_n, dest.width() - trail_begin);
                auto V2 = V.submatrix(0, trail_begin, n_removed,  dest.width() - trail_begin);
                V1.apply_tpq(R12, V2, T1, 0, nb, ws);
            }

            n_removed++;
        }

        end = rdtsc();
        PerfLog::get().log_total("REMOVE COLS QR BYTES", 2 * _m * _n);
        PerfLog::get().log_total("REMOVE COLS QR TIME", end - start);
    }

    void remove_cols_incremental_qr_householder(const std::list<int64_t>& cols_to_remove, Vector<DT>& t)
    {
        int64_t start, end;
        start = rdtsc();

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

        end = rdtsc();
        PerfLog::get().log_total("REMOVE COLS QR BYTES", 2 * _m * _n);
        PerfLog::get().log_total("REMOVE COLS QR TIME", end - start);
    }

    //Use givens rotations to annihilate the element below each diagonal element
    void annihilate_subdiag_givens(Matrix<DT>& T)
    {
        for(int64_t i = 0; i < _n && i < _m-1; i++) {
            //1. figure out Givens rotation to annihilate element below diagonal.
            rotg(lea(i,i), lea(i+1,i), T.lea(0,i), T.lea(1,i));
            
            //2. Trailing update of Givens rotation
            if(i+1 < _n)
                rot(_n-(i+1), lea(i,i+1), _cs, lea(i+1,i+1), _cs, T(0,i), T(1,i));
        }
    }

    void apply_givens_rots(Matrix<DT>& T)
    {
        for(int64_t i = 0; i < _m-1; i++) {
            rot(_n, lea(i,0), _cs, lea(i+1,0), _cs, T(0,i), T(1,i));
        }
    }

    void remove_column_iqr_givens(int64_t column, Matrix<DT>& T, int64_t nb) 
    {
        //First shift dense block above diagonal and to the right of the column removed
        //to the left by one.
        for(int i = 0; i < _m; i++) (*this)(i,column) = 0;
        shift_dense_left(0, column, column, _n-column, 1);

        //Proceed along the diagonal by blocks
        for(int64_t i = column; i+1 < _n; i += nb)
        {
            //Partition the matrix
            auto R11 = this->submatrix(i, i+1, nb+1, nb);
            auto T1  = T.submatrix(0, i+1, 2, nb+1);

            //Annihilate subdiagonal elements and shift left
            R11.annihilate_subdiag_givens(T1);
            shift_triangle_left(i, i, nb, 1);
            
            //Trailing update
            for(int64_t j = i + nb; j+1 < _n; j += nb) {
                auto R12 = this->submatrix(i, j+1, nb+1, nb);
                R12.apply_givens_rots(T1);
                shift_dense_left(i, j, nb, nb, 1); 
            }
        }

        this->enlarge_m(-1);
        this->enlarge_n(-1);
    }
};

template<>
void Matrix<double>::mvm(double alpha, const Vector<double>& x, double beta, Vector<double>& y) const;

template<>
void Matrix<double>::trsv(CBLAS_UPLO uplo, Vector<double>& x) const;

template<>
void Matrix<double>::trsm(CBLAS_UPLO uplo, CBLAS_SIDE side, Matrix<double>& X) const;

template<>
void Matrix<double>::mmm(double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta);

template<>
void Matrix<double>::syrk(CBLAS_UPLO uplo, double alpha, const Matrix<double>& A, double beta);

template<>
void Matrix<double>::qr(Vector<double>& t);

template<>
void Matrix<double>::chol(char uplo);

template<>
void Matrix<float>::chol(char uplo);

template<>
void Matrix<double>::tpqr(Matrix<double>& B, Matrix<double>& T, int64_t l_in, int64_t nb_in);

template<>
void Matrix<double>::apply_tpq(Matrix<double>& A, Matrix<double>& B, const Matrix<double>& T, int64_t l_in, int64_t nb_in, Matrix<double>& ws) const;

template<>
void Matrix<float>::mvm(float alpha, const Vector<float>& x, float beta, Vector<float>& y) const;

template<>
void Matrix<float>::trsv(CBLAS_UPLO uplo, Vector<float>& x) const;

template<>
void Matrix<float>::trsm(CBLAS_UPLO uplo, CBLAS_SIDE side, Matrix<float>& X) const;

template<>
void Matrix<float>::mmm(float alpha, const Matrix<float>& A, const Matrix<float>& B, float beta);

template<>
void Matrix<float>::syrk(CBLAS_UPLO uplo, float alpha, const Matrix<float>& A, float beta);

template<>
void Matrix<float>::qr(Vector<float>& t);

//Use tpqrt to annihilate rectangle above the triangle.
//Then stores the reflectors in that block
//TODO: handle row-major
template<>
void Matrix<float>::tpqr(Matrix<float>& B, Matrix<float>& T, int64_t l_in, int64_t nb_in);

template<>
void Matrix<float>::apply_tpq(Matrix<float>& A, Matrix<float>& B, const Matrix<float>& T, int64_t l_in, int64_t nb_in, Matrix<float>& ws) const;

#endif
