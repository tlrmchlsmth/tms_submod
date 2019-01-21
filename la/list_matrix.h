#ifndef TMS_SUBMOD_LIST_MATRIX_H
#define TMS_SUBMOD_LIST_MATRIX_H
#include <random>
#include <list>

#include "mkl.h"
#include <assert.h>
#include <iomanip>
#include "../perf/perf.h"
#include "../perf_log.h"
#include "vector.h"

template<class DT>
class ColListMatrix {
public:
    std::vector<int64_t> _cols;
    DT * _buffer;
    int64_t _n;
    int64_t _k;
    int64_t _cs;

    ColListMatrix(int64_t n, int64_t width) : _n(n), _k(0), _cols(), _cs(n){
        const int ret = posix_memalign((void **) &_buffer, 4096, _n * width * sizeof(DT));
        if (ret != 0) {
            std::cout << "Could not allocate memory for Matrix. Exiting ..." << std::endl;
            exit(1);
        }

        _cols.reserve(width);
        for(int64_t j = 0; j < width; j++) {
            _cols.push_back(j);
        }
    }

    inline int64_t width()  const { return _k; }
    inline int64_t height() const { return _n; }
    
    inline Vector<DT> subcol(int64_t col) {
        assert(col < _k && "Matrix index out of bounds.");
        return Vector<DT>(&_buffer[_cols[col]*_cs], _n, _n, 1, false);
    }
    inline const Vector<DT> subcol(int64_t col) const {
        assert(col < _k && "Matrix index out of bounds.");
        return Vector<DT>(&_buffer[_cols[col]*_cs], _n, _n, 1, false);
    }

    inline Vector<DT> next_col() {
        assert(_k < _cols.size());
        return Vector<DT>(&_buffer[_cols[_k]*_cs], _n, _n, 1, false);
    }
    
    inline void enlarge_width() {
        _k++;
    }

    void remove_cols(const std::list<int64_t>& cols_to_remove)
    {
        int64_t n_removed = 1;

        for(auto iter = cols_to_remove.begin(); iter != cols_to_remove.end(); iter++) {
            int64_t block_begin = *iter - (n_removed - 1);
            int64_t block_end = _n;
            if(std::next(iter,1) != cols_to_remove.end()) {
                block_end = *std::next(iter,1);
            }

            std::rotate(_cols.begin() + block_begin, _cols.begin() + block_begin + n_removed, _cols.begin() + block_end);

            n_removed++;
        }

        _k -= cols_to_remove.size();
         
    }
    
    void mvm(DT alpha, const Vector<DT>& x, DT beta, Vector<DT>& y) const {
        y.scale(beta);
        for(int64_t j = 0; j < _k; j++) {
            auto aj = Vector<DT>(&_buffer[_cols[j]*_cs], _n, _n, 1, false);
            y.axpy(alpha * x(j), aj);
        }
    }

    void transposed_mvm(DT alpha, const Vector<DT>& x, DT beta, Vector<DT>& y) const {
        y.scale(beta);
        _Pragma("omp parallel for")
        for(int64_t j = 0; j < _k; j++) {
            auto aj = Vector<DT>(&_buffer[_cols[j]*_cs], _n, _n, 1, false);
            y(j) += alpha*aj.dot(x);
        }
    }

};

//specialization
template<>
inline void ColListMatrix<double>::mvm(double alpha, const Vector<double>& x, double beta, Vector<double>& y) const {
    y.scale(beta);
    for(int64_t j = 0; j < _k; j++) {
        cblas_daxpy(_n, alpha * x(j), &_buffer[_cols[j]*_cs], 1, y._values, y._stride);
    }
}
template<>
inline void ColListMatrix<double>::transposed_mvm(double alpha, const Vector<double>& x, double beta, Vector<double>& y) const {
    y.scale(beta);
    _Pragma("omp parallel for")
    for(int64_t j = 0; j < _k; j++) {
        y(j) += alpha * cblas_ddot(_n, &_buffer[_cols[j]*_cs], 1, x._values, x._stride);
    }
}

#endif
