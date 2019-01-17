#ifndef SCMM_H
#define SCMM_H

#include "submodular.h"
#include "../la/vector.h"
#include "../la/matrix.h"

template<class DT>
class MinOneX {
public:
    MinOneX(int64_t k) {}
    DT eval(DT x, int64_t j) {
        return std::min(1.0, x);
    }
};

template<class DT>
class Log {
public:
    Log(int64_t k) {}
    DT eval(DT x, int64_t j) {
        return log(x + 1.0);
    }
};

template<class DT>
class Sqrt {
public:
    Sqrt(int64_t k) {}
    DT eval(DT x, int64_t j) {
        return sqrt(x);
    }
};

template<class DT>
class MinusAXSqr {
public:
    Vector<DT> _a;
    MinusAXSqr(int64_t k) : _a(k) {
        std::random_device rd; 
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> dist(0.0, .005);
        _a.fill_rand(gen, dist);
    }
    DT eval(DT x, int64_t j) {
        return -_a(j) * x * x;
    }
};

template<class DT, class CONV>
class SCMM : public SubmodularFunction<DT> {
public:
    int64_t _n;
    int64_t _k;
    Matrix<DT> _M;
    CONV _sigma;
    DT _normalization;
    
    //Workspace
    Vector<DT> _y;

    SCMM(int64_t n, int64_t k) : SubmodularFunction<DT>(n),
         _n(n), _M(k, n), _sigma(k), _y(k) 
    {
        std::random_device rd; 
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> dist(1.0, 5.0);
        _M.fill_rand(gen, dist);

        _normalization = 0.0;
        for(int j = 0; j < _k; j++) {
            _normalization -= _sigma.eval(0.0, j);
        }

    }

    SCMM(int64_t n) : SubmodularFunction<DT>(n),
         _n(n), _k(8), _M(_k, n), _sigma(_k), _y(_k) 
    {
        std::random_device rd; 
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> dist(1.0, 5.0);
        _M.fill_rand(gen, dist);

        _normalization = 0.0;
        for(int j = 0; j < _k; j++) {
            _normalization -= _sigma.eval(0.0, j);
        }
    }

    DT eval(const std::vector<bool>& A) {
        _y.set_all(0.0);
        for(int64_t i = 0; i < _n; i++) { 
            if(A[i]) {
                auto mi = _M.subcol(i);
                _y.axpy(1.0, mi);
            }
        }

        DT val = 0.0;
        for(int64_t j = 0; j < _k; j++) {
            val += _sigma.eval(_y(j), j);
        }
        
        return val + _normalization;
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& p) {
        _y.set_all(0.0);
        DT f_perm_i_minus_one = _normalization;
        for(int i = 0; i < _n; i++) {
            auto mi = _M.subcol(perm[i]);
            _y.axpy(1.0, mi);
            
            DT f_perm_i = 0.0;
            for(int j = 0; j < _k; j++) {
                f_perm_i += _sigma.eval(_y(j), j);
            }
            p(perm[i]) = f_perm_i - f_perm_i_minus_one;
            f_perm_i_minus_one = f_perm_i;
        }
    }
};

#endif
