#ifndef COVERAGE_H
#define COVERAGE_H

#include "submodular.h"
#include "../la/vector.h"


template<class DT>
class Coverage : public SubmodularFunction<DT> {
public:
    double alpha; //Cardinality penalty coefficient
    int64_t _n;
    int64_t _m;
    int64_t N;
    std::vector<std::vector<int64_t>> edges_a_b;
    std::vector<bool> b_ws;

    Coverage(int64_t n, int64_t m, int64_t N_in) : SubmodularFunction<DT>(n), _n(n), _m(m), N(N_in), alpha(0.0) {
        edges_a_b.reserve(n);
        b_ws.reserve(m);
        for(int64_t i = 0; i < n; i++) {
            edges_a_b.emplace_back();
        }
        for(int64_t i = 0; i < m; i++) {
            b_ws.push_back(false);
        }
    }

    //Must be convex
    //Must normalize s.t. g(0) = 0
    DT g(int64_t k) {
        return 0;
    }

    DT eval(const std::vector<bool>& A) {
        std::fill(b_ws.begin(), b_ws.end(), false);
        b_ws[0] = true;
        b_ws[N*N] = true;
        b_ws[2*N*N] = true;

        int64_t cardinality;
        assert(A.size() == _n);
        int64_t sum = 0;
        for(int64_t i = 0; i < _n; i++) {
            if(A[i]) {
                for(auto e : edges_a_b[i]) {
                    assert(e < _m);
                    if(!b_ws[e]) sum++;
                    b_ws[e] = true;
                }
                cardinality++;
            }
        }
        

        return sum + alpha*g(cardinality);
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) {
        assert(perm.size() == _n && x.length() == _n);
        assert(std::abs(g(0)- 0.0) < 1e-5);

        std::fill(b_ws.begin(), b_ws.end(), false);
        b_ws[0] = true;
        b_ws[N*N] = true;
        b_ws[2*N*N] = true;

        for(int64_t i = 0; i < _n; i++) {
            assert(perm[i] <= _n);
            int64_t gain_i = 0.0;
            for(auto e : edges_a_b[perm[i]]) {
                assert(e < _m);
                if(b_ws[e] == false) gain_i++;
                b_ws[e] = true;
            }
            x(perm[i]) = gain_i + alpha*(g(i) - g(i-1));

            //assert(g(i+1) >= (g(i+2) + g(i)) / 2);
        }
    }
};

template<class DT>
Coverage<DT> build_mmma_coverage(int64_t m, int64_t n, int64_t k) {
    auto cdag = Coverage<DT>(m*n*k, m*n + m*k + n*m, m);

    int64_t m_stride = n*k;
    int64_t n_stride = k;
    int64_t k_stride = 1;

    for(int64_t i = 0; i < m; i++) {
        for(int64_t j = 0; j < n; j++) {
            for(int64_t p = 0; p < k; p++) {
                int64_t addr = i*m_stride + j*n_stride + p*k_stride;
                cdag.edges_a_b[addr].push_back(i + j*m);                 //Element of C
                cdag.edges_a_b[addr].push_back(m*n + i + p*m);           //Element of A
                cdag.edges_a_b[addr].push_back(m*n + m*k + p + j*k);     //Element of B
            }
        }
    }

    return cdag;
};

template<class DT>
class Coverage2 : public SubmodularFunction<DT> {
public:
    Coverage<DT> _cdag;
    int64_t _s;

    Coverage2(Coverage<DT> cdag, int64_t s) : SubmodularFunction<DT>(cdag._n-1), _cdag(cdag), _s(s) { }
    Coverage2(int64_t N, int64_t s) : SubmodularFunction<DT>(N*N*N-1), _s(s), _cdag(build_mmma_coverage<DT>(N,N,N)) { }

    DT eval(const std::vector<bool>& A) {
        std::vector<bool> B(_cdag._n);
        for(int64_t i = 0; i < _s; i++) {
            B[i] = A[i];
        } 
        B[_s] = true;
        for(int64_t i = _s+1; i < _cdag._n; i++) {
            B[i] = A[i-1];
        }

        return _cdag.eval(B);
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& y) {
        std::vector<int64_t> perm_cdag(_cdag._n);
        Vector<DT> x_cdag(_cdag._n);
       
        perm_cdag[0] = _s;
        for(int i = 1; i < _cdag._n; i++) {
            assert(i < _cdag._n);
            if(perm[i-1] < _s) {
                perm_cdag[i] = perm[i-1];
            } else {
                perm_cdag[i] = perm[i-1] + 1;
            }
        }
        
        _cdag.gains(perm_cdag, x_cdag);
        
        auto from = x_cdag.subvector(1, y.length());
        assert(from.length() == y.length());
        y.copy(from);
    }
};

#endif
