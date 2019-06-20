#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#include "../la/vector.h"
#include "../la/matrix.h"
#include "../perf_log.h"

#include <omp.h>

template<class DT>
class SubmodularFunction {
public:
    //Workspace for the greedy algorithm
    std::vector<bool> A;
    std::vector<int64_t> permutation;
    int64_t n;
    Vector<DT> ws;

    SubmodularFunction(int64_t n_in) : n(n_in), A(n_in), ws(n)
    {
        permutation.reserve(n);
        for(int i = 0; i < n; i++) 
            permutation.push_back(i);
    }
    virtual void initialize_default(){}

    virtual ~SubmodularFunction() {}

    virtual DT eval(const std::vector<bool>& A) = 0;

    std::vector<bool> get_set() const {
        std::vector<bool> V(n);
        std::fill(V.begin(), V.end(), 1);
        return V;
    }

    virtual DT gain(const std::vector<bool>& A, DT FA, int64_t b) {
        std::vector<bool> Ab = A;
        Ab[b] = 1;
        DT FAb = this->eval(Ab);
        return FAb - FA;
    }

    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& p) 
    {
        std::fill(A.begin(), A.end(), 0);
        DT FA_old = 0.0;
        for(int i = 0; i < p.length(); i++) {
            DT gain_i = gain(A, FA_old, perm[i]);
            p(perm[i]) = gain_i;
            A[perm[i]] = 1;
            FA_old += gain_i;
        }
    }

    void polyhedron_greedy_decending(const Vector<DT>& x, Vector<DT>& p) 
    {
        std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) > x(b); } );
        gains(permutation, p);
    }

    DT polyhedron_greedy_ascending(const Vector<DT>& x, Vector<DT>& p, std::vector<bool>& A_out) 
    {
        std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) < x(b); } );
        gains(permutation, p);
        
        //Get current value of F(A)
        /*DT val = 0.0;
        for(int64_t i = 0; i < x.length(); i++) {
            if(x(i) <= 0) val += p(i);
            if(x(i) <= -1e-10) std::cout << "x(i) " << x(i) << " p(i) " << p(i) << std::endl;;
        }*/

        for(int i = 0; i < n; i++) 
            A_out[i] = false;
/*
        DT val = 0.0;
        for(int64_t i = 0; i < n; i++) {
            if(i < x.length()-1)
                assert(x(permutation[i]) <= x(permutation[i+1]));
            if(p(permutation[i]) <= 0.0) {
                val += p(permutation[i]);
                A_out[permutation[i]] = true;
            } else {
                break;
            }
        }
*/
        DT val = 0.0;
        DT min_val = 0.0;
        DT prefix_len = 0;
        for(int64_t i = 0; i < n; i++) {
            val += p(permutation[i]);
            if(val < min_val) {
                min_val = val;
                prefix_len = i+1;
            }
        }
        for(int64_t i = 0; i < prefix_len; i++) {
           A_out[permutation[i]] = true; 
        }
        return min_val;

//        for(int64_t i = 0; i < n; i++) {
//            std::cout << x(permutation[i]) << "\t" << p(permutation[i]) << std::endl;
//        }

//        return val;
    }
};





#endif
