#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#include "../la/vector.h"
#include "../la/matrix.h"
#include "../perf_log.h"

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

    virtual ~SubmodularFunction() {}

    virtual DT eval(const std::vector<bool>& A) = 0;

/*    std::vector<bool> get_set() const {
        std::vector<bool> V(n);
        std::fill(V.begin(), V.end(), 1);
        return V;
    }*/

    virtual DT gain(std::vector<bool>& A, DT FA, int64_t b) {
        A[b] = true;
        DT FAb = this->eval(A);
        A[b] = false;
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
        
        //
        //  In exact arithmetic, we should be able to return A[i] = x(i) < 0.0.
        //  However, for numerical reasons, this sometimes gives an incorrect result.
        //  Therefore, we take the minimum prefix sum of p(permutation[i]).
        //  This is guaranteed to give a valid F(A) because we are summing the gains in exactly the order that we incrementally added the elements to the working set.
        //
        //  This approach may give a better F(A) than if we set A[i] = x(i) < 0.0 (or some tolerance) ,
        //  and could let us terminate earlier.
        //
        std::fill(A_out.begin(), A_out.end(), false); 
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
    }

    DT m_hat(const std::vector<bool>& S) 
    {
        DT m_hat_S = 0.0;
        std::vector<bool> S_i(n);
        std::fill(S_i.begin(), S_i.end(), false);
        for(int64_t i = 0; i < n; i++) {
            if(S[i]) {
                S_i[i] = true;
                m_hat_S += this->eval(S_i);
                S_i[i] = false;
            }
        }
        return m_hat_S;
    }
    
    virtual DT greedy_maximize(int64_t cardinality_constraint, std::vector<bool>& A) 
    {
        assert(A.size() == n);

        std::fill(A.begin(), A.end(), false);
        auto F_A = eval(A); 

        int64_t k = 0;
        while(k < cardinality_constraint) {
            DT greatest_gain = -1.0;
            int64_t elem_to_add = -1;
            for(int64_t i = 0; i < n; i++) {
                if(A[i]) continue;

                //Get the gain
                auto gain_i = gain(A, F_A, i);
                if(gain_i > greatest_gain) {
                    greatest_gain = gain_i;
                    elem_to_add = i;
                }
            }

            if(greatest_gain < 0.0) break;
            A[elem_to_add] = true;
            F_A += greatest_gain;
        }

        return F_A;
    }

};

#endif
