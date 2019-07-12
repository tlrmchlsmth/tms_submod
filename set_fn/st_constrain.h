#ifndef ST_CONSTRAIN_H
#define ST_CONSTRAIN_H

#include "submodular.h"
#include "../la/vector.h"
#include "../la/matrix.h"

/*
 * This class represents a monotone submodular function plus a modular function such that F(0) = 0 and F(V) = 0 
 * SFN - monotone submodular function with ground set size n+2
 * Two extra elements - one of them will always be in A, the other will never be
 */
template<class DT, class SFN>
class STConstrain : public SubmodularFunction<DT> {
public:
    int64_t n;

    SFN submodular;
    int64_t s;
    int64_t t;
    std::vector<bool> A_ws;
    std::vector<int64_t> perm_ws;
    Vector<DT> x_ws;
    DT baseline;

    STConstrain(int64_t n_in) : SubmodularFunction<DT>(n_in), n(n_in), submodular(n_in+2), A_ws(n_in+2), perm_ws(n_in+2), x_ws(n_in+2) 
    {
        std::random_device rd;
        std::mt19937 gen{rd()};

        //Pick s and t 
        std::uniform_int_distribution<> st_dist(0, n+2-1);
        s = st_dist(gen);
        do { t = st_dist(gen); } while(t == s);

        //Get baseline
        std::fill(A_ws.begin(), A_ws.end(), false);
        A_ws[s] = true;
        baseline = submodular.eval(A_ws);
    }

    STConstrain(int64_t n_in, SFN fn_in) : SubmodularFunction<DT>(n_in), n(n_in), submodular(fn_in), A_ws(n_in+2), perm_ws(n_in+2), x_ws(n_in+2) 
    {
        assert(fn_in.n == n+2);
        std::random_device rd;
        std::mt19937 gen{rd()};

        //Pick s and t 
        std::uniform_int_distribution<> st_dist(0, n+2-1);
        s = st_dist(gen);
        do { t = st_dist(gen); } while(t == s);
        
        recalculate_baseline();
    }

    void recalculate_baseline() {
        //Get baseline
        std::fill(A_ws.begin(), A_ws.end(), false);
        A_ws[s] = true;
        baseline = submodular.eval(A_ws);
    }

    DT eval(const std::vector<bool>& A_in) 
    {
        int64_t i1 = std::min(s,t);
        int64_t i2 = std::max(s,t);

        std::fill(A_ws.begin(), A_ws.end(), false);
        for(int64_t i = 0; i < n; i++) {
            //int64_t offset = 0 + (i >= s) + (i >= t);
            int64_t offset = 0 + (i >= i1) + (i+1 >= i2);
            A_ws[i + offset] = A_in[i];
        }
        A_ws[s] = true;
        A_ws[t] = false;

        return submodular.eval(A_ws) - baseline;
    }

    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& x) {
        perm_ws.front() = s;
        perm_ws.back() = t;

        int64_t i1 = std::min(s,t);
        int64_t i2 = std::max(s,t);

        for(int64_t i = 0; i < n; i++) {
            int64_t offset = 0 + (perm[i] >= i1) + (perm[i]+1 >= i2);
            perm_ws[i+1] = perm[i] + offset;
        }
        submodular.gains(perm_ws, x_ws);

        for(int64_t i = 0; i < n; i++) {
            int64_t offset = 0 + (i >= i1) + (i+1 >= i2);
            x(i) = x_ws(i + offset); 
        }
    }
};
#endif
