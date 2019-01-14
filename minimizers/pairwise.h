#ifndef TMS_SUBMOD_PAIRWISE_H
#define TMS_SUBMOD_PAIRWISE_H

#include "../la/vector.h"
#include "../la/matrix.h"
#include "../la/inc_qr_matrix.h"
#include "../submodular.h"

#define REALLOC_WS
#define MAX_S_WIDTH

template<class DT>
std::vector<bool> Pairwise(SubmodularFunction<DT>& F, DT eps, DT answer)
{
    int64_t k = 1;
    int64_t max_away = 512;

    //Iterates
    Vector<DT> x(F.n);
    Vector<DT> a_base(max_away); // x = S a
    auto a = a_base.subvector(0, 1);

    Matrix<DT> S_base(F.n,max_away);
    auto S = S_base.submatrix(0, 0, F.n, 1);

    //Workspace
    Vector<DT> STx_base(max_away);
    Vector<DT> d(F.n);
    Vector<DT> tmp(F.n);
    
    //Initialize x, a, S
    Vector<DT> s0 = S.subcol(0);
    s0.fill_rand();
    F.polyhedron_greedy_decending(s0, x); 
    s0.copy(x);
    a(0) = 1.0;

    DT F_best = std::numeric_limits<DT>::max();
    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        assert(S.width() > 0);
        assert(S.width() < S_base.width());
        assert(a.has_nan() == false);
        assert(a.max() <= 1.0);
        assert(a.min() >= 0.0);

        #ifdef REALLOC_WS 
        if(S.width() + 1 >= S_base.width()) {
            int64_t new_size = S_base.width() * 1.5;
            S_base.realloc(F.n, new_size);
            a_base.realloc(new_size);
            STx_base.realloc(new_size);
            S = S_base.submatrix(0,0,F.n,S.width());
            a = a_base.subvector(0,S.width());
        }
        #endif

        //v = argmax x^T v, v in S
        auto STx = STx_base.subvector(0, S.width());
        S.transpose(); S.mvm(1.0, x, 0.0, STx); S.transpose();
        int64_t v_index = STx.index_of_max();

        #ifdef MAX_S_WIDTH
        //Get rid of minimum (don't keep track of it) 
        //We can do this instead of reallocating workspace
        if(S.width()+1 >= S_base.width()) {
            int64_t min_index = STx.index_of_min();
            assert(min_index != v_index);
            S.remove_col(min_index);
            a.remove(min_index);
            if(v_index > min_index)
                v_index--;
        }
        #endif

        //Get s
        auto s = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x, s);
        F_best = std::min(F_curr, F_best);


        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < 1e-5) break;

        //Check to see if s was already in S
        bool s_in_S = false;
        int64_t s_index = -1;
        for(int64_t i = 0; i < S.width(); i++) {
            const auto si = S.subcol(i);
            tmp.copy(s);
            tmp.axpy(-1.0, si);
            if(tmp.norm2() < 1e-10) {
                s_in_S = true;
                s_index = i;
                break;
            }
        }

        //Get v
        const auto v = S.subcol(v_index);
        auto alpha_v = a(v_index);

        //Get pairwise direction
        d.copy(s);
        d.axpy(-1.0, v);

        //Calculate gamma
        DT gamma = std::min(std::max(-d.dot(x) / d.dot(d), 0.0), alpha_v);

        //Update S and a wrt s
        if(s_in_S) {
            a(s_index) += gamma;
        } else {
            S.enlarge_n(1);
            a.enlarge(1);
            a(a.length()-1) = gamma;
        }
       
       //Update S and a wrt v
        if(gamma != alpha_v) {
            a(v_index) -= gamma;
        } else {
            S.remove_col(v_index);
            a.remove(v_index);
        }
        
        //Update x
        x.axpy(gamma, d);

        //Update duality gap
        DT sum_x_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) { if(x(i) < 0.0) sum_x_lt_0 += x(i); }
        duality_gap = std::abs(F_best - sum_x_lt_0);
        k++;
        //if(k % 100 == 0)
        //    std::cout << k << "\t" << duality_gap << "\t" << F_best << "\t" << sum_x_lt_0 << "\t" << xtx_minus_xts << "\t" << gamma << "\t" << alpha_v << "\t" << a.sum() << "\t" << F_best - answer << "\t" << x.norm2() << std::endl;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}

#endif
