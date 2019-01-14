#ifndef TMS_SUBMOD_AWAY_STEPS_H
#define TMS_SUBMOD_AWAY_STEPS_H

#include "../la/vector.h"
#include "../la/matrix.h"
#include "../submodular.h"

#define REALLOC_WS
#define MAX_S_WIDTH

template<class DT>
std::vector<bool> AwaySteps(SubmodularFunction<DT>& F, DT eps)
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
    Vector<DT> dFW(F.n); //Frank-Wolfe direction
    Vector<DT> dA(F.n);  //Away direction

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
        //Get s
        auto s = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x, s);
        F_best = std::min(F_curr, F_best);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < 1e-5) break;

        //Get Frank-Wolfe and Away directions
        dFW.copy(s); dFW.axpy(-1.0, x);
        //v = argmax x^T v, v in S
        auto STx = STx_base.subvector(0, S.width());
        S.transpose(); S.mvm(1.0, x, 0.0, STx); S.transpose();
        int64_t v_index = STx.index_of_max();
        const auto v = S.subcol(v_index);
        auto alpha_v = a(v_index);
        dA.copy(x); dA.axpy(-1.0, v);

#ifdef MAX_S_WIDTH
        //Get rid of minimum (don't keep track of it) 
        //We can do this instead of reallocating workspace
        if(S.width()+1 >= S_base.width()) {
            int64_t min_index = STx.index_of_min();
            if(v_index > min_index) 
                v_index--;
            S.remove_col(min_index);
            a.remove(min_index);

            //Todo: fix this extra copying
            auto s_to = S_base.subcol(S.width());
            s_to.copy(s);
            s = s_to;
        }
#endif

        if(-x.dot(dFW) >= -x.dot(dA)) {
            //Forward direction
            DT gamma = std::min(std::max(-dFW.dot(x) / dFW.dot(dFW), 0.0), 1.0);

            //Update weights and S
            if(gamma == 1.0) {
                //S = {s}
                S = S_base.submatrix(0,0,F.n,1);
                auto s0 = S.subcol(0);
                s0.copy(s);

                //a = [1]
                a = a_base.subvector(0,1);
                a(0) = 1.0;

                //update x
                x.copy(s);
            } else {
                //Check to see if s is in S already
                bool s_already_in_S = false;
                int64_t index_s = -1;
                for(int64_t i = 0; i < S.width(); i++) {
                    const auto si = S.subcol(i);
                    tmp.copy(s);
                    tmp.axpy(-1.0, si);
                    if(tmp.norm2() < 1e-10) {
                        s_already_in_S = true;
                        index_s = i;
                        break;
                    }
                }
                
                //Update a and S
                a.scale(1.0-gamma);
                if( s_already_in_S ) {
                    a(index_s) += gamma;
                } else {
                    S.enlarge_n(1);
                    a = a_base.subvector(0, a.length()+1);
                    a(a.length() - 1) = gamma;
                }
                
                //update x
                x.axpy(gamma, dFW);
            }
        } else {
            //Away direction
            DT gamma_max = alpha_v / (1.0 - alpha_v);
            DT gamma = std::min(std::max(-dA.dot(x) / dA.dot(dA), 0.0), gamma_max);

            bool drop_step = (gamma == gamma_max);

            if(drop_step) {
                S.remove_col(v_index);
                a.remove(v_index);
            }
            
            //Update a
            a.scale(1+gamma);
            if(!drop_step) {
                a(v_index) -= gamma;
            }
            
            //Update x
            x.axpy(gamma, dA);
        }

        //Update duality gap
        DT sum_x_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) { if(x(i) <= 0.0) sum_x_lt_0 += x(i); }
        duality_gap = std::abs(F_best - sum_x_lt_0);
        k++;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}

#endif
