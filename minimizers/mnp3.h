#ifndef TMS_SUBMOD_MNP3_H
#define TMS_SUBMOD_MNP3_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../la/list_matrix.h"
#include "../set_fn/submodular.h"
#include "../perf_log.h"

//At the end, y is equal to the new value of x_hat
//mu is a tmp vector with a length = m
template<class DT>
void mnp3_update_w(Vector<DT>& w, Vector<DT>& v_base, 
        ColListMatrix<DT>& S, IncQRMatrix<DT>& R, DT tolerance)
{
    auto v = v_base.subvector(0, S.width());

    while(1) {
        //
        //Find minimum norm point in affine hull spanned by S
        //
        v.set_all(1.0);
        R.transpose(); R.trsv(v); R.transpose();
        R.trsv(v);
        v.scale(1.0 / v.sum());

        //Check to see if y is written as positive convex combination of S
        if(v.min() > tolerance) break;

        // It's not a convex combination.
        
        // 
        // Project Sv back into polytope
        //
        
        // Find w for which Sw in conv(S) is closest to Sv 
        DT beta = 1.0;
        for(int64_t i = 0; i < S.width(); i++) {
            if(v(i) < tolerance)
                beta = std::min(beta, w(i) / (w(i) - v(i)));
        }
        w.axpby(beta, v, 1.0 - beta);

        //
        // Remove some vectors from R and S
        //

        //Determine which columns of S and R are useless
        std::list<int64_t> toRemove;
        int64_t j = 0;
        for(int64_t i = 0; i < S.width(); i++){
            if(w(i) <= tolerance){
                toRemove.push_back(i);
            } else {
                v(j) = w(i);
                j++;
            }
        }
        assert(toRemove.size() > 0); 
        
        //Remove unnecessary columns from S and fixup R so that S = QR for some Q
        S.remove_cols(toRemove);
        R.remove_cols_inc_qr(toRemove);
        w.enlarge(-toRemove.size());
        v.enlarge(-toRemove.size());
        w.copy(v);
    }

    w.copy(v);
}

template<class DT>
std::vector<bool> mnp3(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
{
    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A(F.n);

    Vector<DT> x_hat(F.n);

    Vector<DT> w_base(F.n+1);
    Vector<DT> v_base(F.n+1);
    auto w = w_base.subvector(0, 1);
    w(0) = 1.0;

    //Initialize S and R.
    ColListMatrix<DT> S(F.n, F.n+1);
    IncQRMatrix<DT> R_base(F.n+1);
    auto R = R_base.submatrix(0, 1);

    auto s0 = S.next_col();
    F.polyhedron_greedy_decending(wA, s0);
    R(0,0) = s0.norm2();
    DT pt_p_max = s0.dot(s0);
    S.enlarge_width();

    int64_t major_cycles = 0;
    while(1) {
        assert(S.width() <= F.n);
        
        //Find current x
        S.mvm(1.0, w, 0.0, x_hat);
        DT xnrm = x_hat.norm2();
        DT xt_x = xnrm * xnrm;

        //Snap to zero
        if(xt_x < tolerance) {
            x_hat.set_all(0.0);
            xt_x = 0.0;
            xnrm = 0.0;
        }

        // Get p_hat using the greedy algorithm
        auto p_hat = S.next_col();
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);

        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A[i] = x_hat(i) <= 0.0;
        }
        
        // Update R to account for modifying S.
        R.add_col_inc_qr(S, p_hat);
        S.enlarge_width();
        w.enlarge(1);
        w(w.length()-1) = 0.0;

        // Get suboptimality bound
        DT sum_x_hat_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) {
            sum_x_hat_lt_0 += std::min(x_hat(i), 0.0);
        }

        //Test to see if we are done
        DT xt_p = x_hat.dot(p_hat);
        pt_p_max = std::max(p_hat.dot(p_hat), pt_p_max);
        if( xt_p > xt_x - tolerance * pt_p_max || std::abs(F_best - sum_x_hat_lt_0) < eps) break;

        // Update x_hat
        mnp3_update_w(w, v_base, S, R, tolerance);
       
        major_cycles++;
        PerfLog::get().log_total("S WIDTH", S.width());
    }

    PerfLog::get().log_total("ITERATIONS", major_cycles);
    return A;
}

template<class DT>
std::vector<bool> mnp3(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return mnp3(F, wA, eps, tolerance);
}

#endif
