#ifndef TMS_SUBMOD_MNP_FW_H
#define TMS_SUBMOD_MNP_FW_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../set_fn/submodular.h"
#include "../perf_log.h"

/*
 * Slightly more favorable version of MNP that keeps track of w instead of x,
 * where x = Sw.
 *
 * Benefit comes from not having to do a solve during the case that Sw is not in conv(S)
 */

//At the end, y is equal to the new value of x_hat
//mu is a tmp vector with a length = m
template<class DT>
void mnp_fw_update_w(Vector<DT>& w, Vector<DT>& v_base, 
        Matrix<DT>& S, IncQRMatrix<DT>& R, DT tolerance)
{
    auto v = v_base.subvector(0, S.width());
    int64_t minor_cycles = 0;
    while(1) {
        minor_cycles++;

        //
        //Find minimum norm point in affine hull spanned by S
        //
        v.set_all(1.0);
        R.transpose(); R.trsv(v); R.transpose();
        R.trsv(v);
        v.scale(1.0 / v.sum());
        assert(!v.has_nan());

        //Check to see if y is written as positive convex combination of S
        if(v.min() > 0) break;

        // It's not a convex combination.
        
        // 
        // Project Sv back into polytope
        //
        
        // Find w for which Sw in conv(S) is closest to Sv 
        DT beta = 1.0;
        for(int64_t i = 0; i < S.width(); i++) {
            if(v(i) <= 0 /*1e-10*//*tolerance*/)
                beta = std::min(beta, w(i) / (w(i) - v(i)));
        }
        w.axpby(beta, v, 1.0 - beta);

        //
        // Remove some vectors from R and S
        //

        //Determine which columns of S and R are useless
        std::list<int64_t> to_remove;
        int64_t j = 0;
        for(int64_t i = 0; i < S.width(); i++){
            if(w(i) <= tolerance){
                to_remove.push_back(i);
            } else {
                v(j) = w(i);
                j++;
            }
        }
        if(to_remove.size() == 0) to_remove.push_back(w.index_of_min());
        assert(to_remove.size() > 0); 
        
        //Remove unnecessary columns from S and fixup R so that S = QR for some Q
        S.remove_cols(to_remove);
        R.remove_cols_inc_qr(to_remove);
        w.enlarge(-to_remove.size());
        v.enlarge(-to_remove.size());
        v.scale(1.0 / v.sum());
        w.copy(v);
    }
    PerfLog::get().log_total("MINOR CYCLES", minor_cycles);

    w.copy(v);
}

template<class DT>
std::vector<bool> mnp_fw(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
{
    PerfLog::get().add_sequence("MNP CUMMULATIVE TIME");
    PerfLog::get().add_sequence("MNP DUALITY");

    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A_curr(F.n);
    std::vector<bool> A_best(F.n);

    Vector<DT> d_FW(F.n);

    Vector<DT> x_hat(F.n);

    Vector<DT> w_base(F.n+1);
    Vector<DT> v_base(F.n+1);
    auto w = w_base.subvector(0, 1);
    w(0) = 1.0;

    //Initialize S and R.
    Matrix<DT> S_base(F.n,F.n+1);
    IncQRMatrix<DT> R_base(F.n+1);
    auto S = S_base.submatrix(0, 0, F.n, 1);
    auto R = R_base.submatrix(0, 1);

    Vector<DT> s0 = S.subcol(0);
    F.polyhedron_greedy_decending(wA, s0);
    R(0,0) = s0.norm2();

    int64_t k = 0;
    int64_t initial_time = rdtsc();
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
        Vector<DT> p_hat = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat, A_curr);
        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A_best[i] = A_curr[i];
        }

        // Get suboptimality bound
        DT sum_x_hat_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) {
            sum_x_hat_lt_0 += std::min(x_hat(i), 0.0);
        }
        DT duality_gap = F_best - sum_x_hat_lt_0;

        //Test to see if we are done
        DT xt_p = x_hat.dot(p_hat);
        if( xt_p > xt_x - tolerance || duality_gap < eps) {
            PerfLog::get().log_sequence("MNP CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("MNP DUALITY", duality_gap);
            break;
        }

        // Update R to account for modifying S.
        R.add_col_inc_qr(S, p_hat);
        S.enlarge_n(1);
        w.enlarge(1);
        w(w.length()-1) = 0.0;

        if(std::isnan(R(R.height()-1, R.width()-1))) {
            std::cout << "Warning: rho is nan.\n" << std::endl;
        }

        //In this case we necessarily already have our answer
        if(R(R.height()-1, R.width()-1) <= 1e-10) break; 
        if(std::isnan(R(R.height()-1, R.width()-1))) break; 

        //Take a FW step
        d_FW.copy(p_hat);
        d_FW.axpy(-1.0, x_hat);
        DT gamma = std::min(std::max(-x_hat.dot(d_FW) / d_FW.dot(d_FW), 0.0), 1.0);
        if(gamma == 0.0) {
            std::cout << "gamma is zero" << std::endl;
            break;
        } else if(gamma == 1.0) {
            std::cout << "gamma is 1.0!" << std::endl;
        }
        w.scale(1.0 - gamma);
        w(w.length()-1) = gamma;
        assert((1.0 - w.sum()) < 1e-10);

        // Update x_hat
        mnp_fw_update_w(w, v_base, S, R, 1e-10);
       
        PerfLog::get().log_total("S WIDTH", S.width());
        if(k % LOG_FREQ == 0) {
            PerfLog::get().log_sequence("MNP CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("MNP DUALITY", duality_gap);
        }
        k++;
    }

    wA.copy(x_hat);
    PerfLog::get().log_total("ITERATIONS", k);
    return A_best;
}

template<class DT>
std::vector<bool> mnp_fw(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return mnp_fw(F, wA, eps, tolerance);
}

#endif
