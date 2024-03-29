#ifndef TMS_SUBMOD_MNP_ORDER_K_H
#define TMS_SUBMOD_MNP_ORDER_K_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../set_fn/submodular.h"
#include "../perf_log.h"
#include "mnp.h"

template<class DT>
std::vector<bool> mnp_order_k(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
{
    PerfLog::get().add_sequence("MNP CUMMULATIVE TIME");
    PerfLog::get().add_sequence("MNP DUALITY");

    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A(F.n);


    Vector<DT> x_hat(F.n);

    Vector<DT> w_base(F.n+1);
    Vector<DT> v_base(F.n+1);
    auto w = w_base.subvector(0, 1);
    w(0) = 1.0;

    Vector<DT> w2_base(F.n+1);

    //Initialize S and R.
    Matrix<DT> S_base(F.n,F.n+1);
    IncQRMatrix<DT> R_base(F.n+1);
    auto S = S_base.submatrix(0, 0, F.n, 1);
    auto R = R_base.submatrix(0, 1);

    Matrix<DT> S2_base(F.n,F.n+1);
    IncQRMatrix<DT> R2_base(F.n+1);

    Vector<DT> s0 = S.subcol(0);
    F.polyhedron_greedy_decending(wA, s0);
    R(0,0) = s0.norm2();
    DT pt_p_max = s0.dot(s0);

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

        //
        //Regular FW direction.
        //
        // Get p_hat using the greedy algorithm
        Vector<DT> p_hat = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);
        
        //Determine current duality gap.
        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A[i] = x_hat(i) <= 0.0;
        }

        DT sum_x_hat_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) {
            sum_x_hat_lt_0 += std::min(x_hat(i), 0.0);
        }
        DT duality_gap = std::abs(F_best - sum_x_hat_lt_0);

        //Test to see if we are done
        DT xt_p = x_hat.dot(p_hat);
        pt_p_max = std::max(p_hat.dot(p_hat), pt_p_max);
        if( xt_p > xt_x - tolerance * pt_p_max || duality_gap < eps) {
            PerfLog::get().log_sequence("MNP CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("MNP DUALITY", duality_gap);
            break;
        }

        Vector<DT> d(F.n);
        d.copy(p_hat);
        d.axpy(-1.0, x_hat); 
        DT alpha_minimizing = std::min(std::max((x_hat.dot(x_hat) - x_hat.dot(p_hat)) / d.dot(d), 0.0), 1.0); //The alpha that minimizes the norm
        DT alpha_max = alpha_minimizing / 2.0; //Hyperparameter

        // Speculative directions
        int64_t n_speculations = 64; //Hyperparameter
        DT best_alpha = 0.0;
        DT best_resulting_norm = x_hat.dot(x_hat);
        int64_t best_index = 0;
        for(int i = 0; i < n_speculations; i++) {
            //  Initialize data structures for this direction
            auto S2 = S2_base.submatrix(0, 0, S.height(), S.width());
            auto R2 = R2_base.submatrix(0, S.width());
            auto w2 = w2_base.subvector(0, S.width());
            S2.copy(S);
            R2.current_matrix().copy(R.current_matrix());
            w2.copy(w);

            //Get this speculative direction
            DT alpha = alpha_max * (double) i / n_speculations;
            Vector<DT> x_hat2(F.n);
            x_hat2.copy(x_hat);
            x_hat2.axpby(alpha, p_hat, 1.0 - alpha);
            Vector<DT> p_hat2 = S2_base.subcol(S2.width());
            F.polyhedron_greedy_ascending(x_hat2, p_hat2);
            
            //Enlarge S, R, w
            R2.add_col_inc_qr(S2, p_hat2);
            S2.enlarge_n(1);
            w2.enlarge(1);
            w2(w2.length()-1) = 0.0;

            //Do the update for this direction
            // Update x_hat order 2
            mnp_update_w(w2, v_base, S2, R2, tolerance);
            S2.mvm(1.0, w2, 0.0, x_hat2);
            DT xnrm2 = x_hat2.norm2();
            if(xnrm2 < best_resulting_norm) {
                best_resulting_norm = xnrm2;
                best_alpha = alpha;
                best_index = i;
            }
            //std::cout << std::endl << best_alpha << "\t" << alpha << "\t" << alpha_minimizing << "\t" << best_index << std::endl;

           // std::cout << std::setw(8) << xnrm2;
        }
//        std::cout << std::endl;

        //Actually go in the best direction
        DT alpha = best_alpha;
        x_hat.axpby(alpha, p_hat, 1.0 - alpha);
        F.polyhedron_greedy_ascending(x_hat, p_hat);

        //Enlarge S, R, w
        R.add_col_inc_qr(S, p_hat);
        S.enlarge_n(1);
        w.enlarge(1);
        w(w.length()-1) = 0.0;

        //Do (redundant) update.
        mnp_update_w(w, v_base, S, R, tolerance);
        S.mvm(1.0, w, 0.0, x_hat);

        PerfLog::get().log_total("S WIDTH", S.width());
        if(k % LOG_FREQ == 0) {
            PerfLog::get().log_sequence("MNP CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("MNP DUALITY", duality_gap);
        }
        k++;
    }

    wA.copy(x_hat);
    PerfLog::get().log_total("ITERATIONS", k);
    return A;
}

template<class DT>
std::vector<bool> mnp_order_k(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return mnp_order_k(F, wA, eps, tolerance);
}

#endif
