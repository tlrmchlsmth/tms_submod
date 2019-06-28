#ifndef TMS_SUBMOD_BVH_H
#define TMS_SUBMOD_BVH_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../set_fn/submodular.h"
#include "../perf_log.h"

template<class DT>
void recompute_D_R(const Matrix<DT>& S, Matrix<DT>& D, IncQRMatrix<DT>& R) {
    if(D.width() == 0) return;

    auto s_0 = S.subcol(0);
    for(int64_t j = 0; j < D.width(); j++) {
        auto d = D.subcol(j);
        auto s = S.subcol(j+1);
        d.copy(s);
        d.axpy(-1.0, s_0);
    }

    R.current_matrix().syrk(CblasUpper, 1.0, D.transposed(), 0.0);
    R.current_matrix().chol('U');

    PerfLog::get().log_total("RECOMPUTE DR", 1);
}

template<class DT>
void cleanup_cols(Vector<DT>& w, Matrix<DT>& S, Matrix<DT>& D, IncQRMatrix<DT>& R, DT tolerance) {
    std::list<int64_t> to_remove;
    std::list<int64_t> to_remove_D;
    bool remove_s0 = w(0) < tolerance;
    if(remove_s0) to_remove.push_back(0);
    for(int64_t j = 1; j < w.length(); j++) {
        if(w(j) < tolerance) {
            to_remove.push_back(j);
            to_remove_D.push_back(j-1);
        }
    }
    if(S.width() >= S.height() - 1 && to_remove.size() == 0) {
        auto w_linear = w.subvector(1, w.length()-1);
        to_remove.push_back(w_linear.index_of_abs_min());
        to_remove_D.push_back(w_linear.index_of_abs_min()-1);
    }

    S.remove_cols(to_remove);
    w.remove_elems(to_remove);
    //assert(alpha < 1.0 - tolerance || to_remove.size() > 0);
    if(remove_s0) {
        D.enlarge_n(-to_remove.size());
        R.enlarge_n(-to_remove.size());
        recompute_D_R(S, D, R);
    } else if(to_remove.size() > 0){
        D.remove_cols(to_remove_D);
        R.remove_cols_inc_qr(to_remove_D);
    }

    w.scale(1.0 / w.sum());
}

template<class DT>
void bvh_update_w(Vector<DT>& w, Vector<DT>& x,
        Matrix<DT>& S, Matrix<DT>& D, IncQRMatrix<DT>& R, DT tolerance, DT tolerance2)
{
    int64_t minor_cycles = 0;
    while(1) {
        minor_cycles++;
        assert(w.min() > -tolerance);
        //Compute:
        //  x = Sw
        //  g = D^T 2x
        //  D = [b_1 - b_k, b_2 - b_k, ...]
        S.mvm(1.0, w, 0.0, x);
        auto w_linear = w.subvector(1, w.length()-1);

        //Compute u
        Vector<DT> u(S.width());
        auto u_linear = u.subvector(1, u.length()-1);
        D.transposed().mvm(-2.0, x, 0.0, u_linear);
        R.transpose(); R.trsv(u_linear); R.transpose();
        R.trsv(u_linear); //Now u is the direction we will be moving in.

        //Return to major cycle if u = 0
        if(u_linear.dot(u_linear) < tolerance2) { 
            cleanup_cols(w, S, D, R, 1e-10);
            break;
        }
        u_linear.axpy(1.0, w_linear);
        u(0) = 1.0 - u_linear.sum();

        //Compute lambda
        //Pick best lambda that gives us a valid v.
        DT lambda = (w(0) - u(0)) / w(0);
        for(int64_t i = 1; i < w.length(); i++) {
            lambda = std::max(lambda, (w(i) - u(i)) / w(i));
        }
        lambda = 1.0 / lambda;
        
        if(std::abs(lambda) < tolerance2) {
            S.mvm(1.0, w, 0.0, x);
            cleanup_cols(w, S, D, R, tolerance);
            break;
        }
        Vector<DT> v(S.width());
        v.copy(w);
        v.axpby(lambda, u, 1.0 - lambda); 
        assert(std::abs(v.min()) < tolerance);

        //Compute alpha
        Vector<DT> Sw (S.height());
        Vector<DT> Sv (S.height());
        auto v_linear = v.subvector(1, v.length()-1);
        S.mvm(1.0, w, 0.0, Sw);
        S.mvm(1.0, v, 0.0, Sv);
        Vector<DT> Sw_minus_Sv (S.height());
        Sw_minus_Sv.copy(Sw);
        Sw_minus_Sv.axpy(-1.0, Sv);
        DT alpha = (Sw.dot(Sw) - Sw.dot(Sv)) / Sw_minus_Sv.dot(Sw_minus_Sv);
        alpha = std::min(std::max(0.0, alpha), 1.0);

        if(alpha < 1.0 - tolerance) {
            w.axpby(alpha, v, 1.0 - alpha);
        } else {
            w.copy(v);
        }

        cleanup_cols(w, S, D, R, tolerance);

        if(S.width() == 1) {
            auto s0 = S.subcol(0);
            x.copy(s0);
            break;
        }

        w_linear = w.subvector(1, w.length()-1);
        w(0) = 1.0 - w_linear.sum();
        assert(w.min() > -tolerance);
    }

    PerfLog::get().log_total("MINOR CYCLES", minor_cycles);
}

template<class DT>
std::vector<bool> bvh(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance, DT tolerance2) 
{
    PerfLog::get().add_sequence("BVH CUMMULATIVE TIME");
    PerfLog::get().add_sequence("BVH DUALITY");

    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A_best(F.n);
    std::vector<bool> A_curr(F.n);

    Vector<DT> d_FW(F.n);

    Vector<DT> x_hat(F.n);
    Vector<DT> w_base(F.n+2);
    Vector<DT> v_base(F.n+1);
    auto w = w_base.subvector(0, 1);
    w(0) = 1.0;

    //Initialize, S, D, R
    Matrix<DT> S_base(F.n,F.n+2);
    Matrix<DT> D_base(F.n,F.n+1);
    IncQRMatrix<DT> R_base(F.n+1);
    auto S = S_base.submatrix(0, 0, F.n, 1);
    auto D = D_base.submatrix(0, 0, F.n, 0);
    auto R = R_base.submatrix(0, 0);

    Vector<DT> s0 = S.subcol(0);
    F.polyhedron_greedy_decending(wA, s0);

    //Find current x
    S.mvm(1.0, w, 0.0, x_hat);
    int64_t k = 0;
    int64_t initial_time = rdtsc();
    while(1) {
        assert(S.width() <= F.n + 1);

        // Get p_hat using the greedy algorithm
        auto p_hat = S_base.subcol(S.width());
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
        DT duality_gap = std::abs(F_best - sum_x_hat_lt_0);

        //Test to see if we are done
        DT xt_p = x_hat.dot(p_hat);
        DT xt_x = x_hat.dot(x_hat);
        if( xt_p > xt_x - tolerance || duality_gap < eps ) {
            break;
        }

        //Update R and D to account for modifying S
        auto d_hat = D_base.subcol(D.width());
        d_hat.copy(p_hat);
        auto s0 = S.subcol(0);
        d_hat.axpy(-1.0, s0);
        R.add_col_inc_qr(D, d_hat);
        D.enlarge_n(1);
        S.enlarge_n(1);

        //If rho00 is 0, (and our math is right), we already have our answer
        if(R(R.height()-1, R.width()-1) <= 1e-10) break;
        if(std::isnan(R(R.height()-1, R.width()-1))) break; 

        //Take a FW step
        //(Just because we need to be in the interior of the polytope)
        d_FW.copy(p_hat);
        d_FW.axpy(-1.0, x_hat);
        DT gamma = std::min(std::max(-x_hat.dot(d_FW) / d_FW.dot(d_FW), 0.0), .1);
        w.scale(1.0 - gamma);
        assert(w.min() > -tolerance);
        w.enlarge(1);
        w(w.length()-1) = gamma;
        assert(std::abs(1.0 - w.sum()) < tolerance);

        //Minor cycle
        bvh_update_w(w, x_hat, S, D, R, tolerance, tolerance2);

        //Snap to zero
        if(x_hat.dot(x_hat) < tolerance)
            x_hat.set_all(0.0);
       
        PerfLog::get().log_total("S WIDTH", S.width());
        if(k % LOG_FREQ == 0) {
            PerfLog::get().log_sequence("BVH CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("BVH DUALITY", duality_gap);
        }
        k++;
    }

    wA.copy(x_hat);
    PerfLog::get().log_total("ITERATIONS", k);
    return A_best;
}


template<class DT>
std::vector<bool> bvh(SubmodularFunction<DT>& F, DT eps, DT tolerance, DT tolerance2) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return bvh(F, wA, eps, tolerance, tolerance2);
}

#endif
