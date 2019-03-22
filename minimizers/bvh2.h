#ifndef TMS_SUBMOD_BVH2_H
#define TMS_SUBMOD_BVH2_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../set_fn/submodular.h"
#include "../perf_log.h"

template<class DT>
void recompute_D_R(const Matrix<DT>& S, Matrix<DT>& D, IncQRMatrix<DT>& R) {
    auto s_0 = S.subcol(0);
    for(int64_t j = 0; j < D.width(); j++) {
        auto d = D.subcol(j);
        auto s = S.subcol(j+1);
        d.copy(s);
        d.axpy(-1.0, s_0);
    }

    R.current_matrix().mmm(1.0, D.transposed(), D, 0.0); //should be syrk not mmm
    R.current_matrix().chol('U');
}

template<class DT>
void bvh2_update_w(Vector<DT>& w, Vector<DT>& x,
        Matrix<DT>& S, Matrix<DT>& D, IncQRMatrix<DT>& R, DT tolerance)
{
    double last_xtx = 0;
    
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
        R.trsv(u_linear);

        //Return to major cycle if u = w
        if(u_linear.dot(u_linear) < tolerance) { 
            //if(u_linear.abs_max() < tolerance) { 
            break;  //Or should it be 0.0 rather than tolerance?
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

        if(std::abs(lambda) < 1e-10) {
            S.mvm(1.0, w, 0.0, x);
            std::cout << lambda << std::endl;
            std::cout << "Return B" << std::endl;
            break;
        }
        Vector<DT> v(S.width());
        v.copy(w);
        v.axpby(lambda, u, 1.0 - lambda); 
        assert(std::abs(v.min()) < tolerance);

        //Compute alpha
        Vector<DT> Dw (D.height());
        Vector<DT> Dv (D.height());
        auto v_linear = v.subvector(1, w.length()-1);
        S.mvm(1.0, w, 0.0, Dw); //w is 1 too long...
        S.mvm(1.0, v, 0.0, Dv);
        Vector<DT> Dw_minus_Dv (D.height());
        Dw_minus_Dv.copy(Dw);
        Dw_minus_Dv.axpy(-1.0, Dv);
        DT alpha = (Dw.dot(Dw) - Dw.dot(Dv)) / Dw_minus_Dv.dot(Dw_minus_Dv);
        alpha = std::min(std::max(0.0, alpha), 1.0);

        if(alpha < 1.0 - tolerance) {
            w.axpby(alpha, v, 1.0 - alpha);
        } else {
            w.copy(v);
        }

        //Remove columns from B
        std::list<int64_t> to_remove;
        bool remove_s0 = w(0) < tolerance;
        for(int64_t j = 0; j < w.length(); j++) {
            if(w(j) < tolerance)
                to_remove.push_back(j);
        }
        assert(alpha < 1.0 - tolerance || to_remove.size() > 0);
        S.remove_cols(to_remove);
        w.remove_elems(to_remove);
        if(remove_s0) {
            D.enlarge_n(-to_remove.size());
            R.enlarge_n(-to_remove.size());
            recompute_D_R(S, D, R);
        } else {
            D.remove_cols(to_remove);
            R.remove_cols_inc_qr(to_remove);
        }

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
std::vector<bool> bvh2(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
{
    PerfLog::get().add_sequence("BVH CUMMULATIVE TIME");
    PerfLog::get().add_sequence("BVH DUALITY");

    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A(F.n);

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
    DT pt_p_max = s0.dot(s0);

    //Find current x
    S.mvm(1.0, w, 0.0, x_hat);
    double last_xtx = x_hat.dot(x_hat);
    int64_t k = 0;
    int64_t initial_time = rdtsc();
    while(1) {
        //assert(S.width() <= F.n + 1);
        if(S.width() >= F.n + 1) {
            int64_t j_to_remove = w.index_of_abs_min();
            S.remove_col(j_to_remove);
            w.remove_elem(j_to_remove);
            auto w_linear = w.subvector(0, w.length()-1);
            w(w.length()-1) = 1.0 - w_linear.sum();
            D.enlarge_n(-1);
            exit(1);
        }

        // Get p_hat using the greedy algorithm
        auto p_hat = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);

        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A[i] = x_hat(i) <= 0.0;
        }

        //Update R and D to account for modifying S
        auto d_hat = D_base.subcol(D.width());
        d_hat.copy(p_hat);
        auto s0 = S.subcol(0);
        d_hat.axpy(-1.0, s0);
        R.add_col_inc_qr(D, d_hat);
        D.enlarge_n(1);
        S.enlarge_n(1);

        // Get suboptimality bound
        DT sum_x_hat_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) {
            sum_x_hat_lt_0 += std::min(x_hat(i), 0.0);
        }
        DT duality_gap = std::abs(F_best - sum_x_hat_lt_0);

        //Test to see if we are done
        DT xt_p = x_hat.dot(p_hat);
        DT xt_x = x_hat.dot(x_hat);
        assert(last_xtx - xt_x > -tolerance);
        last_xtx = xt_x;
        pt_p_max = std::max(p_hat.dot(p_hat), pt_p_max);
        if( xt_p > xt_x - tolerance * pt_p_max || duality_gap < eps) {
            break;
        }

        //Take a FW step
        Vector<DT> d(F.n);
        d.copy(p_hat);
        d.axpy(-1.0, x_hat);
        DT gamma = std::min(std::max(-x_hat.dot(d) / d.dot(d), 0.0), .5);
        w.scale(1.0 - gamma);
        assert(w.min() > -tolerance);
        w.enlarge(1);
        w(w.length()-1) = gamma;
        assert(std::abs(1.0 - w.sum()) < tolerance);

        //Minor cycle
        bvh2_update_w(w, x_hat, S, D, R, tolerance);

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
    return A;
}


template<class DT>
std::vector<bool> bvh2(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return bvh2(F, wA, eps, tolerance);
}

#endif
