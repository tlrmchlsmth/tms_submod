#ifndef TMS_SUBMOD_BVH_H
#define TMS_SUBMOD_BVH_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../set_fn/submodular.h"
#include "../perf_log.h"

template<class DT>
void bvh_update_w_correction2(Vector<DT>& w, Vector<DT>& x,
        Matrix<DT>& S, Matrix<DT>& D, DT tolerance)
{
    double last_xtx = 0;
    //
    //Variable table.
    // x = x^t
    // S = B
    // y_g = g
    // y_d = h
    // w_d = u
    // w_b = v
    int iter = 0;
    int64_t minor_cycles = 0;
    while(1) {
        minor_cycles++;
        assert(w.min() > -tolerance);
        iter++;
        //Compute:
        //  x = Sw
        //  g = D^T 2x
        //  D = [b_1 - b_k, b_2 - b_k, ...]
        S.mvm(1.0, w, 0.0, x);
        auto w_linear = w.subvector(0, w.length()-1);

        auto s_k = S.subcol(S.width()-1);
        for(int64_t j = 0; j < D.width(); j++) {
            auto d_j = D.subcol(j);
            auto s_j = S.subcol(j);
            d_j.copy(s_j);
            d_j.axpy(-1.0, s_k);
        }

        //Make sure S and D are right
        /*
        double xtx = x.dot(x);
        Vector<DT> x2(x.length());
        D.mvm(1.0, w_linear, 0.0, x2);
        x2.axpy(1.0, s_k);
        std::cout << "xtx " << xtx << " x2tx2 " << x2.dot(x2) << std::endl;
        assert(std::abs(xtx - x2.dot(x2)) < tolerance);

        //Make sure x is decreasing 
        if(iter > 1){
            std::cout << "Minor cycle xtx " << xtx << " last xtx " << last_xtx << std::endl;
            assert(last_xtx - xtx > -tolerance);
        }
        last_xtx = xtx;
        */
        
        //Compute u
        Vector<DT> d(S.width());
        auto d_linear = d.subvector(0, d.length()-1);
        D.transposed().mvm(-2.0, x, 1.0, d_linear);

        Matrix<DT> DT_D(D.width(), D.width());
        DT_D.mmm(1.0, D.transposed(), D, 0.0); //should be syrk not mmm
        DT_D.chol('U');
        DT_D.transposed().trsv(d);
        DT_D.trsv(d);

        Vector<DT> u(S.width());
        auto u_linear = u.subvector(0, u.length()-1);
        u_linear.copy(w_linear);
        u.axpy(1.0, d);
        u(u.length()-1) = 1.0 - u_linear.sum();

        //Return to major cycle if u = w
        Vector<DT> d(S.width());
        d.copy(u);
        d.axpy(-1.0, w);
        if(d.abs_max() < tolerance) { 
            break;  //Or should it be 0.0 rather than tolerance?
        }
        
        //Compute lambda
        //Need to pick best lambda that gives us a valid v.
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
        auto v_linear = v.subvector(0, w.length()-1);
//        D.mvm(1.0, w_linear, 0.0, Dw); //w is 1 too long...
//        D.mvm(1.0, v_linear, 0.0, Dv);
        S.mvm(1.0, w, 0.0, Dw); //w is 1 too long...
        S.mvm(1.0, v, 0.0, Dv);
        Vector<DT> Dw_minus_Dv (D.height());
        Dw_minus_Dv.copy(Dw);
        Dw_minus_Dv.axpy(-1.0, Dv);
/*
        std::cout << "Dw^T Dw " << Dw.dot(Dw) << std::endl;
        std::cout << "Dv^T Dv " << Dv.dot(Dv) << std::endl;
        std::cout << "Dw^T Dv " << Dw.dot(Dv) << std::endl;
        std::cout << "(Dw-Dv)^2 " << Dw_minus_Dv.dot(Dw_minus_Dv) << std::endl;
*/
        DT alpha = (Dw.dot(Dw) - Dw.dot(Dv)) / Dw_minus_Dv.dot(Dw_minus_Dv); //This gives a zero of f' along the line from Dw to Dv. Is this a local max or min?
        alpha = std::min(std::max(0.0, alpha), 1.0);

        //Test the three possibilities
/*        Vector<DT> p1(S.height());
        Vector<DT> p2(S.height());
        Vector<DT> p3(S.height());
        Vector<DT> w1(w.length());
        w1.copy(w);
        w1.axpby(alpha, v, 1.0 - alpha);
        S.mvm(1.0, w, 0.0, p1);
        S.mvm(1.0, v, 0.0, p2);
        S.mvm(1.0, w1, 0.0, p3);
        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "w " << p1.dot(p1) << " v " << p2.dot(p2) << " interp " << p3.dot(p3) << std::endl;
        assert(v.abs_min() < tolerance);*/

        //assert(Dw.dot(Dw) + Dv.dot(Dv) < 2*Dw.dot(Dv)); //Assert that it is a local maximum

       
        if(alpha < 1.0 - tolerance) {
            w.axpby(alpha, v, 1.0 - alpha);

/*            if(std::abs(alpha) <= tolerance){
                w_linear = w.subvector(0, w.length()-1);
                w(w.length()-1) = 1.0 - w_linear.sum();
                S.mvm(1.0, w, 0.0, x);
                return;
            }*/
        } else {
            w.copy(v);
        }

        //Remove columns from B
        std::list<int64_t> to_remove;
        for(int64_t j = 0; j < w.length(); j++) {
            if(w(j) < tolerance)
                to_remove.push_back(j);
        }
        assert(alpha < 1.0 - tolerance || to_remove.size() > 0);
        S.remove_cols(to_remove);
        w.remove_elems(to_remove);
        D.enlarge_n(-to_remove.size());
        if(S.width() == 1) {
            auto s0 = S.subcol(0);
            x.copy(s0);
            std::cout << "Return C" << std::endl;
            break;
        }

        w_linear = w.subvector(0, w.length()-1);
        w(w.length()-1) = 1.0 - w_linear.sum();
        assert(w.min() > -tolerance);
    }

    PerfLog::get().log_total("MINOR CYCLES", minor_cycles);
}
template<class DT>
void bvh_update_w(Vector<DT>& w, Vector<DT>& x,
        Matrix<DT>& S, Matrix<DT>& D, DT tolerance)
{
    double last_xtx = 0;
    //
    //Variable table.
    // x = x^t
    // S = B
    // y_g = g
    // y_d = h
    // w_d = u
    // w_b = v
    int iter = 0;
    int64_t minor_cycles = 0;
    while(1) {
        minor_cycles++;
        assert(w.min() > -tolerance);
        iter++;
        //Compute:
        //  x = Sw
        //  g = D^T 2x
        //  D = [b_1 - b_k, b_2 - b_k, ...]
        S.mvm(1.0, w, 0.0, x);
        auto w_linear = w.subvector(0, w.length()-1);

        auto s_k = S.subcol(S.width()-1);
        for(int64_t j = 0; j < D.width(); j++) {
            auto d_j = D.subcol(j);
            auto s_j = S.subcol(j);
            d_j.copy(s_j);
            d_j.axpy(-1.0, s_k);
        }

        //Make sure S and D are right
        /*
        double xtx = x.dot(x);
        Vector<DT> x2(x.length());
        D.mvm(1.0, w_linear, 0.0, x2);
        x2.axpy(1.0, s_k);
        std::cout << "xtx " << xtx << " x2tx2 " << x2.dot(x2) << std::endl;
        assert(std::abs(xtx - x2.dot(x2)) < tolerance);

        //Make sure x is decreasing 
        if(iter > 1){
            std::cout << "Minor cycle xtx " << xtx << " last xtx " << last_xtx << std::endl;
            assert(last_xtx - xtx > -tolerance);
        }
        last_xtx = xtx;
        */
        
        //Compute u
        Vector<DT> u(S.width());
        auto u_linear = u.subvector(0, u.length()-1);
        u_linear.copy(w_linear);
        D.transposed().mvm(-2.0, x, 1.0, u_linear);
        u(u.length()-1) = 1.0 - u_linear.sum();

        //Return to major cycle if u = w
        Vector<DT> d(S.width());
        d.copy(u);
        d.axpy(-1.0, w);
        if(d.abs_max() < tolerance) { 
            break;  //Or should it be 0.0 rather than tolerance?
        }
        
        //Compute lambda
        //Need to pick best lambda that gives us a valid v.
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
        auto v_linear = v.subvector(0, w.length()-1);
//        D.mvm(1.0, w_linear, 0.0, Dw); //w is 1 too long...
//        D.mvm(1.0, v_linear, 0.0, Dv);
        S.mvm(1.0, w, 0.0, Dw); //w is 1 too long...
        S.mvm(1.0, v, 0.0, Dv);
        Vector<DT> Dw_minus_Dv (D.height());
        Dw_minus_Dv.copy(Dw);
        Dw_minus_Dv.axpy(-1.0, Dv);
/*
        std::cout << "Dw^T Dw " << Dw.dot(Dw) << std::endl;
        std::cout << "Dv^T Dv " << Dv.dot(Dv) << std::endl;
        std::cout << "Dw^T Dv " << Dw.dot(Dv) << std::endl;
        std::cout << "(Dw-Dv)^2 " << Dw_minus_Dv.dot(Dw_minus_Dv) << std::endl;
*/
        DT alpha = (Dw.dot(Dw) - Dw.dot(Dv)) / Dw_minus_Dv.dot(Dw_minus_Dv); //This gives a zero of f' along the line from Dw to Dv. Is this a local max or min?
        alpha = std::min(std::max(0.0, alpha), 1.0);

        //Test the three possibilities
/*        Vector<DT> p1(S.height());
        Vector<DT> p2(S.height());
        Vector<DT> p3(S.height());
        Vector<DT> w1(w.length());
        w1.copy(w);
        w1.axpby(alpha, v, 1.0 - alpha);
        S.mvm(1.0, w, 0.0, p1);
        S.mvm(1.0, v, 0.0, p2);
        S.mvm(1.0, w1, 0.0, p3);
        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "w " << p1.dot(p1) << " v " << p2.dot(p2) << " interp " << p3.dot(p3) << std::endl;
        assert(v.abs_min() < tolerance);*/

        //assert(Dw.dot(Dw) + Dv.dot(Dv) < 2*Dw.dot(Dv)); //Assert that it is a local maximum

       
        if(alpha < 1.0 - tolerance) {
            w.axpby(alpha, v, 1.0 - alpha);

/*            if(std::abs(alpha) <= tolerance){
                w_linear = w.subvector(0, w.length()-1);
                w(w.length()-1) = 1.0 - w_linear.sum();
                S.mvm(1.0, w, 0.0, x);
                return;
            }*/
        } else {
            w.copy(v);
        }

        //Remove columns from B
        std::list<int64_t> to_remove;
        for(int64_t j = 0; j < w.length(); j++) {
            if(w(j) < tolerance)
                to_remove.push_back(j);
        }
        assert(alpha < 1.0 - tolerance || to_remove.size() > 0);
        S.remove_cols(to_remove);
        w.remove_elems(to_remove);
        D.enlarge_n(-to_remove.size());
        if(S.width() == 1) {
            auto s0 = S.subcol(0);
            x.copy(s0);
            std::cout << "Return C" << std::endl;
            break;
        }

        w_linear = w.subvector(0, w.length()-1);
        w(w.length()-1) = 1.0 - w_linear.sum();
        assert(w.min() > -tolerance);
    }

    PerfLog::get().log_total("MINOR CYCLES", minor_cycles);
}

template<class DT>
std::vector<bool> bvh(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
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

    //Initialize S and space for D.
    Matrix<DT> S_base(F.n,F.n+2);
    auto S = S_base.submatrix(0, 0, F.n, 1);
    Matrix<DT> D_base(F.n,F.n+1);
    auto D = D_base.submatrix(0, 0, F.n, 0);

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
        }

        // Get p_hat using the greedy algorithm
        Vector<DT> p_hat = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);
        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A[i] = x_hat(i) <= 0.0;
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
//        std::cout << "Major cycle xtx " << xt_x << " last xtx " << last_xtx << std::endl;
        assert(last_xtx - xt_x > -tolerance);
        last_xtx = xt_x;
        pt_p_max = std::max(p_hat.dot(p_hat), pt_p_max);
//        std::cout << "Iteration: " << k << " Duality gap: " << duality_gap << " xTx: " << xt_x << " ptpmax " << pt_p_max << std::endl;
        if( xt_p > xt_x - tolerance * pt_p_max || duality_gap < eps) {
//            std::cout << "Done. Iteration: " << k << " Duality gap: " << duality_gap << " xTx: " << xt_x << " ptpmax " << pt_p_max << std::endl;
            break;
        }

        //Minor cycle
        
        //First take a FW step
        Vector<DT> d(F.n);
        d.copy(p_hat);
        d.axpy(-1.0, x_hat);
        DT gamma = std::min(std::max(-x_hat.dot(d) / d.dot(d), 0.0), .5);
        w.scale(1.0 - gamma);
        assert(w.min() > -tolerance);

        //Update x
        S.enlarge_n(1);
        D.enlarge_n(1);
        w.enlarge(1);
        w(w.length()-1) = gamma;
//       w(w.length()-1) = 0.0;
        assert(std::abs(1.0 - w.sum()) < tolerance);
       // bvh_update_w(w, x_hat, S, D, tolerance);
        bvh_update_w_correction2(w, x_hat, S, D, tolerance);

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

#include "mnp.h"
template<class DT>
std::vector<bool> bvh_test(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
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

    //Initialize S and space for D.
    Matrix<DT> S_base(F.n,F.n+2);
    auto S = S_base.submatrix(0, 0, F.n, 1);
    Matrix<DT> D_base(F.n,F.n+1);
    auto D = D_base.submatrix(0, 0, F.n, 0);


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
        }

        // Get p_hat using the greedy algorithm
        Vector<DT> p_hat = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);
        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A[i] = x_hat(i) <= 0.0;
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
//        std::cout << "Major cycle xtx " << xt_x << " last xtx " << last_xtx << std::endl;
        assert(last_xtx - xt_x > -tolerance);
        last_xtx = xt_x;
        pt_p_max = std::max(p_hat.dot(p_hat), pt_p_max);
//        std::cout << "Iteration: " << k << " Duality gap: " << duality_gap << " xTx: " << xt_x << " ptpmax " << pt_p_max << std::endl;
        if( xt_p > xt_x - tolerance * pt_p_max || duality_gap < eps) {
//            std::cout << "Done. Iteration: " << k << " Duality gap: " << duality_gap << " xTx: " << xt_x << " ptpmax " << pt_p_max << std::endl;
            break;
        }

        //Minor cycles

        //Resize matrices
        S.enlarge_n(1);
        D.enlarge_n(1);
        w.enlarge(1);
    
        //Setup data structures for MNP minor cycles
        Matrix<DT> S_old(F.n,S._n);
        S_old.copy(S);

        IncQRMatrix<DT> R(S._n);
        R._a.mmm(1.0, S.transposed(), S, 0.0);
        R._a.chol('U');
        R._am_a = true;
        R._am_upper_tri = true;
        
        Vector<DT> v(S._n);

        Vector<DT> w_old(S.width());
        w_old.copy(w);

        //Take a FW step before doing simplicical decomposition minor cycles
//        Vector<DT> d(F.n);
//        d.copy(p_hat);
//        d.axpy(-1.0, x_hat);
//        DT gamma = std::min(std::max(-x_hat.dot(d) / d.dot(d), 0.0), .5);
//        w.scale(1.0 - gamma);
//        w(w.length()-1) = gamma;
        w(w.length()-1) = 0.0;
        
        //Do simplicical decomposition minor cycles
        bvh_update_w(w, x_hat, S, D, tolerance); 

        //Do MNP minor cycles
        mnp_update_w(w_old, v, S_old, R, tolerance); 

        //Compare answers
        Vector<DT> x_mnp(F.n);
        S_old.mvm(1.0, w_old, 0.0, x_mnp);
        std::cout << std::setw(8) << k << std::setw(8) <<  S.width() << std::setw(8) << S_old.width() << " bvh xTx " << std::setw(8) << x_hat.dot(x_hat) << " mnp xTx " << std::setw(8) << x_mnp.dot(x_mnp);
        if(x_hat.dot(x_hat) < x_mnp.dot(x_mnp) - tolerance)
           std::cout << "\tGood!";
        std::cout << std::endl;
        assert(x_hat.dot(x_hat) <= x_mnp.dot(x_mnp) + tolerance);

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
std::vector<bool> bvh(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return bvh(F, wA, eps, tolerance);
}

template<class DT>
std::vector<bool> bvh_test(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    for(int64_t i = 0; i < F.n; i++)
        wA(i) = i;
    return bvh_test(F, wA, eps, tolerance);
}

#endif
