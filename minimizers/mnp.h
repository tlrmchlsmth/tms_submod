#ifndef TMS_SUBMOD_MNP_H
#define TMS_SUBMOD_MNP_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../submodular.h"
#include "../perf_log.h"
#include "../perf/perf.h"

//R is upper triangular
template<class DT>
DT check_STS_eq_RTR(Matrix<DT>& S, Matrix<DT>& R) 
{
    Vector<DT> y(S.width());
    y.fill_rand();

    Vector<DT> Sy(S.height());
    Vector<DT> STSy(S.width());
    auto ST = S.transposed();
    S.mvm(1.0, y, 0.0, Sy); 
    ST.mvm(1.0, Sy, 0.0, STSy); 

    Vector<DT> Ry(R.height());
    Vector<DT> RTRy(R.width());
    auto RT = R.transposed();
    R.set_subdiagonal(0.0);

    R.mvm(1.0, y, 0.0, Ry); 
    RT.mvm(1.0, Ry, 0.0, RTRy); 

    RTRy.axpy(-1.0, STSy);
    return RTRy.norm2();
}

//R is upper triangular
template<class DT>
DT check_S_eq_RTR(Matrix<DT>& S, Matrix<DT>& R) 
{
    Vector<DT> y(S.width());
    y.fill_rand();

    Vector<DT> Sy(S.height());
    S.mvm(1.0, y, 0.0, Sy); 

    Vector<DT> Ry(R.height());
    Vector<DT> RTRy(R.width());
    auto RT = R.transposed();
    R.set_subdiagonal(0.0);

    R.mvm(1.0, y, 0.0, Ry); 
    RT.mvm(1.0, Ry, 0.0, RTRy); 

    RTRy.axpy(-1.0, Sy);
    return RTRy.norm2();
}

template<class DT>
class MinNormPoint
{
public:
    MinNormPoint() {}

    //At the end, y is equal to the new value of x_hat
    //mu is a tmp vector with a length = m
    //Returns: whether R and R_new should be swapped
    bool min_norm_point_update_xhat(Vector<DT>& x_hat, Vector<DT>& y,
            Vector<DT>& mu_ws, Vector<DT>& lambda_ws,
            Matrix<DT>& S, Matrix<DT>* R_in, Matrix<DT>* R_next_in, DT tolerance,
            Matrix<DT>& T, Matrix<DT>& H, int64_t nb,
            Matrix<DT>& QR_ws,
            PerfLog* perf_log)
    {
        Matrix<DT>* R = R_in;
        Matrix<DT>* R_next = R_next_in;
        bool to_ret = false;

        DT x_hat_nrm2 = x_hat.norm2();

        bool keep_going = true;
        while(keep_going) {
            int64_t minor_start = rdtsc();

            auto mu = mu_ws.subvector(0, R->width()); //mu.perf_log = perf_log; Don't log mu so we don't overcount
            auto lambda = lambda_ws.subvector(0, R->width()); //lambda.perf_log = perf_log;

            //Find minimum norm point in affine hull spanned by S
            int64_t solve_start = rdtsc();
            mu.set_all(1.0);
                int64_t trsv1_start = rdtsc();
            R->transpose(); R->trsv(CblasLower, mu); R->transpose();
                if(perf_log) { 
                    perf_log->log_total("SOLVE1 TRSV1 TIME", rdtsc() - trsv1_start); 
                    perf_log->log_total("SOLVE1 TRSV1 FLOPS", R->width() * R->width()); 
                    perf_log->log_total("SOLVE1 TRSV1 BYTES", sizeof(DT) * (R->width() * R->width() / 2.0 + 2.0*R->width())); 
                }

                int64_t trsv2_start = rdtsc();
            R->trsv(CblasUpper, mu);
                if(perf_log) { 
                    perf_log->log_total("SOLVE1 TRSV2 TIME", rdtsc() - trsv2_start); 
                    perf_log->log_total("SOLVE1 TRSV2 FLOPS", R->width() * R->width()); 
                    perf_log->log_total("SOLVE1 TRSV2 BYTES", sizeof(DT) * (R->width() * R->width() / 2.0 + 2.0*R->width())); 
                }

            mu.scale(1.0 / mu.sum());
                int64_t mvm_start = rdtsc();
            S.mvm(1.0, mu, 0.0, y);
                if(perf_log) { 
                    perf_log->log_total("SOLVE1 MVM TIME", rdtsc() - mvm_start); 
                    perf_log->log_total("SOLVE1 MVM FLOPS", 2*S.width() * S.height()); 
                    perf_log->log_total("SOLVE1 MVM BYTES", sizeof(DT) * (S.width() * S.height() + 2.0*S.height() + S.width())); 
                }
            if(perf_log) perf_log->log_total("SOLVE TIME", rdtsc() - solve_start);
                if(perf_log) perf_log->log_total("SOLVE1 TIME", rdtsc() - solve_start);

            //Check to see if y is written as positive convex combination of S
            if(mu.min() > -tolerance) {
                keep_going = false;
            } else {
                // Step 4:
                // It's not a convex combination
                // Project y back into polytope and remove some vectors from S
                
                // Get representation of xhat in terms of S; enforce that we get
                // affine combination (i.e., sum(lambda)==1)
                int64_t solve_start = rdtsc();
                S.transpose(); S.mvm(1.0, x_hat, 0.0, lambda); S.transpose();
                    if(perf_log) { 
                        perf_log->log_total("SOLVE2 MVM TIME", rdtsc() - solve_start); 
                        perf_log->log_total("SOLVE2 MVM FLOPS", 2*S.width() * S.height()); 
                        perf_log->log_total("SOLVE2 MVM BYTES", sizeof(DT) * (S.width() * S.height() + 2.0*S.height() + S.width())); 
                    }
                    int64_t trsv1_start = rdtsc();
                R->transpose(); R->trsv(CblasLower, lambda); R->transpose();
                    if(perf_log) { 
                        perf_log->log_total("SOLVE2 TRSV1 TIME", rdtsc() - trsv1_start); 
                        perf_log->log_total("SOLVE2 TRSV1 FLOPS", R->width() * R->width()); 
                        perf_log->log_total("SOLVE2 TRSV1 BYTES", sizeof(DT) * (R->width() * R->width() / 2.0 + 2.0*R->width())); 
                    }
                    int64_t trsv2_start = rdtsc();
                R->trsv(CblasUpper, lambda);
                    if(perf_log) { 
                        perf_log->log_total("SOLVE2 TRSV2 TIME", rdtsc() - trsv2_start); 
                        perf_log->log_total("SOLVE2 TRSV2 FLOPS", R->width() * R->width()); 
                        perf_log->log_total("SOLVE2 TRSV2 BYTES", sizeof(DT) * (R->width() * R->width() / 2.0 + 2.0*R->width())); 
                    }
                lambda.scale(1.0 / lambda.sum());
                if(perf_log) perf_log->log_total("SOLVE2 TIME", rdtsc() - solve_start);


                //Note: it is imperitive to not let z drift out of the convex hull of S.
                int64_t z_start = rdtsc();
                // Find z in conv(S) that is closest to y
                //DT beta = std::numeric_limits<DT>::max();
                DT beta = 1.0;
                for(int64_t i = 0; i < lambda.length(); i++) {
                    DT bound = 1.0;
                      if(mu(i) < tolerance) {
                        bound = lambda(i) / (lambda(i) - mu(i));
                    }
                    if(bound > tolerance && bound < beta) {
                        beta = bound;
                    }
                }
                if(perf_log) perf_log->log_total("MISC TIME", rdtsc() - z_start);

                x_hat.axpby(beta, y, (1-beta));

                int64_t remove_start = rdtsc();
                std::list<int64_t> toRemove; //TODO: pre initialize
                for(int64_t i = 0; i < lambda.length(); i++){
                    if((1-beta) * lambda(i) + beta * mu(i) <= tolerance)
                        toRemove.push_back(i);
                }

                if(perf_log) perf_log->log_hist("COLUMNS REMOVED", toRemove.size());
                if(toRemove.size() == 0) {
                    std::cout << "Warning: no columns to remove!" << std::endl;
                    toRemove.push_back(0);
                }
                if(perf_log) perf_log->log_total("MISC TIME", rdtsc() - remove_start);
                
                //Remove unnecessary columns from S and fixup R so that S = QR for some Q
                S.remove_cols(toRemove);
                R_next->_m = R->_m;
                R_next->_n = R->_n;
                R->remove_cols_incremental_qr_tasks_kressner(*R_next, toRemove, T, H, 128, 32, QR_ws);
                std::swap(R, R_next);
                to_ret = !to_ret;
            }

            int64_t minor_end = rdtsc();
            if(perf_log) perf_log->log_total("MINOR TIME", minor_end - minor_start);
        }
        return to_ret;
    }

    std::vector<bool> minimize(SubmodularFunction<DT>& F) 
    {
        bool done = false;
        Vector<DT> wA(F.n);
        wA.fill_rand();

        return this->minimize(F, wA, &done, 1000000, 1e-10, 1e-10, false, NULL);
    }

    std::vector<bool> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance, bool print) 
    {
        bool done = false;
        Vector<DT> wA(F.n);
        wA.fill_rand();

        return this->minimize(F, wA, &done, 1000000, eps, tolerance, print, NULL);
    }
    std::vector<bool> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance) 
    {
        bool done = false;
        Vector<DT> wA(F.n);
        wA.fill_rand();

        return this->minimize(F, wA, &done, 1000000, eps, tolerance, false, NULL);
    }
    
    std::vector<bool> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance, bool print, PerfLog* perf_log)
    {
        bool done = false;
        Vector<DT> wA(F.n);
        wA.fill_rand();
        return  minimize(F, wA, &done, 1000000, eps, tolerance, print, perf_log);
    }

    std::vector<bool> minimize(SubmodularFunction<DT>& F, Vector<DT>& wA, bool* done, int64_t max_iter, DT eps, DT tolerance, bool print, PerfLog* perf_log) 
    {
        int64_t eval_F_freq = 10;
        int64_t cycles_since_last_F_eval = eval_F_freq;
        int64_t m = F.n;
        *done = false;

        //To return
        std::vector<bool> A(m);
        DT F_best = std::numeric_limits<DT>::max();

        //Characteristic vector
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        //Workspace for x_hat and next x_hat
        Vector<DT> xh1(m);
        Vector<DT> xh2(m);
        Vector<DT>* x_hat = &xh1;
        Vector<DT>* x_hat_next = &xh2;
        xh1.perf_log = perf_log;
        xh2.perf_log = perf_log;

        //Workspace for updating x_hat
        int64_t nb = 32;
        int64_t max_removed_cols = m/2;
        Vector<DT> mu_ws(m);
        Vector<DT> lambda_ws(m);
        Matrix<DT> T(nb, m);
        Matrix<DT> H(max_removed_cols, m); //TODO: handle case where we remove more than this
        Matrix<DT> QR_ws(nb, m);

        //Initialize S and R.
        Matrix<DT> S_base(m,m+1);
        Matrix<DT> R_base1(m+1,m+1);
        Matrix<DT> R_base2(m+1,m+1);

        Matrix<DT> S = S_base.submatrix(0, 0, m, 1); S.perf_log = perf_log;

        //2 matrices for R, so we can do out of place column removal
        Matrix<DT> R1 = R_base1.submatrix(0, 0, 1, 1);  R1.perf_log = perf_log;
        Matrix<DT> R2 = R_base2.submatrix(0, 0, 1, 1);  R2.perf_log = perf_log;
        Matrix<DT>* R_base = &R_base1; Matrix<DT>* R_base_next = &R_base2;
        Matrix<DT>* R = &R1; Matrix<DT>* R_next = &R2;

        Vector<DT> first_col_s = S.subcol(0);
        F.polyhedron_greedy(1.0, wA, first_col_s, perf_log);
        (*x_hat).copy(first_col_s);
        (*R)(0,0) = first_col_s.norm2();


        if(perf_log) perf_log->add_histogram("NUM COLUMNS", 0, m, 50);
        if(perf_log) perf_log->add_histogram("COLUMNS REMOVED", 0, m/4, 50);
    
        DT pj_max = 0.0;

        //Step 2:
        int64_t major_cycles = 0;
        while(major_cycles++ < max_iter) {
            if(perf_log) perf_log->log_hist("NUM COLUMNS", S.width());
            int64_t major_start = rdtsc();

            //Snap to zero
            DT x_hat_norm2 = x_hat->norm2();
            if(x_hat_norm2*x_hat_norm2 < tolerance)
                (*x_hat).set_all(0.0);

            // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
            Vector<DT> p_hat = S_base.subcol(S.width()); p_hat.perf_log = perf_log;
            DT F_curr = F.polyhedron_greedy_eval(-1.0, *x_hat, p_hat, perf_log);
            pj_max = std::max(p_hat.dot(p_hat), pj_max);

            if (F_curr < F_best) {
                F_best = F_curr;
//                std::fill(A.begin(), A.end(), 0);
                for(int64_t i = 0; i < x_hat->length(); i++)
                    A[i] = (*x_hat)(i) <= 0.0;
                   // if((*x_hat)(i) <= 0.0) A[i] = 1;
            }
            
            // Update R to account for modifying S.
            // Let [r0 rho1]^T be the vector to add to r
            // r0 = R' \ (S' * p_hat)
            int64_t add_col_start = rdtsc();
            Vector<DT> r0 = R_base->subcol(0, R->width(), R->height()); r0.perf_log = perf_log;

                int64_t add_col_mvm_start = rdtsc();
            S.transpose(); S.mvm(1.0, p_hat, 0.0, r0); S.transpose();
                if(perf_log) { 
                    perf_log->log_total("ADD COL MVM TIME", rdtsc() - add_col_mvm_start); 
                    perf_log->log_total("ADD COL MVM FLOPS", 2*S.width() * S.height()); 
                    perf_log->log_total("ADD COL MVM BYTES", sizeof(DT) * (S.width() * S.height() + 2.0*S.height() + S.width())); 
                }

                int64_t add_col_trsv_start = rdtsc();
            R->transpose(); R->trsv(CblasLower, r0); R->transpose();
                if(perf_log) { 
                    perf_log->log_total("ADD COL TRSV TIME", rdtsc() - add_col_trsv_start); 
                    perf_log->log_total("ADD COL TRSV FLOPS", R->width() * R->width()); 
                    perf_log->log_total("ADD COL TRSV BYTES", sizeof(DT) * (R->width() * R->width() / 2.0 + 2.0*R->width())); 
                }

            // rho1^2 = p_hat' * p_hat - r0' * r0;
            DT phat_norm2 = p_hat.norm2();
            DT r0_norm2 = r0.norm2();
            DT rho1 = sqrt(std::abs(phat_norm2*phat_norm2 - r0_norm2*r0_norm2));

            R->enlarge_m(1); R->enlarge_n(1);
            (*R)(R->width()-1, R->height()-1) = rho1;
            S.enlarge_n(1);
            if(perf_log) { perf_log->log_total("ADD COL TIME", rdtsc() - add_col_start); }

            //Slow version of checking current function value
            /*std::unordered_set<int64_t> A_curr;
            int64_t eval_start = rdtsc();
            for(int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) <= 0.0) A_curr.insert(i);
            }
            auto F_curr2 = F.eval(A_curr);
            if(std::abs(F_curr - F_curr2) > 1e-5) std::cout << F_curr - F_curr2 << std::endl;
            F_curr = F_curr2;
            if (F_curr < F_best) 
                F_best = F_curr;
            if(perf_log) perf_log->log_total("EVAL F TIME", rdtsc() - eval_start);*/

            int64_t misc_start = rdtsc();
            // Get suboptimality bound
            DT sum_x_hat_lt_0 = 0.0;
            for (int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) <= 0.0)
                    sum_x_hat_lt_0 += (*x_hat)(i);
            }
//            DT subopt = F_best - sum_x_hat_lt_0;
            if(perf_log) perf_log->log_total("MISC TIME", rdtsc() - misc_start);

//            if(print || major_cycles % 100 == 0) { std::cout << "Suboptimality bound: " << F_best-subopt << " <= min_A F(A) <= F(A_best) = " << F_best << "; delta <= " << subopt << std::endl; }

            DT xt_p = x_hat->dot(p_hat);
            DT xnrm = x_hat->norm2();
            DT xt_x = xnrm * xnrm;
//            if(print || major_cycles % 100 == 0) std::cout << "x'p " << xt_p << " x'x " << xt_x << std::endl;
//            if (std::abs(xt_p - xt_x) < tolerance || std::abs(F_best - sum_x_hat_lt_0) < eps) {
//            if( xt_x - xt_p  < tolerance || std::abs(F_best - sum_x_hat_lt_0) < eps) {
            if( xt_p > xt_x - tolerance * pj_max || std::abs(F_best - sum_x_hat_lt_0) < eps) {
                // We are done: x_hat is already closest norm point
//                if (std::abs(xt_p - xt_x) < tolerance) subopt = 0.0;
                break;
            } else if (std::abs(xt_p + xt_x) < tolerance) {
                std::cout << "Setting x hat to zero" << std::endl;
                //We had to go through 0 to get to p_hat from x_hat.
                x_hat_next->set_all(0.0);
            } else {
                bool switch_R = min_norm_point_update_xhat(*x_hat, *x_hat_next,
                        mu_ws, lambda_ws,
                        S, R, R_next, tolerance,
                        T, H, nb, QR_ws,
                        perf_log);
                if(switch_R) {
                    std::swap(R, R_next);
                    std::swap(R_base, R_base_next);
                }
            }
            if(x_hat_next->has_nan()) exit(1);
           
            //x hat should improve every iteration. 
//            DT x_hat_nrm2 = x_hat_next->norm2();
//            if(x_hat_nrm2 > xnrm2) 
//                std::cout << "x_hat isn't improving" << std::endl;

//            DT xnextnrm = x_hat_next->norm2();
//            if(xnextnrm > xnrm) {
//                std::cout << "|xnext| " << xnextnrm << " |x| " << xnrm << std::endl;
//            }

            std::swap(x_hat, x_hat_next);

            if(perf_log) perf_log->log_total("MAJOR TIME", rdtsc() - major_start);
        }

        //Return
//        if(print){
//            std::cout << "Done. |A| = " << A.size() << " F = " << F.eval(A) << std::endl;
//        }
        if(major_cycles < max_iter) {
            *done = true;
        }
        return A;
    }
};

//Blocked Randomized S? I can't remember what BRS was supposed to stand for
template<class DT>
class BRSMinNormPoint
{
    int64_t b;
public:
    BRSMinNormPoint(int64_t blksz) : b(blksz) {}

    void scramble(std::vector<int64_t>& v) {
        std::random_device rd;
        std::mt19937 gen{rd()};
        for(int64_t i = 0; i < v.size()-2; i++) {
            std::uniform_int_distribution<int64_t> dist(i, v.size()-1);
            int64_t j = dist(gen);
            std::swap(v[i], v[j]);
        }
    }

    DT speculate(SubmodularFunction<DT>& F, Matrix<DT>& P_hat, Vector<DT>& x_hat, DT alpha, PerfLog* perf_log)
    {
        //Determine first col of P_hat with Edmonds greedy
        auto p_hat_0 = P_hat.subcol(0);
        DT F_curr = F.polyhedron_greedy_eval(alpha, x_hat, p_hat_0, perf_log);

        //Add more columns to P. We want each of them to be extreme points
        //x in B(f) is an extreme point in B(f) iff for some maximal chain 0 = S0 subset of S1 subset of ... subset of Sn = V
        //x(Si - Si-1) = f(Si) - f(Si-1)
        //
        int64_t start_spec = rdtsc();
        for(int64_t j = 1; j < P_hat.width(); j++) {
            auto p_hat_j = P_hat.subcol(j);
            scramble(F.permutation);
            F.gains(F.permutation, p_hat_j);
        }
        if(perf_log) perf_log->log_total("SPECULATION TIME", rdtsc() - start_spec);

        return F_curr;
    }

    //At the end, y is equal to the new value of x_hat
    //mu is a tmp vector with a length = m
    //Returns: whether R and R_new should be swapped
    bool min_norm_point_update_xhat(Vector<DT>& x_hat, Vector<DT>& y,
            Matrix<DT>& Y_ws, int64_t b,
            Matrix<DT>& W_ws, Matrix<DT>& Lambda_ws,
            Matrix<DT>& S, Matrix<DT>* R_in, Matrix<DT>* R_next_in, DT tolerance,
            Matrix<DT>& T, Matrix<DT>& H, int64_t nb,
            Matrix<DT>& QR_ws,
            PerfLog* perf_log)
    {

        Matrix<DT>* R = R_in;
        Matrix<DT>* R_next = R_next_in;
        bool to_ret = false;
        int64_t minor_start = rdtsc();
        std::list<int64_t> toRemove; //TODO: pre initialize

        //
        //First see if one of our b candidates gives us a valid x_hat_next.
        //
        
        //Partition R
        int64_t n = R->height()-1;
        auto R00 = R->submatrix(0, 0, n, n);
        auto R01 = R->submatrix(0, n, n, b);
        auto r11 = R->subrow(n, n, b);
       
        //Partition W
        auto W  = W_ws.submatrix(0, 0, n+1, b);
        auto W0 = W_ws.submatrix(0, 0, n, b);
        auto w1 = W_ws.subrow(n, 0, b);
        auto w00 = W0.subcol(0);
        w00.set_all(1.0);
        R00.transpose(); R00.trsv(CblasLower, w00); R00.transpose();

        w1.set_all(1.0);
        R01.transpose(); R01.mvm(-1.0, w00, 1.0, w1); R01.transpose();
        for(int64_t j = 0; j < b; j++) {
            w1(j) = w1(j) / (r11(j) * r11(j)); //Last step of forward pass, along with first step of backwards pass
        }
        for(int64_t j = 1; j < b; j++) {
            auto w0j = W0.subcol(j);
            w0j.copy(w00);
        }
        for(int64_t j = 0; j < b; j++) {
            auto w0j = W0.subcol(j);
            auto r01 = R01.subcol(j);
            w0j.axpy(-w1(j), r01);
        }
        R00.trsm(CblasUpper, CblasLeft, W0);

        for(int64_t j = 0; j < b; j++) {
            auto w_j = W.subcol(j);
            w_j.scale(1.0 / w_j.sum());
        }

        auto Y  = Y_ws.submatrix(0, 0, S.height(), b);  Y.perf_log = perf_log;
        auto S0 =    S.submatrix(0, 0, S.height(), n); S0.perf_log = perf_log;
        Y.mmm(1.0, S0, W0, 0.0);
        for(int64_t j = 0; j < b; j++) {
            auto y_j = Y.subcol(j);
            auto s_j = S.subcol(n+j);
            y_j.axpy(w1(j), s_j);
        }

        //Take the best column of Y that is written as a positive convex combination of S
        int64_t best = -1;
        DT best_norm = std::numeric_limits<DT>::max();
        for(int64_t j = 0; j < b; j++) {
            auto w = W.subcol(j);
            if((j==0 && w.min() >= -tolerance) || w.min() > 1e-10) {
                auto yj = Y.subcol(j);
                DT y_norm = yj.norm2();
                if(y_norm < best_norm) {
                    best = j;
                    best_norm = y_norm;
                }
            }
        }
        if(best > -1) {
            //We're going to take the best value, and throw away the rest.
            toRemove.clear(); 
            auto y_best = Y.subcol(best);
            for(int64_t j = 0; j < b; j++) {
                if(j != best) {
                    toRemove.push_back(j + S.width() - b);
                }
            }
            S.remove_cols(toRemove);
            R->remove_cols(toRemove);
            assert(R->height() == R->width());

            //Doublecheck that STS q = RTR q
//            DT error = check_STS_eq_RTR(S, *R); 
//            if(error > 1e-10) {
//                std::cout << "In xhat update: ||STS q - RTR q|| " << error << std::endl;
//                exit(1);
//            }

            y.copy(y_best);
        } else {
            // 
            //Step 4. 
            //None of the candidates are in the convex hull of S
            //Right now, we choose the 0th candidate.
            //TODO: Change this. We should choose the candidate that yields the minimum norm z
            //
            S.enlarge_n(-b+1);
            R->enlarge_n(-b+1);
            assert(R->height() == R->width());

            auto y_best = Y.subcol(0);
            y.copy(y_best);

            bool x_hat_in_conv_S = false;

            while(!x_hat_in_conv_S) 
            {
                // Project y back into polytope and remove some vectors from S
                auto w = W.subcol(0, 0, R->height()); w.perf_log = perf_log;
                auto lambda = Lambda_ws.subcol(0, 0, R->height()); lambda.perf_log = perf_log;
                assert(R->height() == R->width());

                // Get representation of xhat in terms of S; enforce that we get
                // affine combination (i.e., sum(lambda)==1)
                auto ST = S.transposed();
                auto RT = R->transposed();
                ST.mvm(1.0, x_hat, 0.0, lambda);
                RT.trsv(CblasLower, lambda);
                R->trsv(CblasUpper, lambda);
                lambda.scale(1.0 / lambda.sum());
               
                DT error = check_STS_eq_RTR(S, *R);

                int64_t z_start = rdtsc();
                // Find z in conv(S) that is closest to y
                //TODO: is this beta right
                DT beta = 1.0;
                for(int64_t i = 0; i < lambda.length(); i++) {
                    DT bound = 1.0;
                    if(w(i) < tolerance) {
                        bound = lambda(i) / (lambda(i) - w(i));
                    }
                    if( bound > tolerance && bound < beta) {
                        beta = bound;
                    }
                }
                if(perf_log) perf_log->log_total("MISC TIME", rdtsc() - z_start);

                x_hat.axpby(beta, y, (1-beta));

                int64_t remove_start = rdtsc();
                toRemove.clear();
                for(int64_t i = 0; i < lambda.length(); i++){
                    if((1-beta) * lambda(i) + beta * w(i) < tolerance)
                        toRemove.push_back(i);
                }
                if(toRemove.size() == 0) toRemove.push_back(0);
                if(perf_log) perf_log->log_total("MISC TIME", rdtsc() - remove_start);
                
                //Remove unnecessary columns from S and fixup R so that S = QR for some Q
                S.remove_cols(toRemove);
                R_next->_m = R->_m;
                R_next->_n = R->_n;
                R->remove_cols_incremental_qr_tasks_kressner(*R_next, toRemove, T, H, 128, 32, QR_ws);
                std::swap(R, R_next);
                to_ret = !to_ret;

                //Get y in aff(S)
                w._len = R->_m;
                w.set_all(1.0);
                R->transpose(); R->trsv(CblasLower, w); R->transpose();
                R->trsv(CblasUpper, w);
                w.scale(1.0 / w.sum());
                S.mvm(1.0, w, 0.0, y);
                
                x_hat_in_conv_S = w.min() >= -tolerance;
            }
        }

        if(perf_log) perf_log->log_total("MINOR TIME", rdtsc() - minor_start);
        return to_ret;
    }

    std::vector<bool> minimize(SubmodularFunction<DT>& F, Vector<DT>& wA, bool* done, int64_t max_iter, DT eps, DT tolerance, bool print, PerfLog* perf_log)
    {
        int64_t eval_F_freq = 10;
        int64_t cycles_since_last_F_eval = eval_F_freq;
        int64_t m = F.n;
        *done = false;

        //To return
        std::vector<bool> A(m);
        DT F_best = std::numeric_limits<DT>::max();

        //Step 1: Initialize by picking a point in the polytiope.

        //Characteristic vector
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        //Workspace for X_hat and next X_hat
        Vector<DT> xh1(m); xh1.perf_log = perf_log;
        Vector<DT> xh2(m); xh2.perf_log = perf_log;
        Vector<DT>* x_hat = &xh1;
        Vector<DT>* x_hat_next = &xh2;

        //Workspace for updating X_hat
        int64_t nb = 32;
        int64_t max_removed_cols = m/2;
        Matrix<DT> Mu_ws(m, b);
        Matrix<DT> Y_ws(m, b);
        Matrix<DT> Lambda_ws(m, b);
        Matrix<DT> T(nb, m);
        Matrix<DT> H(max_removed_cols, m); //TODO: handle case where we remove more than this
        Matrix<DT> QR_ws(nb, m);

        //Initialize S and R.
        Matrix<DT> S_base(m,m+1);
        Matrix<DT> R_base1(m+1,m+1);
        Matrix<DT> R_base2(m+1,m+1);

        Matrix<DT> S = S_base.submatrix(0, 0, m, 1); S.perf_log = perf_log;

        //2 matrices for R, so we can do out of place column removal
        Matrix<DT> R1 = R_base1.submatrix(0, 0, 1, 1);  R1.perf_log = perf_log;
        Matrix<DT> R2 = R_base2.submatrix(0, 0, 1, 1);  R2.perf_log = perf_log;
        Matrix<DT>* R_base = &R_base1; Matrix<DT>* R_base_next = &R_base2;
        Matrix<DT>* R = &R1; Matrix<DT>* R_next = &R2;

        Vector<DT> first_col_s = S.subcol(0);
        F.polyhedron_greedy(1.0, wA, first_col_s, perf_log);
        (*x_hat).copy(first_col_s);
        (*R)(0,0) = first_col_s.norm2();

        
        //Step 2:
        int64_t major_cycles = 0;
        while(major_cycles++ < max_iter) {
            int64_t major_start = rdtsc();

            //Snap to zero
            DT x_hat_norm2 = x_hat->norm2();
            if(x_hat_norm2*x_hat_norm2 < tolerance) {
                (*x_hat).set_all(0.0);
            }

            // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
            // Get the rest of the vectors in P_hat by executing the Edmonds greedy algorithm
            // with a random ordering of marginal gains
            auto P_hat = S_base.submatrix(0, S.width(), S.height(), b); P_hat.perf_log = perf_log;
            DT F_curr = speculate(F, P_hat, *x_hat, -1.0, perf_log);
            auto p_hat_0 = P_hat.subcol(0);
            int64_t b_added = P_hat.width();

            if (F_curr < F_best) {
                F_best = F_curr;
                for(int64_t i = 0; i < x_hat->length(); i++)
                    A[i] = (*x_hat)(i) <= 0.0;
            }

            // Update R to account for modifying S.
            // Let [R0 r1 0]' be the vectors to add to R
            //
            // Partition S_hat -> (S P_hat), R_hat-> (R R0
            //                                        0 r1T
            //                                        0  0 )
            // For each candidate vector, We want S'S = R'R
            // Then R0 = R' \ (S' * P_hat)
            //  and R1'R1 = P_hat' P_hat - R0'R0
            
            //Determine R0
            auto R0 = R_base->submatrix(0, R->width(), R->height(), b_added); R0.perf_log = perf_log;
            auto r1 = R_base->subrow(R->height(), R->width(), b_added); r1.perf_log = perf_log;
            auto ST = S.transposed();
            auto RT = R->transposed();
            R0.mmm(1.0, ST, P_hat, 0.0);
            RT.trsm(CblasLower, CblasLeft, R0);

            //Determine r1T. Each rho1i 
            //
            // rho1^2 = p_hat' * p_hat - r0' * r0;
            // Can this be done with a SYRK?
            for(int64_t j = 0; j < b_added; j++) {
                auto p_j = P_hat.subcol(j);
                DT p_j_norm2 = p_j.norm2();
                auto r0_j = R0.subcol(j);
                DT r0_j_norm2 = r0_j.norm2();
                DT rho = sqrt(std::abs(p_j_norm2*p_j_norm2 - r0_j_norm2*r0_j_norm2));
                r1(j) = rho;
            }
            
            S.enlarge_n(b_added);
            R->enlarge_n(b_added);
            R->enlarge_m(1);

            //Doublecheck that STS y = RTR y
            DT error = check_STS_eq_RTR(S, *R);

            // Get suboptimality bound
            int64_t misc_start = rdtsc();
            DT sum_x_hat_lt_0 = 0.0;
            for (int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) < tolerance)
                    sum_x_hat_lt_0 += (*x_hat)(i);
            }
//            DT subopt = F_best - sum_x_hat_lt_0;
            if(perf_log) { perf_log->log_total("MISC TIME", rdtsc() - misc_start); }

//            if(print) { std::cout << "Suboptimality bound: " << F_best-subopt << " <= min_A F(A) <= F(A_best) = " << F_best << "; delta <= " << subopt << std::endl; }

            DT xt_p = (*x_hat).dot(p_hat_0);
            DT xnrm2 = (*x_hat).norm2();
            DT xt_x = xnrm2 * xnrm2;
            if(print)
                std::cout << "x'p " << xt_p << " x'x " << xt_x << std::endl;
            if ((xt_x - xt_p < tolerance) || std::abs(F_best - sum_x_hat_lt_0) < eps ) {
                // We are done: x_hat is already closest norm point
//                if (std::abs(xt_p - xt_x) < tolerance) {
//                    subopt = 0.0;
//                }

                break;
            } else if (std::abs(xt_p + xt_x) < tolerance) {
                //We had to go through 0 to get to p_hat from x_hat.
                x_hat_next->set_all(0.0);
            } else {
                bool switch_R = min_norm_point_update_xhat(*x_hat, *x_hat_next, Y_ws, b_added,
                        Mu_ws, Lambda_ws,
                        S, R, R_next, tolerance,
                        T, H, nb, QR_ws,
                        perf_log);

                if(switch_R) {
                    std::swap(R, R_next);
                    std::swap(R_base, R_base_next);
                }
            }
            if(x_hat_next->has_nan()) {std::cout << "X has a nan. exiting. "; exit(1); }

//            x_hat->axpy(-1.0, *x_hat_next);
//            if(x_hat->norm2() < eps) {
//                std::cout << "x_hat isn't changing" << std::endl;
//            }
            std::swap(x_hat, x_hat_next);

            if(perf_log) perf_log->log_total("MAJOR TIME", rdtsc() - major_start);
        }
//        if(major_cycles > max_iter) {
//            std::cout << "Timed out." << std::endl;
//        }
        if(major_cycles < max_iter) {
            *done = true;
        }

        return A;
    }
};


#endif
