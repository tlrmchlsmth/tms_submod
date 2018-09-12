#include <unordered_set>

#include "matrix.h"
#include "vector.h"
#include "submodular.h"
#include "perf_log.h"
#include "perf/perf.h"


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

    //TODO: can be trmv instead
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

    //TODO: can be trmv instead
    R.mvm(1.0, y, 0.0, Ry); 
    RT.mvm(1.0, Ry, 0.0, RTRy); 

    RTRy.axpy(-1.0, Sy);
    return RTRy.norm2();
}

template<class DT>
class Minimizer
{
public:
    virtual std::unordered_set<int64_t> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance, bool print, PerfLog* log) = 0;
    std::unordered_set<int64_t> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance, bool print) 
    {
        return this->minimize(F, eps, tolerance, print, NULL);
    }
    std::unordered_set<int64_t> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance) 
    {
        return this->minimize(F, eps, tolerance, false, NULL);
    }
};

template<class DT>
class MinNormPoint : Minimizer<DT>
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
            PerfLog* log)
    {
        Matrix<DT>* R = R_in;
        Matrix<DT>* R_next = R_next_in;
        bool to_ret = false;

        bool keep_going = true;
        while(keep_going) {
            int64_t minor_start = rdtsc();

            auto mu = mu_ws.subvector(0, R->width()); //mu.log = log; Don't log mu so we don't overcount
            auto lambda = lambda_ws.subvector(0, R->width()); //lambda.log = log;

            //Find minimum norm point in affine hull spanned by S
            int64_t solve_start = rdtsc();
            mu.set_all(1.0);
            R->transpose(); R->trsv(CblasLower, mu); R->transpose();
            R->trsv(CblasUpper, mu);
            mu.scale(1.0 / mu.sum());
            S.mvm(1.0, mu, 0.0, y);
            if(log) log->log("SOLVE TIME", rdtsc() - solve_start);

            //Check to see if y is written as positive convex combination of S
            if(mu.min() >= -tolerance) {
                keep_going = false;
            } else {
                // Step 4:
                // It's not a convex combination
                // Project y back into polytope and remove some vectors from S
                
                // Get representation of xhat in terms of S; enforce that we get
                // affine combination (i.e., sum(lambda)==1)
                int64_t solve_start = rdtsc();
                S.transpose(); S.mvm(1.0, x_hat, 0.0, lambda); S.transpose();
                R->transpose(); R->trsv(CblasLower, lambda); R->transpose();
                R->trsv(CblasUpper, lambda);
                lambda.scale(1.0 / lambda.sum());
                if(log) log->log("SOLVE TIME", rdtsc() - solve_start);

                int64_t z_start = rdtsc();
                // Find z in conv(S) that is closest to y
                DT beta = std::numeric_limits<DT>::max();
                for(int64_t i = 0; i < lambda.length(); i++) {
                    DT bound = lambda(i) / (lambda(i) - mu(i)); 
                    if( bound > tolerance && bound < beta) {
                        beta = bound;
                    }
                }
                if(log) {
                    log->log("MISC TIME", rdtsc() - z_start);
                }
                x_hat.axpby(beta, y, (1-beta));

                int64_t remove_start = rdtsc();
                std::list<int64_t> toRemove; //TODO: pre initialize
                for(int64_t i = 0; i < lambda.length(); i++){
                    if((1-beta) * lambda(i) + beta * mu(i) < tolerance)
                        toRemove.push_back(i);
                }
                if(toRemove.size() == 0) toRemove.push_back(0);
                if(log) log->log("MISC TIME", rdtsc() - remove_start);
                
                //Remove unnecessary columns from S and fixup R so that S = QR for some Q
                S.remove_cols(toRemove);
                R_next->_m = R->_m;
                R_next->_n = R->_n;
                R->remove_cols_incremental_qr_tasks_kressner(*R_next, toRemove, T, H, 128, 32, QR_ws);
                std::swap(R, R_next);
                to_ret = !to_ret;
            }

            int64_t minor_end = rdtsc();
            if(log) log->log("MINOR TIME", minor_end - minor_start);
        }
        return to_ret;
    }
    
    std::unordered_set<int64_t> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance, bool print_in, PerfLog* log)
    {
        Vector<DT> wA(F.n);
        bool done = false;
        bool print = print_in;
        std::unordered_set<int64_t> A;
        wA.fill_rand();
        A = minimize(F, wA, &done, 500000, eps, tolerance, print, log);
        return A;
    }

    std::unordered_set<int64_t> minimize(SubmodularFunction<DT>& F, Vector<DT>& wA, bool* done, int64_t max_iter, DT eps, DT tolerance, bool print, PerfLog* log) 
    {
        std::unordered_set<int64_t> V = F.get_set();

        int64_t eval_F_freq = 10;
        int64_t cycles_since_last_F_eval = eval_F_freq;
        int64_t m = V.size();

        //Characteristic vector
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        //Workspace for x_hat and next x_hat
        Vector<DT> xh1(m);
        Vector<DT> xh2(m);
        Vector<DT>* x_hat = &xh1;
        Vector<DT>* x_hat_next = &xh2;
        xh1.log = log;
        xh2.log = log;

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

        Matrix<DT> S = S_base.submatrix(0, 0, m, 1); S.log = log;

        //2 matrices for R, so we can do out of place column removal
        Matrix<DT> R1 = R_base1.submatrix(0, 0, 1, 1);  R1.log = log;
        Matrix<DT> R2 = R_base2.submatrix(0, 0, 1, 1);  R2.log = log;
        Matrix<DT>* R_base = &R_base1; Matrix<DT>* R_base_next = &R_base2;
        Matrix<DT>* R = &R1; Matrix<DT>* R_next = &R2;

        Vector<DT> first_col_s = S.subcol(0);
        F.polyhedron_greedy(1.0, wA, first_col_s, log);
        (*x_hat).copy(first_col_s);
        (*R)(0,0) = first_col_s.norm2();

        DT F_best = std::numeric_limits<DT>::max();
        
        //Step 2:
        int64_t major_cycles = 0;
        while(major_cycles++ < max_iter) {
            int64_t major_start = rdtsc();

            //Snap to zero
            DT x_hat_norm2 = x_hat->norm2();
            if(x_hat_norm2*x_hat_norm2 < 0)
                (*x_hat).set_all(0.0);

            // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
            Vector<DT> p_hat = S_base.subcol(S.width()); p_hat.log = log;
            double F_curr = F.polyhedron_greedy(-1.0, *x_hat, p_hat, tolerance, log);
            if (F_curr < F_best) 
                F_best = F_curr;
            
            // Update R to account for modifying S.
            // Let [r0 rho1]^T be the vector to add to r
            // r0 = R' \ (S' * p_hat)
            int64_t add_col_start = rdtsc();
            Vector<DT> r0 = R_base->subcol(0, R->width(), R->height()); r0.log = log;
            S.transpose(); S.mvm(1.0, p_hat, 0.0, r0); S.transpose();
            R->transpose(); R->trsv(CblasLower, r0); R->transpose();

            // rho1^2 = p_hat' * p_hat - r0' * r0;
            DT phat_norm2 = p_hat.norm2();
            DT r0_norm2 = r0.norm2();
            DT rho1 = sqrt(std::abs(phat_norm2*phat_norm2 - r0_norm2*r0_norm2));

            R->enlarge_m(1); R->enlarge_n(1);
            (*R)(R->width()-1, R->height()-1) = rho1;
            S.enlarge_n(1);
            if(log) { log->log("ADD COL TIME", rdtsc() - add_col_start); }

            /*
            //Slow version of checking current function value
            int64_t eval_start = rdtsc();
            for(int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) < tolerance) A_curr.insert(i);
            }

            auto F_curr = F.eval(A_curr);
            if(log) log->log("EVAL F TIME", rdtsc() - eval_start);
            */

            int64_t misc_start = rdtsc();
            // Get suboptimality bound
            DT sum_x_hat_lt_0 = 0.0;
            for (int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) < tolerance)
                    sum_x_hat_lt_0 += (*x_hat)(i);
            }
            DT subopt = F_best - sum_x_hat_lt_0;
            if(log) { log->log("MISC TIME", rdtsc() - misc_start); }

            if(print) { std::cout << "Suboptimality bound: " << F_best-subopt << " <= min_A F(A) <= F(A_best) = " << F_best << "; delta <= " << subopt << std::endl; }

            DT xt_p = (*x_hat).dot(p_hat);
            DT xnrm2 = (*x_hat).norm2();
            DT xt_x = xnrm2 * xnrm2;
            if(print)
                std::cout << "x'p " << xt_p << " x'x " << xt_x << std::endl;
            if ((std::abs(xt_p - xt_x) < tolerance) || (subopt<eps)) {
                // We are done: x_hat is already closest norm point
                if (std::abs(xt_p - xt_x) < tolerance) {
                    subopt = 0.0;
                }

                break;
            } else if (std::abs(xt_p + xt_x) < tolerance) {
                //We had to go through 0 to get to p_hat from x_hat.
                x_hat_next->set_all(0.0);
            } else {
                bool switch_R = min_norm_point_update_xhat(*x_hat, *x_hat_next,
                        mu_ws, lambda_ws,
                        S, R, R_next, tolerance,
                        T, H, nb, QR_ws,
                        log);
                if(switch_R) {
                    std::swap(R, R_next);
                    std::swap(R_base, R_base_next);
                }
            }
            if(x_hat_next->has_nan()) exit(1);

            //x_hat->axpy(-1.0, *x_hat_next);
//            if(x_hat->norm2() < eps) {
//                std::cout << "x_hat isn't changing" << std::endl;
//            }
            std::swap(x_hat, x_hat_next);

            if(log) {
                log->log("MAJOR TIME", rdtsc() - major_start);
            }
        }

        //Return
        std::unordered_set<int64_t> A_best;
        for(int64_t i = 0; i < x_hat->length(); i++) {
            if((*x_hat)(i) < tolerance) A_best.insert(i);
        }
        if(print) {
            std::cout << "Done. |A| = " << A_best.size() << " F_best = " << F.eval(A_best) << std::endl;
        }
        if(major_cycles < max_iter) {
            *done = true;
        }
        return A_best;
    }
};

template<class DT>
class BRSMinNormPoint : Minimizer<DT>
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

    void speculate(SubmodularFunction<DT>& F, Matrix<DT>& P_hat, Vector<DT>& x_hat, DT alpha, PerfLog* log)
    {
        //Determine first col of P_hat with Edmonds greedy
        auto p_hat_0 = P_hat.subcol(0);
        F.polyhedron_greedy(alpha, x_hat, p_hat_0, log);

        //Add more columns to P. We want each of them to be extreme points
        //x in B(f) is an extreme point in B(f) iff for some maximal chain 0 = S0 subset of S1 subset of ... subset of Sn = V
        //x(Si - Si-1) = f(Si) - f(Si-1)
        //
        int64_t start_spec = rdtsc();
        for(int64_t j = 1; j < P_hat.width(); j++) {
            auto p_hat_j = P_hat.subcol(j);
            scramble(F.permutation);
            F.marginal_gains(F.permutation, p_hat_j);
        }
        if(log) log->log("SPECULATION TIME", rdtsc() - start_spec);
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
            PerfLog* log)
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

        auto Y  = Y_ws.submatrix(0, 0, S.height(), b);  Y.log = log;
        auto S0 =    S.submatrix(0, 0, S.height(), n); S0.log = log;
        Y.mmm(1.0, S0, W0, 0.0);
        for(int64_t j = 0; j < b; j++) {
            auto y_j = Y.subcol(j);
            auto s_j = S.subcol(n+j);
            y_j.axpy(w1(j), s_j);
        }

        //Take the best column of Y that is written as a positive convex combination of S
        int64_t best = -1;
//        DT best_val = -1.0;
        DT best_val = std::numeric_limits<DT>::max();
        for(int64_t j = 0; j < b; j++) {
            auto w = W.subcol(j);
            if((j==0 && w.min() >= -tolerance) || w.min() > 1e-10) {
                auto yj = Y.subcol(j);
                DT y_norm = yj.norm2();
                if(y_norm < best_val) {
//                if(w.min() > best_min) {
                    best = j;
                    best_val = y_norm;
                }
            }
        }
        if(best > -1) {
            //We're going to take it, and throw away the rest.
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
            DT error = check_STS_eq_RTR(S, *R); 
            if(error > 1e-5) {
                std::cout << "In xhat update: ||STS q - RTR q|| " << error << std::endl;
                exit(1);
            }

            y.copy(y_best);
/*            std::cout << "Xhat update A" << std::endl;
            std::cout << "xt_x " << x_hat.dot(x_hat) << std::endl;
            std::cout << "yt_y " << y.dot(y) << std::endl;*/
//            assert(y.dot(y) < x_hat.dot(x_hat));
//            std::cout << std::endl;
        } else {
            // 
            //Step 4. 
            //x_hat isn't in the convex hull of S plus any of the candidates 
            //In this case, we choose the 0th candidate.
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
                auto w = W.subcol(0, 0, R->height()); w.log = log;
                auto lambda = Lambda_ws.subcol(0, 0, R->height()); lambda.log = log;
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
//                std::cout << "STS RTR checksum is " << error << std::endl;
                //assert(error < 1e-10);

                int64_t z_start = rdtsc();
                // Find z in conv(S) that is closest to y
                DT beta = 1.0;
                for(int64_t i = 0; i < lambda.length(); i++) {
                    DT bound = lambda(i) / (lambda(i) - w(i)); 
                    if( bound > tolerance && bound < beta) {
                        beta = bound;
                    }
                }
                if(log) {
                    log->log("MISC TIME", rdtsc() - z_start);
                }
/*                std::cout << x_hat.dot(x_hat) << std::endl;
                std::cout << "beta is " << beta << std::endl;*/
                x_hat.axpby(beta, y, (1-beta));
//                std::cout << x_hat.dot(x_hat) << std::endl;

                int64_t remove_start = rdtsc();
                toRemove.clear();
                for(int64_t i = 0; i < lambda.length(); i++){
                    if((1-beta) * lambda(i) + beta * w(i) < tolerance)
                        toRemove.push_back(i);
                }
                if(toRemove.size() == 0) toRemove.push_back(0);
                if(log) {
                    log->log("MISC TIME", rdtsc() - remove_start);
                }
                
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

/*            std::cout << "Xhat update B" << std::endl;
            std::cout << "xt_x " << x_hat.dot(x_hat) << std::endl;
            std::cout << "yt_y " << y.dot(y) << std::endl;
            assert(y.dot(y) < x_hat.dot(x_hat));
            std::cout << std::endl;*/
        }

        if(log) log->log("MINOR TIME", rdtsc() - minor_start);
        return to_ret;
    }

    std::unordered_set<int64_t> minimize(SubmodularFunction<DT>& F, DT eps, DT tolerance, bool print, PerfLog* log) 
    {
        std::unordered_set<int64_t> V = F.get_set();

        int64_t eval_F_freq = 10;
        int64_t cycles_since_last_F_eval = eval_F_freq;
        int64_t m = V.size();

        //Step 1: Initialize by picking a point in the polytiope.
        std::unordered_set<int64_t> A;
        A.reserve(m);

        //Characteristic vector
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        Vector<DT> wA(m);
        wA.set_all(0.0);
        /*for(int64_t i = 0; i < m; i++) {
            if(dist(gen) > .5) {
                wA(i) = 1.0;
                A.insert(i);
            } else {
                wA(i) = 0.0;
            }
        }*/

        //Workspace for X_hat and next X_hat
        Vector<DT> xh1(m); xh1.log = log;
        Vector<DT> xh2(m); xh2.log = log;
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

        Matrix<DT> S = S_base.submatrix(0, 0, m, 1); S.log = log;

        //2 matrices for R, so we can do out of place column removal
        Matrix<DT> R1 = R_base1.submatrix(0, 0, 1, 1);  R1.log = log;
        Matrix<DT> R2 = R_base2.submatrix(0, 0, 1, 1);  R2.log = log;
        Matrix<DT>* R_base = &R_base1; Matrix<DT>* R_base_next = &R_base2;
        Matrix<DT>* R = &R1; Matrix<DT>* R_next = &R2;

        Vector<DT> first_col_s = S.subcol(0);
        F.polyhedron_greedy(1.0, wA, first_col_s, log);
        (*x_hat).copy(first_col_s);
        (*R)(0,0) = first_col_s.norm2();

        //Initialize A_best and F_best
        std::unordered_set<int64_t> A_best;
        A_best.reserve(m);
        DT F_best = std::numeric_limits<DT>::max();
        std::unordered_set<int64_t> A_curr;
        A_curr.reserve(m);
        
        //Step 2:
        int64_t max_iter = 1e6;
        int64_t major_cycles = 0;
        while(major_cycles++ < max_iter) {
            int64_t major_start = rdtsc();

            //Snap to zero
            //TODO: Krause's code uses xhat->norm2() < 0,
            DT x_hat_norm2 = x_hat->norm2();
            if(x_hat_norm2*x_hat_norm2 < tolerance) {
                (*x_hat).set_all(0.0);
            }

            // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
            auto P_hat = S_base.submatrix(0, S.width(), S.height(), b); P_hat.log = log;
            speculate(F, P_hat, *x_hat, -1.0, log);
            auto p_hat_0 = P_hat.subcol(0);
            int64_t b_added = P_hat.width();


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
            auto R0 = R_base->submatrix(0, R->width(), R->height(), b_added); R0.log = log;
            auto r1 = R_base->subrow(R->height(), R->width(), b_added); r1.log = log;
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
            //
            //
/*
            //Old math that tried to enforce R'R = S'S
            auto P_hatT = P_hat.transposed();
            auto R0T = R0.transposed();
            R1.syrk(CblasUpper,  1.0, P_hatT, 0.0);
            R1.syrk(CblasUpper, -1.0, R0T, 1.0);
            R1.chol('U');
*/
            S.enlarge_n(b_added);
            R->enlarge_n(b_added);
            R->enlarge_m(1);

            //Doublecheck that STS y = RTR y
            DT error = check_STS_eq_RTR(S, *R);

//            std::cout << "||STS y - RTR y|| " << error << std::endl;
/*            if(error > 1e-5) {
                //Sometimes this will happen because I choose columns of P randomly.
                std::cout << "R" << std::endl;
                R->print();
                std::cout << std::endl;
                exit(1);
            }*/
           
            // Check current function value
            A_curr.clear();
            bool compute_f_cur = false;
            for(int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) < tolerance) {
                    A_curr.insert(i);
                    if(A_best.count(i) == 0) 
                        compute_f_cur = true;
                }
            }
            compute_f_cur = compute_f_cur || A_curr.size() != A_best.size();
            
            if(compute_f_cur) {
                int64_t eval_start = rdtsc();
                auto F_curr = F.eval(A_curr);
                if(log) { log->log("EVAL F TIME", rdtsc() - eval_start); }

                int64_t misc_start = rdtsc();
                if (F_curr < F_best) {
                    // Save best F and get the unique minimal minimizer
                    F_best = F_curr;
                    A_best = A_curr;
                }
                if(log) { log->log("MISC TIME", rdtsc() - misc_start); }
            }

            // Get suboptimality bound
            int64_t misc_start = rdtsc();
            DT sum_x_hat_lt_0 = 0.0;
            for (int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) < tolerance)
                    sum_x_hat_lt_0 += (*x_hat)(i);
            }
            DT subopt = F_best - sum_x_hat_lt_0;
            if(log) { log->log("MISC TIME", rdtsc() - misc_start); }

            if(print) { std::cout << "Suboptimality bound: " << F_best-subopt << " <= min_A F(A) <= F(A_best) = " << F_best << "; delta <= " << subopt << std::endl; }

            DT xt_p = (*x_hat).dot(p_hat_0);
            DT xnrm2 = (*x_hat).norm2();
            DT xt_x = xnrm2 * xnrm2;
            if(print)
                std::cout << "x'p " << xt_p << " x'x " << xt_x << std::endl;
            if ((std::abs(xt_p - xt_x) < tolerance) || (subopt<eps)) {
/*                std::cout << "Stopping because ";
                if(subopt < eps) std::cout << " subopt " << subopt << " is less than eps " << std::endl;
                else {
                    std::cout << " xt_p - xt_x " << xt_p - xt_x << " is less than tolerance " << std::endl;
                    std::cout << "||x|| " << x_hat->norm2() << " ||p|| " << p_hat_0.norm2() << std::endl;
                    std::cout << "xt_p " << x_hat->dot(p_hat_0) << " xt_x " << x_hat->dot(*x_hat) << std::endl;

                }*/
                // We are done: x_hat is already closest norm point
                if (std::abs(xt_p - xt_x) < tolerance) {
                    subopt = 0.0;
                }

                break;
            } else if (std::abs(xt_p + xt_x) < tolerance) {
                //We had to go through 0 to get to p_hat from x_hat.
                x_hat_next->set_all(0.0);
            } else {
/*                bool switch_R = min_norm_point_update_xhat(*x_hat, *x_hat_next,
                        mu_ws, lambda_ws,
                        S, R, R_next, tolerance,
                        T, H, nb, QR_ws,
                        log);*/
                bool switch_R = min_norm_point_update_xhat(*x_hat, *x_hat_next, Y_ws, b_added,
                        Mu_ws, Lambda_ws,
                        S, R, R_next, tolerance,
                        T, H, nb, QR_ws,
                        log);

                if(switch_R) {
                    std::swap(R, R_next);
                    std::swap(R_base, R_base_next);
                }
            }
            if(x_hat_next->has_nan()) {std::cout << "X has a nan. exiting. "; exit(1); }

            x_hat->axpy(-1.0, *x_hat_next);
            if(x_hat->norm2() < eps) {
                std::cout << "x_hat isn't changing" << std::endl;
            }
            std::swap(x_hat, x_hat_next);

            if(log) log->log("MAJOR TIME", rdtsc() - major_start);
        }
        if(print) {
            std::cout << "Done. |A| = " << A_curr.size() << " F_best = " << F.eval(A_curr) << std::endl;
        }
//        if(major_cycles > max_iter) {
//            std::cout << "Timed out." << std::endl;
//        }

//        return A_best;
        return A_curr;
    }
};
