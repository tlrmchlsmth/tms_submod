#include <unordered_set>

#include "matrix.h"
#include "vector.h"
#include "submodular.h"
#include "perf_log.h"
#include "perf/perf.h"

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

        while(true) {
            int64_t minor_start = rdtsc();

            auto mu = mu_ws.subvector(0, R->width()); mu.log = log;
            auto lambda = lambda_ws.subvector(0, R->width()); lambda.log = log;

            //Find minimum norm point in affine hull spanned by S
            mu.set_all(1.0);
            R->transpose(); R->trsv(CblasLower, mu); R->transpose();
            R->trsv(CblasUpper, mu);
            mu.scale(1.0 / mu.sum());
            S.mvm(1.0, mu, 0.0, y);

            //Check to see if y is written as positive convex combination of S
            if(mu.min() >= -tolerance)
                break;
            
            // Step 4:
            // It's not a convex combination
            // Project y back into polytope and remove some vectors from S
            
            // Get representation of xhat in terms of S; enforce that we get
            // affine combination (i.e., sum(lambda)==1)
            S.transpose(); S.mvm(1.0, x_hat, 0.0, lambda); S.transpose();
            R->transpose(); R->trsv(CblasLower, lambda); R->transpose();
            R->trsv(CblasUpper, lambda);
            lambda.scale(1.0 / lambda.sum());

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

            int64_t minor_end = rdtsc();
            if(log) {
                log->log("MINOR TIME", minor_end - minor_start);
            }
        }
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
        for(int64_t i = 0; i < m; i++) {
            if(dist(gen) > .5) {
                wA(i) = 1.0;
                A.insert(i);
            } else {
                wA(i) = 0.0;
            }
        }

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

        //Initialize A_best and F_best
        std::unordered_set<int64_t> A_best;
        A_best.reserve(m);
        DT F_best = std::numeric_limits<DT>::max();
        std::unordered_set<int64_t> A_curr;
        A_curr.reserve(m);
        
        //Step 2:
        //int64_t max_iter = 10000;
        int64_t major_cycles = 0;
//        while(major_cycles++ < max_iter) {
        while(true) {
            major_cycles++;
            int64_t major_start = rdtsc();

            //Snap to zero
            //TODO: Krause's code uses xhat->norm2() < 0,
            DT x_hat_norm2 = x_hat->norm2();
            if(x_hat_norm2*x_hat_norm2 < tolerance) {
                (*x_hat).set_all(0.0);
            }

            // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
            Vector<DT> p_hat = S_base.subcol(S.width()); p_hat.log = log;
            int64_t greedy_start = rdtsc();
            F.polyhedron_greedy(-1.0, *x_hat, p_hat, log);
            if(log) { log->log("GREEDY TIME", rdtsc() - greedy_start); }

            // Update R to account for modifying S.
            // Let [r0 rho1]^T be the vector to add to r
            // r0 = R' \ (S' * p_hat)
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

            x_hat->axpy(-1.0, *x_hat_next);
            if(x_hat->norm2() < eps) {
                std::cout << "x_hat isn't changing" << std::endl;
            }
            std::swap(x_hat, x_hat_next);

            if(log) {
                log->log("MAJOR TIME", rdtsc() - major_start);
            }
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
