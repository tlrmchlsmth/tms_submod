#ifndef TMS_SUBMOD_MNP_H
#define TMS_SUBMOD_MNP_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../submodular.h"
#include "../perf_log.h"

//R is upper triangular
template<class DT>
DT check_STS_eq_RTR(const Matrix<DT>& S, const IncQRMatrix<DT>& R_in)
{
    assert(R_in._n > 0);
    auto R = R_in.current_matrix();

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


//At the end, y is equal to the new value of x_hat
//mu is a tmp vector with a length = m
//Returns: whether R and R_new should be swapped
template<class DT>
void mnp_update_xhat(Vector<DT>& x_hat, Vector<DT>& y,
        Vector<DT>& mu_ws, Vector<DT>& lambda_ws,
        Matrix<DT>& S, IncQRMatrix<DT>& R, DT tolerance)
{
    bool keep_going = true;
    while(keep_going) {
        auto mu = mu_ws.subvector(0, R.width());
        auto lambda = lambda_ws.subvector(0, R.width());

        //Find minimum norm point in affine hull spanned by S
        mu.set_all(1.0);
        R.transpose(); R.trsv(mu); R.transpose();

        R.trsv(mu);
        mu.scale(1.0 / mu.sum());
        S.mvm(1.0, mu, 0.0, y);

        //Check to see if y is written as positive convex combination of S
        if(mu.min() > -tolerance) {
            keep_going = false;
        } else {
            // Step 4:
            // It's not a convex combination
            // Project y back into polytope and remove some vectors from S
            
            // Get representation of xhat in terms of S; enforce that we get
            // affine combination (i.e., sum(lambda)==1)
            S.transpose(); S.mvm(1.0, x_hat, 0.0, lambda); S.transpose();
            R.transpose(); R.trsv(lambda); R.transpose();
            R.trsv(lambda);
            lambda.scale(1.0 / lambda.sum());

            //Note: it is imperitive to not let z drift out of the convex hull of S.
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

            x_hat.axpby(beta, y, (1-beta));

            int64_t remove_start = rdtsc();
            std::list<int64_t> toRemove;
            for(int64_t i = 0; i < lambda.length(); i++){
                if((1-beta) * lambda(i) + beta * mu(i) <= tolerance){
                    toRemove.push_back(i);
                }
            }

            if(toRemove.size() == 0) {
                std::cout << "Warning: no columns to remove!" << std::endl;
                toRemove.push_back(0);
            }
            
            //Remove unnecessary columns from S and fixup R so that S = QR for some Q
            S.remove_cols(toRemove);
            R.remove_cols_inc_qr(toRemove);
        }
    }

    x_hat.copy(y);
}

template<class DT>
std::vector<bool> mnp(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
{
    int64_t eval_F_freq = 10;
    int64_t cycles_since_last_F_eval = eval_F_freq;
    int64_t m = F.n;

    //To return
    std::vector<bool> A(m);
    DT F_best = std::numeric_limits<DT>::max();

    //Workspace for x_hat and next x_hat
    Vector<DT> x_hat(m);
    Vector<DT> y(m);

    //Workspace for updating x_hat
    int64_t nb = 32;
    int64_t max_removed_cols = m/2;
    Vector<DT> mu_ws(m);
    Vector<DT> lambda_ws(m);

    //Initialize S and R.
    Matrix<DT> S_base(m,m+1);
    IncQRMatrix<DT> R_base(m+1);
    auto S = S_base.submatrix(0, 0, m, 1);
    auto R = R_base.submatrix(0, 1);

    Vector<DT> first_col_s = S.subcol(0);
    F.polyhedron_greedy_decending(wA, first_col_s);
    x_hat.copy(first_col_s);
    R(0,0) = first_col_s.norm2();

    DT pj_max = 0.0;

    //Step 2:
    int64_t major_cycles = 0;
    while(1) {
        assert(S.width() <= F.n);

        //Snap to zero
        DT x_hat_norm2 = x_hat.norm2();
        if(x_hat_norm2*x_hat_norm2 < tolerance)
            x_hat.set_all(0.0);

        // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
        Vector<DT> p_hat = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);
        pj_max = std::max(p_hat.dot(p_hat), pj_max);

        if (F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A[i] = x_hat(i) <= 0.0;
        }
        
        // Update R to account for modifying S.
        R.add_col_inc_qr(S, p_hat);
        S.enlarge_n(1);

        // Get suboptimality bound
        DT sum_x_hat_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) {
            if(x_hat(i) <= 0.0)
                sum_x_hat_lt_0 += x_hat(i);
        }

        DT xt_p = x_hat.dot(p_hat);
        DT xnrm = x_hat.norm2();
        DT xt_x = xnrm * xnrm;
//            if(print || major_cycles % 100 == 0) std::cout << "x'p " << xt_p << " x'x " << xt_x << std::endl;
        if( xt_p > xt_x - tolerance * pj_max || std::abs(F_best - sum_x_hat_lt_0) < eps) {
            // We are done: x_hat is already closest norm point
//                if (std::abs(xt_p - xt_x) < tolerance) subopt = 0.0;
            break;
        } else if (std::abs(xt_p + xt_x) < tolerance) {
            //We had to go through 0 to get to p_hat from x_hat.
            std::cout << "Setting x hat to zero" << std::endl;
            x_hat.set_all(0.0);
        } else {
            mnp_update_xhat(x_hat, y, mu_ws, lambda_ws, S, R, tolerance);
        }
       
        major_cycles++;
    }

    PerfLog::get().log_total("MAJOR CYCLES", major_cycles);
    return A;
}

template<class DT>
std::vector<bool> mnp(SubmodularFunction<DT>& F, DT eps, DT tolerance) {
    Vector<DT> wA(F.n);
    wA.fill_rand();
    return mnp(F, wA, eps, tolerance);
}

#endif
