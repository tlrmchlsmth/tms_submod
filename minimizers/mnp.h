#ifndef TMS_SUBMOD_MNP_H
#define TMS_SUBMOD_MNP_H

#include "../la/matrix.h"
#include "../la/vector.h"
#include "../la/inc_qr_matrix.h"
#include "../set_fn/submodular.h"
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

//At the end, y is equal to the new value of x_hat
//mu is a tmp vector with a length = m
//Returns: whether R and R_new should be swapped
template<class DT>
void mnp_update_xhat(Vector<DT>& x_hat, Vector<DT>& y,
        Matrix<DT>& S, IncQRMatrix<DT>& R, DT tolerance,
        Vector<DT>& mu_ws, Vector<DT>& lambda_ws)
{
    while(1) {
        auto mu = mu_ws.subvector(0, S.width());
        auto lambda = lambda_ws.subvector(0, S.width());

        //
        //Find minimum norm point in affine hull spanned by S
        //
        mu.set_all(1.0);
        R.transpose(); R.trsv(mu); R.transpose();
        R.trsv(mu);
        mu.scale(1.0 / mu.sum());
        S.mvm(1.0, mu, 0.0, y);

        //Check to see if y is written as positive convex combination of S
        if(mu.min() > -tolerance) break;

        // It's not a convex combination.
        
        // 
        // Project y back into polytope
        //
        
        // Get representation of xhat in terms of S; enforce that we get affine combination (i.e., sum(lambda)==1)
        S.transpose(); S.mvm(1.0, x_hat, 0.0, lambda); S.transpose();
        R.transpose(); R.trsv(lambda); R.transpose();
        R.trsv(lambda);
        lambda.scale(1.0 / lambda.sum());

        // Find z in conv(S) that is closest to y
        // Note: This is different from Andreas's code; because there, beta could end up being outside the range [0,1]
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

        //
        // Remove some vectors from R and S
        //

        //Determine which columns of S and R are useless
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

    x_hat.copy(y);
}

template<class DT>
std::vector<bool> mnp(SubmodularFunction<DT>& F, Vector<DT>& wA, DT eps, DT tolerance) 
{
    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A(F.n);

    //Workspace for x_hat and next x_hat
    Vector<DT> x_hat(F.n);
    Vector<DT> y(F.n);

    //Workspace for updating x_hat
    Vector<DT> mu_ws(F.n);
    Vector<DT> lambda_ws(F.n);

    //Initialize S and R.
    Matrix<DT> S_base(F.n,F.n+1);
    IncQRMatrix<DT> R_base(F.n+1);
    auto S = S_base.submatrix(0, 0, F.n, 1);
    auto R = R_base.submatrix(0, 1);

    Vector<DT> s0 = S.subcol(0);
    F.polyhedron_greedy_decending(wA, s0);
    x_hat.copy(s0);
    R(0,0) = s0.norm2();
    DT pt_p_max = s0.dot(s0);

    int64_t major_cycles = 0;
    while(1) {
        assert(S.width() <= F.n);

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
        DT F_curr = F.polyhedron_greedy_ascending(x_hat, p_hat);

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
            sum_x_hat_lt_0 += std::min(x_hat(i), 0.0);
        }

        //Test to see if we are done
        DT xt_p = x_hat.dot(p_hat);
        pt_p_max = std::max(p_hat.dot(p_hat), pt_p_max);
        if( xt_p > xt_x - tolerance * pt_p_max || std::abs(F_best - sum_x_hat_lt_0) < eps) break;

        // Update x_hat
        mnp_update_xhat(x_hat, y, S, R, tolerance, mu_ws, lambda_ws);
       
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
