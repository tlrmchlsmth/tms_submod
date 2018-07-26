#include "matrix.h"
#include "vector.h"

template<class DT>
class Minimizer
{
public:
    virtual void minimize(FV2toR& F, DT eps, DT tolerance, bool print) = 0;
};

template<class DT>
class MinNormPoint : Minimizer<DT>
{
public:
    int64_t minor_cycles;
    int64_t major_cycles;
    MinNormPoint() : minor_cycles(0), major_cycles(0) {}

    //At the end, y is equal to the new value of x_hat
    //mu is a tmp vector with a length = m
    void min_norm_point_update_xhat(Vector<DT>& x_hat, Vector<DT>& y,
            Vector<DT>& mu_ws, Vector<DT>& lambda_ws,
            Matrix<DT>& S, Matrix<DT>& R, DT tolerance)
    {
        while(true) {
            minor_cycles++;

            auto mu = mu_ws.subvector(0, R.width());
            auto lambda = lambda_ws.subvector(0, R.width());

            //Find minimum norm point in affine hull spanned by S
            mu.set_all(1.0);
            R.transpose(); R.trsv(CblasLower, mu); R.transpose();
            R.trsv(CblasUpper, mu);
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
            R.transpose(); R.trsv(CblasLower, lambda); R.transpose();
            R.trsv(CblasUpper, lambda);
            lambda.scale(1.0 / lambda.sum());

            // Find z in conv(S) that is closest to y
            DT beta = std::numeric_limits<DT>::max();
            for(int64_t i = 0; i < lambda.length(); i++) {
                DT bound = lambda(i) / (lambda(i) - mu(i)); 
                if( bound > tolerance && bound < beta) {
                    beta = bound;
                }
            }
            x_hat.axpby(beta, y, (1-beta));

            std::list<int64_t> toRemove; //TODO: pre initialize
            for(int64_t i = 0; i < lambda.length(); i++){
                if((1-beta) * lambda(i) + beta * mu(i) < tolerance)
                    toRemove.push_back(i);
            }

            //Remove unnecessary columns from S and fixup R so that S = QR for some Q
            S.remove_cols(toRemove);
            R.remove_cols_incremental_qr(toRemove, mu);
        }
    }

    void minimize(FV2toR& F, DT eps, DT tolerance, bool print) 
    {
        std::unordered_set<int64_t> V = F.get_set();

        int64_t m = V.size();

        //Step 1: Initialize by picking a point in the polytiope.
        std::unordered_set<int64_t> A;
        A.reserve(m);

        Vector<DT> wA(m);

        //Workspace for x_hat and next x_hat
        Vector<DT> xh1(m);
        Vector<DT> xh2(m);
        Vector<DT>* x_hat = &xh1;
        Vector<DT>* x_hat_next = &xh2;

        //Workspace for updating x_hat
        Vector<DT> mu_ws(m);
        Vector<DT> lambda_ws(m);

        //Initialize S and R.
        Matrix<DT> S_base(m,m+1);
        Matrix<DT> R_base(m+1,m+1);

        Matrix<DT> S = S_base.submatrix(0, 0, m, 1);
        Matrix<DT> R = R_base.submatrix(0, 0, 1, 1);

        Vector<DT> first_col_s = S.subcol(0);
        F.polyhedron_greedy(1.0, wA, first_col_s);
        (*x_hat).copy(first_col_s);
        R(0,0) = first_col_s.norm2();

        //Initialize A_best F_best
        std::unordered_set<int64_t> A_best;
        A_best.reserve(m);
        DT F_best = std::numeric_limits<DT>::max();
        
        //Step 2:
        int n_iter = 0;
        int max_iter = 5000;
        while(n_iter++ < max_iter) {
            major_cycles++;
            //Snap to zero
            //TODO: Krause's code uses xhat->norm2() < tolerance,
            //but if I do this, sometimes my code doesn't terminate.
            if(x_hat->norm2()*x_hat->norm2() < tolerance) {
                (*x_hat).set_all(0.0);
            }


            // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
            Vector<DT> p_hat = S_base.subcol(S.width());
            F.polyhedron_greedy(-1.0, *x_hat, p_hat);


            // Update R to account for modifying S.
            // Let [r0 rho1]^T be the vector to add to r
            // r0 = R' \ (S' * p_hat)
            Vector<DT> r0 = R_base.subcol(0, R.width(), R.height());
            S.transpose(); S.mvm(1.0, p_hat, 0.0, r0); S.transpose();
            R.transpose(); R.trsv(CblasLower, r0); R.transpose();

            // rho1^2 = p_hat' * p_hat - r0' * r0;
            DT rho1 = sqrt(std::abs(p_hat.norm2()*p_hat.norm2() - r0.norm2()*r0.norm2()));
            
            R.enlarge_m(1); R.enlarge_n(1);
            R(R.width()-1, R.height()-1) = rho1;
            S.enlarge_n(1);

            // Check current function value
            // TODO: Here I check to see if each element of x_hat is less than the tolerance,
            // but Krause's code uses 0 instead.
            auto F_cur = F.eval(V, [&](int64_t i) -> bool { return ((*x_hat)(i)) < tolerance; } );
            if (F_cur < F_best) {
                // Save best F and get the unique minimal minimizer
                F_best = F_cur;
                A_best.clear();
                for (int64_t i = 0; i < (*x_hat).length(); i++) {
                    if ((*x_hat)(i) < tolerance) {
                        A_best.insert(i);
                    }
                }
            }

            // Get suboptimality bound
            // TODO: Check against tolerance vs. check against 0.0
            DT sum_x_hat_lt_0 = 0.0;
            for (int64_t i = 0; i < x_hat->length(); i++) {
                if((*x_hat)(i) < tolerance)
                    sum_x_hat_lt_0 += (*x_hat)(i);
            }
            DT subopt = F_best - sum_x_hat_lt_0;

            if(print)
                std::cout << "Suboptimality bound: " << F_best-subopt << " <= min_A F(A) <= F(A_best) = " << F_best << "; delta <= " << subopt << std::endl;

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
            } else {
                min_norm_point_update_xhat(*x_hat, *x_hat_next,
                        mu_ws, lambda_ws,
                        S, R, tolerance);
            }
            std::swap(x_hat, x_hat_next);
        }
        if(n_iter > max_iter) {
            std::cout << "Timed out." << std::endl;
        }
        if(print) {
            std::cout << "Final F_best " << F_best << " |A| " << A_best.size() << std::endl;
        }
    }
};
