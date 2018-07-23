#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include <list>
#include <set>

#include <random>
#include "vector.h"
#include "matrix.h"
#include "submodular_fn.h"
#include "perf/perf.h"
#include "test/validate/validate.h"

void polyhedron_greedy(FV2toR& F, std::vector<int64_t>& V, vector<int64_t> permutation,
        double alpha, Vector<double>& weights, Vector<double>& xout) 
{

    //sort weights descending
    if (alpha > 0.0) {
        std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) > weights(b); } );
    } else if (alpha < 0.0) {
        std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) < weights(b); } );
    }

    std::vector<int64_t> A;
    A.reserve(V.size());

//  double F_old = F.eval(A);
    double F_old = 0.0;
    for(int i = 0; i < V.size(); i++) {
        A.push_back(V[permutation[i]]);
        double F_A = F.eval(A);
        //Add in marginal gain from adding F_A to the last F
        xout(permutation[i]) = F_A - F_old;
        F_old = F_A;
    }
}


//At the end, y is equal to the new value of x_hat
//mu is a tmp vector with a length = m
void min_norm_point_update_xhat(Vector<double>& x_hat, Vector<double>& y,
        Vector<double>& mu_ws, Vector<double>& lambda_ws,
        Matrix<double>& S, Matrix<double>& R, double tolerance)
{
    while(true) {
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
        double beta = std::numeric_limits<double>::max();
        for(int64_t i = 0; i < lambda.length(); i++) {
            double bound = lambda(i) / (lambda(i) - mu(i)); 
            if( bound > tolerance && bound < beta) {
                beta = bound;
            }
        }
        x_hat.axpby(beta, y, (1-beta));

        list<int64_t> toRemove;
        for(int64_t i = 0; i < lambda.length(); i++){
            if((1-beta) * lambda(i) + beta * mu(i) < tolerance)
                toRemove.push_back(i);
        }

        //Remove unnecessary columns from S and fixup R so that S = QR for some Q
        S.remove_cols(toRemove);
        R.remove_cols_incremental_qr(toRemove, mu);
    }
}

void min_norm_point(FV2toR& F, std::vector<int64_t>& V)
{
    int64_t m = V.size();

    //Step 1: Initialize by picking a point in the polytiope.
    std::vector<int64_t> A;
    A.reserve(m);
    
    double eps = 1e-10;
    double tolerance = 1e-10;

    Vector<double> wA(m);

    //Workspace for x_hat and next x_hat
    Vector<double> xh1(m);
    Vector<double> xh2(m);
    Vector<double>* x_hat = &xh1;
    Vector<double>* x_hat_next = &xh2;

    //Workspace for updating x_hat
    Vector<double> mu_ws(m);
    Vector<double> lambda_ws(m);

    //Initialize S and R.
    Matrix<double> S_base(m,m);
    Matrix<double> R_base(m,m);

    Matrix<double> S = S_base.submatrix(0, 0, m, 1);
    Matrix<double> R = R_base.submatrix(0, 0, 1, 1);

    //Permutation for sorting weights in the greedy algorithm
    std::vector<int64_t> permutation;
    permutation.reserve(V.size());
    for(int i = 0; i < V.size(); i++){ permutation.push_back(i); }

    Vector<double> first_col_s = S.subcol(0);
    polyhedron_greedy(F, V, permutation, 1.0, wA, first_col_s);
    (*x_hat).copy(first_col_s);
    R(0,0) = first_col_s.norm2();

    //Initialize A_best F_best
    std::vector<int64_t> A_best;
    A_best.reserve(m);
    double F_best = std::numeric_limits<double>::max();

    //Step 2:
    while(true) {
        //Snap to zero
        if((*x_hat).norm2() < tolerance)
            (*x_hat).zero_out();   

        // Get p_hat by going from x_hat towards the origin until we hit boundary of polytope P
        //S.enlarge_n(1);
        Vector<double> p_hat = S_base.subcol(S.width());
        polyhedron_greedy(F, V, permutation, -1.0, *x_hat, p_hat);

        // Update R to account for modifying S.
        // r0 = R' \ (S' * p_hat)
        Vector<double> r0 = R_base.subcol(0, R.width(), R.height());
        S.transpose(); S.mvm(1.0, p_hat, 0.0, r0); S.transpose();
        R.transpose(); R.trsv(CblasLower, r0); R.transpose();

        // rho1 = sqrt(p_hat' * p_hat - r0' * r0);
        double rho1 = sqrt(p_hat.norm2()*p_hat.norm2() - r0.norm2()*r0.norm2());
        R.enlarge_m(1); R.enlarge_n(1);
        R(R.width()-1, R.height()-1) = rho1;
        S.enlarge_n(1);

        // Check current function value
        auto F_cur = F.eval(V, [&](int64_t i) -> bool { return ((*x_hat)(i)) < 0.0; } );
        if (F_cur < F_best) {
            // Save best F and get the unique minimal minimizer
            F_best = F_cur;
            A_best.clear();
            for (int64_t i = 0; i < (*x_hat).length(); i++) {
                if ((*x_hat)(i) < 0) {
                    A_best.push_back(i);
                }
            }
        }

        // Get suboptimality bound
        double sum_x_hat_lt_0 = 0.0;
        for (int64_t i = 0; i < x_hat->length(); i++) {
            if((*x_hat)(i) < 0)
                sum_x_hat_lt_0 += (*x_hat)(i);
        }
        double subopt = F_best - sum_x_hat_lt_0;

        std::cout << "Suboptimality bound: " << F_best-subopt << " <= min_A F(A) <= F(A_best) = " << F_best << "; delta <= " << subopt << std::endl;

        double xt_p = (*x_hat).dot(p_hat);
        double xnrm2 = (*x_hat).norm2();
        double xt_x = xnrm2 * xnrm2;
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
}

template<class RNG, class DIST>
list<int64_t> get_cols_to_delete(int64_t m, double percent_to_delete, RNG &gen, DIST& dist)
{
    int64_t n_cols_to_delete = std::round(m * percent_to_delete);
    set<int64_t> cols_to_delete;

    while(cols_to_delete.size() < n_cols_to_delete) {
        cols_to_delete.insert(dist(gen));
    }

    list<int64_t> to_ret;
    for(auto it = cols_to_delete.begin(); it != cols_to_delete.end(); ++it) {
        to_ret.push_back(*it);
    }

    return to_ret;
}

void benchmark_delete_cols()
{
    int64_t start = 16;
    int64_t end = 256;
    int64_t inc = 16;
    int64_t n_reps = 10;
    double percent_to_delete = 0.1; 

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

//    std::cout << "m\tn\tmean (s)\tstdev (s)\tCompulsory GB/s" << std::endl;

    int fw = 10;
    std::cout << std::setw(fw) << "m" << std::setw(fw) << "n" << std::setw(fw) << "nb" << std::setw(fw) << "nb2" << std::setw(2*fw) << "mean (s)" << std::setw(2*fw) <<  "stdev (s)" << std::setw(2*fw) << "BW" << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = 4096;
        int64_t m = 4096;
        int64_t nb = i; 

        std::uniform_int_distribution<> dist(0,n-1);
        std::vector<double> cycles;
        cycles.reserve(n_reps);

        for(int64_t r = 0; r < n_reps; r++) {
            list<int64_t> cols_to_delete = get_cols_to_delete(n, percent_to_delete, gen, dist);

            //1. Create random S
            Matrix<double> S(m,n);
            S.fill_rand(gen, normal);

            //2. Perform a QR factorization of S
//            Matrix<double> RT(n,m);
//            auto R = RT.transposed();
            Matrix<double> R(m,n);
            Vector<double> t(n);
            Matrix<double> T(nb,n);
            Matrix<double> V(cols_to_delete.size(),n);
            Matrix<double> ws(nb, n);
            Matrix<double> Rinit(m,n);
            R.copy(S);
            R.qr(t);
            Rinit.copy(R);
            
            auto R0 = R.submatrix(0,0,n,n);
            auto Rinit0 = Rinit.submatrix(0,0,n,n);

            //3. Call delete_cols_incremental_QR, timing it.
            cycles_count_start();
        //    S.remove_cols(cols_to_delete);
        //    R0.blocked_remove_cols_incremental_qr(cols_to_delete, t, nb);
            Rinit0.blocked_kressner_remove_cols_incremental_qr(R0, cols_to_delete, T, V, nb, 32, ws);
            //R0.kressner_remove_cols_incremental_qr(cols_to_delete, T, V, nb, ws);
            //R.remove_cols_incremental_qr(cols_to_delete, t);
            cycles.push_back(cycles_count_stop().time);
        }

        double mean = std::accumulate(cycles.begin(), cycles.end(), 0.0) / cycles.size();
        std::vector<double> diff(cycles.size());
        std::transform(cycles.begin(), cycles.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / cycles.size());
        std::cout << std::setw(fw) << m;
        std::cout << std::setw(fw) << n;
        std::cout << std::setw(fw) << nb;
        std::cout << std::setw(fw) << 32;
        std::cout << std::setw(2*fw) << mean;
        std::cout << std::setw(2*fw) << stdev;
        std::cout << std::setw(2*fw) << sizeof(double) * (2*n*n) / mean / 1e6  << " MB/s" << std::endl;
    }
}


int main() {
    run_validation_suite();
/*
    std::vector<int64_t> V;
    V.reserve(500);
    for(int i = 0; i < 500; i++){
        V.push_back(i);
    }
    IDivSqrtSize a;
    min_norm_point(a, V); 
*/
    benchmark_delete_cols();
}
