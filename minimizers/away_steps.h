#include "../la/vector.h"
#include "../la/matrix.h"
#include "../la/inc_qr_matrix.h"
#include "../submodular.h"

//R is upper triangular
template<class DT>
DT check_STS_eq_RTR(Matrix<DT>& S, IncQRMatrix<DT>& R_in) 
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

template<class DT>
std::vector<bool> AwaySteps(SubmodularFunction<DT>& F, DT eps)
{
    int64_t k = 1;

    //Iterates
    Vector<DT> x(F.n);
    Vector<DT> a_base(F.n); // x = S a
    Vector<DT> a = a_base.subvector(0, 1);

    Matrix<DT> S_base(F.n,F.n);
    IncQRMatrix<DT> R_base(F.n);

    Matrix<DT> S = S_base.submatrix(0, 0, F.n, 1);
    IncQRMatrix<DT> R = R_base.submatrix(0,1); //S = QR (implicit Q)

    //Workspace
    Vector<DT> v(F.n);
    Vector<DT> STx_base(F.n);
    Vector<DT> dFW(F.n); //Frank-Wolfe direction
    Vector<DT> dA(F.n);  //Away direction
    std::list<int64_t> to_remove;
    
    //Initialize x, a, S, and R
    Vector<DT> s0 = S.subcol(0);
    s0.fill_rand();
    F.polyhedron_greedy(1.0, s0, x, NULL); 
    s0.copy(x);
    R(0,0) = s0.norm2();
    a(0) = 1.0;

    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        assert(S.width() > 0);
        assert(S.width() < S_base.width());

        //Get s
        Vector<DT> s = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < 1e-5) break;

        //Get Frank-Wolfe and Away directions
        dFW.copy(s); dFW.axpy(-1.0, x);
        //v = argmax x^T v, v in S
        auto STx = STx_base.subvector(0, S.width());
        S.transpose(); S.mvm(1.0, x, 0.0, STx); S.transpose();
        auto v = S.subcol(STx.index_of_max());
        auto alpha_v = a(STx.index_of_max());
        dA.copy(x); dA.axpy(-1.0, v);

        Vector<DT>* d;
        DT gamma_max;
        if(x.dot(dFW) < x.dot(dA)) {
            //Forward direction
            d = &dFW;
            gamma_max = 1.0;
        } else {
            //Away direction
            d = &dA;
            gamma_max = alpha_v / (1.0 - alpha_v);
        }

        //Update X
        //Find gamma that minimizes norm of (x + gamma d)
        // gamma = -d^T x / d^T d
        DT gamma = std::min(std::max(-d->dot(x) / d->dot(*d), 0.0), gamma_max);
        if(gamma > 0.0) {
            x.axpy(gamma, *d);
        } else {
            std::cout << "gamma is zero" << std::endl;
        }

        //Update duality gap
        DT sum_x_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) { if(x(i) <= 0.0) sum_x_lt_0 += x(i); }
        duality_gap = std::abs(F_curr - sum_x_lt_0);
        if(duality_gap <= eps) break;

        // Update S and R
        bool updated_r = R.add_col_inc_qr(S,s);
        if(updated_r) {
            S.enlarge_n(1);

            //Update a. S a = x
            //Use R^T R a = S^T x
            a = a_base.subvector(0, S.width());
            a.copy(STx);
            R.transpose(); R.trsv(a); R.transpose();
            R.trsv(a);
            
            //Remove bad columns from S and R.
            bool removed = true;
            do {
                to_remove.clear();
                for(int64_t i = 0; i < a.length() && to_remove.size() < S.width(); i++) {
                    if(a(i) < 0) to_remove.push_back(i);
                }
                if(to_remove.size() == S.width()) {
                    std::cout << "removing all of S" << std::endl;
                    a.print("a");
                }
                if(to_remove.size() == 0 && S.width() == S_base.width()) to_remove.push_back(0);
                removed = to_remove.size() > 0;
                if(to_remove.size() > 0) {
                    //Remove columns from S and R
                    S.remove_cols(to_remove);
                    R.remove_cols_inc_qr(to_remove);

                    //Update a
                    auto STx = STx_base.subvector(0, S.width());
                    a = a_base.subvector(0, S.width());
                    S.transpose(); S.mvm(1.0, x, 0.0, STx); S.transpose();
                    a.copy(STx);
                    R.transpose(); R.trsv(a); R.transpose();
                    R.trsv(a);
                }
            } while(removed == true);
        } else {
            std::cout << "Could not update R" << std::endl;
            R.print("R");
            DT err = check_STS_eq_RTR(S,R);
            std::cout << "R checksum " << err << std::endl;
        }

        k++;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}
/*
template<class DT>
std::vector<bool> AwaySteps2(SubmodularFunction<DT>& F, DT eps)
{
    int64_t k = 1;

    //Iterates
    Vector<DT> x(F.n);
    Vector<DT> dFW(F.n); //Frank-Wolfe direction
    Vector<DT> dA(F.n);  //Away direction
    
    //Initialize x, a, S, and R
    Vector<DT> s(F.n);
    s.fill_rand();
    F.polyhedron_greedy(1.0, s, x, NULL); 
    s.copy(x);

    Vector<DT> v(F.n);

    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        //Get s
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < eps) break;

        //Frank-Wolfe direction
        dFW.copy(s); dFW.axpy(-1.0, x);
        
        //Away direction
        F.polyhedron_greedy(1.0, x, v, NULL);
        dA.copy(x); dA.axpy(-1.0, v);

        Vector<DT>* d;
        if(x.dot(dFW) < x.dot(dA)) {
            //Forward direction
            d = &dFW;
        } else {
            //Away direction
            d = &dA;
        }

        //Update X
        //Find gamma that minimizes norm of (x + gamma d)
        // gamma = -d^T x / d^T d
        DT gamma = std::min(-d->dot(x) / d->dot(*d), 1.0);
        if(gamma > 0.0)
            x.axpy(gamma, *d);

        //Update duality gap
        DT sum_x_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) { if(x(i) <= 0.0) sum_x_lt_0 += x(i); }
        duality_gap = std::abs(F_curr - sum_x_lt_0);
        k++;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}*/
