#include "../la/vector.h"
#include "../la/matrix.h"
#include "../la/inc_qr_matrix.h"
#include "../submodular.h"

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

template<class DT>
DT check_Sa_eq_x(const Matrix<DT>& S, const Vector<DT>& a, const Vector<DT>& x) 
{
    assert(S.width() == a.length());
    assert(S.height() == x.length());

    Vector<DT> t(S.height());
    S.mvm(1.0, a, 0.0, t);
    t.axpy(-1.0, x);
    return t.norm2();
}

template<class DT>
std::vector<bool> AwaySteps(SubmodularFunction<DT>& F, DT eps)
{
    int64_t k = 1;
    int64_t max_away = 2048;

    //Iterates
    Vector<DT> x(F.n);
    Vector<DT> a_base(max_away); // x = S a
    auto a = a_base.subvector(0, 1);

    Matrix<DT> S_base(F.n,max_away);
    auto S = S_base.submatrix(0, 0, F.n, 1);

    //Workspace
    Vector<DT> STx_base(max_away);
    Vector<DT> dFW(F.n); //Frank-Wolfe direction
    Vector<DT> dA(F.n);  //Away direction

    Vector<DT> tmp(F.n);
    
    //Initialize x, a, S
    Vector<DT> s0 = S.subcol(0);
    s0.fill_rand();
    F.polyhedron_greedy(1.0, s0, x, NULL); 
    s0.copy(x);
    a(0) = 1.0;

    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        assert(S.width() > 0);
        assert(S.width() < S_base.width());
        assert(a.has_nan() == false);
        assert(a.max() <= 1.0);
        assert(a.min() >= 0.0);

        //Get s
        auto s = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < 1e-5) break;

        //Get Frank-Wolfe and Away directions
        dFW.copy(s); dFW.axpy(-1.0, x);
        //v = argmax x^T v, v in S
        auto STx = STx_base.subvector(0, S.width());
        S.transpose(); S.mvm(1.0, x, 0.0, STx); S.transpose();
        int64_t v_index = STx.index_of_max();
        const auto v = S.subcol(v_index);
        auto alpha_v = a(v_index);
        dA.copy(x); dA.axpy(-1.0, v);

        //Get rid of minimum (don't keep track of it) 
        if(S.width()+1 >= S_base.width()) {
            int64_t min_index = STx.index_of_min();
            if(min_index == v_index) {
                STx.print("STx");
                x.print("x");
                a.print("a");
                S.print("S");
            }
            assert(min_index != v_index);
            if(v_index > min_index) 
                v_index--;
            S.remove_col(min_index);
            a.remove(min_index);
            auto s_to = S_base.subcol(S.width());
            s_to.copy(s);
            s = s_to;
        }

        if(x.dot(dFW) <= x.dot(dA)) {
            //Forward direction
            DT gamma = std::min(std::max(-dFW.dot(x) / dFW.dot(dFW), 0.0), 1.0);

            //Update weights and S
            if(gamma == 1.0) {
                //S = {s}
                S = S_base.submatrix(0,0,F.n,1);
                auto s0 = S.subcol(0);
                s0.copy(s);

                //a = [1]
                a = a_base.subvector(0,1);
                a(0) = 1.0;

                //update x
                x.copy(s);
            } else {
                //Check to see if s is in S already
                bool s_already_in_S = false;
                int64_t index_s = -1;
                for(int64_t i = 0; i < S.width(); i++) {
                    const auto si = S.subcol(i);
                    tmp.copy(s);
                    tmp.axpy(-1.0, si);
                    if(tmp.norm2() < 1e-2) {
                        s_already_in_S = true;
                        index_s = i;
                        break;
                    }
                }
                
                //Update a and S
                a.scale(1.0-gamma);
                if( s_already_in_S ) {
                    a(index_s) += gamma;
                } else {
                    S.enlarge_n(1);
                    a = a_base.subvector(0, a.length()+1);
                    a(a.length() - 1) = gamma;
                }
                
                //update x
                x.axpy(gamma, dFW);
            }
        } else {
            //Away direction
            DT gamma_max = alpha_v / (1.0 - alpha_v);
            DT gamma = std::min(std::max(-dA.dot(x) / dA.dot(dA), 0.0), gamma_max);
            bool drop_step = (gamma == gamma_max);

            if(drop_step) {
                S.remove_col(v_index);
                a.remove(v_index);
            }
            
            //Update a
            a.scale(1+gamma);
            if(!drop_step) {
                a(v_index) -= gamma;
            }
            
            //Update x
            x.axpy(gamma, dA);
        }
        k++;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}

/*template<class DT>
std::vector<bool> AwaySteps2(SubmodularFunction<DT>& F, DT eps)
{
    int64_t mult = 4;
    int64_t k = 1;

    //Iterates
    Vector<DT> x(F.n);
    Vector<DT> a_base(mult*F.n); // x = S a
    auto a = a_base.subvector(0, 1);

    Matrix<DT> S_base(F.n,mult*F.n);
    auto S = S_base.submatrix(0, 0, F.n, 1);

    //Workspace
    Vector<DT> STx_base(mult*F.n);
    Vector<DT> dFW(F.n); //Frank-Wolfe direction
    Vector<DT> dA(F.n);  //Away direction

    Vector<DT> tmp(F.n);
    
    //Initialize x, a, S
    Vector<DT> s0 = S.subcol(0);
    s0.fill_rand();
    F.polyhedron_greedy(1.0, s0, x, NULL); 
    s0.copy(x);
    a(0) = 1.0;
   
    //Incremental QR factorization of S 
    IncQRMatrix<DT> R_base(mult*F.n);
    IncQRMatrix<DT> R = R_base.submatrix(0,1); //S = QR (implicit Q)
    R(0,0) = s0.norm2();

    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        assert(S.width() > 0);
        assert(S.width() < S_base.width());
        assert(S.width() == a.length()); 

        //Get s
        auto s = S_base.subcol(S.width());
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < 1e-5) break;

        //Get Frank-Wolfe and Away directions
        dFW.copy(s); dFW.axpy(-1.0, x);
        //v = argmax x^T v, v in S
        auto STx = STx_base.subvector(0, S.width());
        S.transpose(); S.mvm(1.0, x, 0.0, STx); S.transpose();
        int64_t v_index = STx.index_of_max();
        const auto v = S.subcol(v_index);
        auto alpha_v = a(v_index);
        dA.copy(x); dA.axpy(-1.0, v);

        if(x.dot(dFW) < x.dot(dA)) {
            //Forward direction
            DT gamma = std::min(std::max(-dFW.dot(x) / dFW.dot(dFW), 0.0), 1.0);

            //Update weights and S
            if(gamma == 1.0) {
                //S = {s}
                S = S_base.submatrix(0,0,F.n,1);
                auto s0 = S.subcol(0);
                s0.copy(s);

                //a = [1]
                a = a_base.subvector(0,1);
                a(0) = 1.0;

                //update x
                x.copy(s);
            } else {
                //Check to see if s is in span S already
                bool s_already_in_S = !R.add_col_inc_qr(S, s);
                if(s_already_in_S) {
                    //Update x.
                    x.axpy(gamma, dFW);

                    //Do a solve to maintain S a = x
                    S.transpose(); S.mvm(1.0, x, 0.0, a); S.transpose();
                    R.transpose(); R.trsv(a); R.transpose();
                    R.trsv(a);
                } else {
                    //S was not in the span of S
                    
                    //Update a and S
                    a.scale(1.0-gamma);
                    S.enlarge_n(1);
                    a = a_base.subvector(0, a.length()+1);
                    a(a.length()-1) = gamma;
                    //update x
                    x.axpy(gamma, dFW);
                }
            }
        } else {
            //Away direction
            DT gamma_max = alpha_v / (1.0 - alpha_v);
            DT gamma = std::min(std::max(-dA.dot(x) / dA.dot(dA), 0.0), gamma_max);

            if(gamma == gamma_max) {
                //Drop step. get rid of vector of s and element of a
                S.remove_col(v_index);
                R.remove_col_inc_qr(v_index);

                //copy trailing part of a into tmp vector
                if(v_index != a.length() - 1) {
                    auto t = tmp.subvector(0, a.length() - v_index - 1);
                    const auto a2 = a.subvector(v_index+1, a.length() - v_index - 1);
                    t.copy(a2);
                    auto a1 = a.subvector(v_index, a.length() - v_index - 1);
                    a1.copy(t);
                }
                a = a_base.subvector(0, a.length() - 1);
            }
            
            //Update a
            a.scale(1+gamma);
            if(gamma != gamma_max) {
                a(v_index) -= gamma;
            }
            
            //Update x
            x.axpy(gamma, dA);
        }
        k++;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}*/
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
