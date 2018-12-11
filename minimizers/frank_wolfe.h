#include "../la/vector.h"
#include "../submodular.h"

template<class DT>
std::vector<bool> FrankWolfe(SubmodularFunction<DT>& F, DT eps)
{
    int64_t k = 1;
    Vector<DT> x(F.n);
    Vector<DT> s(F.n);

    //Initialize x with something in B(F)
    s.fill_rand();
    F.polyhedron_greedy(1.0, s, x, NULL); 

    DT F_best = std::numeric_limits<DT>::max();
    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        //Find s
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);
        F_best = std::min(F_curr, F_best);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < eps) break;

        //Update X
        DT gamma = 2.0 / (k+2.0);
        x.axpby(gamma, s, 1.0 - gamma);

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
}

template<class DT>
std::vector<bool> FrankWolfeLineSearchGamma(SubmodularFunction<DT>& F, DT eps)
{
    int64_t k = 1;
    Vector<DT> x(F.n);
    Vector<DT> s(F.n);

    //Initialize x with something in B(F)
    s.fill_rand();
    F.polyhedron_greedy(1.0, s, x, NULL); 

    DT F_best = std::numeric_limits<DT>::max();
    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        //Find s
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);
        F_best = std::min(F_curr, F_best);

        //Test for termination
        DT xtx_minus_xts = x.dot(x) - x.dot(s);
        if(xtx_minus_xts < eps) break;


        //Calculate d
        auto d = s.subvector(0,F.n);
        d.axpy(-1.0, x);
        DT gamma = std::min(std::max(-x.dot(d) / d.dot(d), 0.0), 1.0);

        //Update x
        x.axpy(gamma, d);

        //Update duality gap
        DT sum_x_lt_0 = 0.0;
        for (int64_t i = 0; i < F.n; i++) { if(x(i) <= 0.0) sum_x_lt_0 += x(i); }
        duality_gap = std::abs(F_best - sum_x_lt_0);
        k++;
    }

    //Return A, minimizer of F
    std::vector<bool> A(F.n);
    for(int64_t i = 0; i < F.n; i++){ A[i] = x(i) <= 0.0; }
    return A;
}
