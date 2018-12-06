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

    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        //Find s
        DT F_curr = F.polyhedron_greedy_eval(-1.0, x, s, NULL);

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
