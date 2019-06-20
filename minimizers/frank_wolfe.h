#include "../la/vector.h"
#include "../set_fn/submodular.h"

template<class DT>
std::vector<bool> FrankWolfe(SubmodularFunction<DT>& F, DT eps)
{
    PerfLog::get().add_sequence("FW CUMMULATIVE TIME");
    PerfLog::get().add_sequence("FW DUALITY");
    int64_t initial_time = rdtsc();

    int64_t k = 1;
    Vector<DT> x(F.n);
    Vector<DT> s(F.n);

    //Initialize x with something in B(F)
    s.fill_rand();
    F.polyhedron_greedy_decending(s, x); 

    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A_curr(F.n);
    std::vector<bool> A_best(F.n);

    DT F_thresh;
    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        //Find s
        DT F_curr = F.polyhedron_greedy_ascending(x, s, A_curr);
        if(F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A_best[i] = A_curr[i];
        }

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

        if(k % LOG_FREQ == 0) {
            PerfLog::get().log_sequence("FW CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("FW DUALITY", duality_gap);
        }
        k++;
    }

    PerfLog::get().log_total("ITERATIONS", k);

    //Return A, minimizer of F
    return A_best;
}

template<class DT>
std::vector<bool> FrankWolfeLineSearchGamma(SubmodularFunction<DT>& F, DT eps)
{
    PerfLog::get().add_sequence("FW_LS CUMMULATIVE TIME");
    PerfLog::get().add_sequence("FW_LS DUALITY");
    int64_t initial_time = rdtsc();

    int64_t k = 1;
    Vector<DT> x(F.n);
    Vector<DT> s(F.n);

    //Initialize x with something in B(F)
    s.fill_rand();
    F.polyhedron_greedy_decending(s, x); 

    DT F_best = std::numeric_limits<DT>::max();
    std::vector<bool> A_curr(F.n);
    std::vector<bool> A_best(F.n);

    DT duality_gap = 1.0;
    while(duality_gap > eps) {
        //Find s
        DT F_curr = F.polyhedron_greedy_ascending(x, s, A_curr);
        if(F_curr < F_best) {
            F_best = F_curr;
            for(int64_t i = 0; i < F.n; i++)
                A_best[i] = A_curr[i];
        }

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

        if(k % LOG_FREQ == 0) {
            PerfLog::get().log_sequence("FW_LS CUMMULATIVE TIME", rdtsc() - initial_time);
            PerfLog::get().log_sequence("FW_LS DUALITY", duality_gap);
        }
        k++;
    }

    PerfLog::get().log_total("ITERATIONS", k);

    //Return A, minimizer of F
    return A_best;
}
