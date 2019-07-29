#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include <random>

#include "set_fn/submodular.h"
#include "set_fn/graph_cut.h"
#include "set_fn/log_det.h"
#include "set_fn/iwata_test.h"
#include "set_fn/deep.h"
#include "set_fn/st_constrain.h"
#include "set_fn/plus_modular.h"

#include "minimizers/mnp.h"
#include "minimizers/bvh.h"

#include "minimizers/frank_wolfe.h"
#include "minimizers/away_steps.h"
#include "minimizers/pairwise.h"

#include "fujishige/wrapper.h"

#include "perf/perf.h"
#include "test/validate.h"
#include "test/bench.h"

#include "la/vector.h"
#include "la/matrix.h"
#include "util.h"

#include "perf_log.h"

//#define SLOW_GREEDY
//#define PRINT_HIST
int64_t fw = 8;

template<class DT>
void frank_wolfe_wolfe_mincut()
{
    int64_t start = 50;
    int64_t end = 10000;
    int64_t inc = 50;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n" << ","; 
    std::cout << std::setw(2*fw) << "MNP T" << ",";
    std::cout << std::setw(2*fw) << "FrankWolfe T" << ",";
    std::cout << std::setw(2*fw) << "AwaySteps T" << ",";
    std::cout << std::setw(2*fw) << "Pairwise T" << ",";
    std::cout << std::setw(2*fw) << "MNP N" << ",";
    std::cout << std::setw(2*fw) << "FrankWolfe N" << ",";
    std::cout << std::setw(2*fw) << "AwaySteps N" << ",";
    std::cout << std::setw(2*fw) << "Pairwise N" << ",";
    std::cout << std::setw(2*fw) << "MNP |S|" << ",";
    std::cout << std::setw(2*fw) << "AS |S|" << ",";
    std::cout << std::setw(2*fw) << "PW |S|" << ",";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-5);
            double mnp_fa = problem.eval(mnp_A);
            double mnp_seconds = (double) cycles_count_stop().time;
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_s_card = PerfLog::get().get_total("S WIDTH");
            
            //Vanilla FW
            PerfLog::get().clear();
            cycles_count_start();
            auto fw_A = FrankWolfe(problem, 1e-5);
            double fw_seconds = (double) cycles_count_stop().time;
            double fw_fa = problem.eval(fw_A);
            int64_t fw_iterations = PerfLog::get().get_total("ITERATIONS");

            //Away Steps FW
            PerfLog::get().clear();
            cycles_count_start();
            auto as_A = AwaySteps(problem, 1e-5, -1);
            double as_seconds = (double) cycles_count_stop().time;
            double as_fa = problem.eval(as_A);
            int64_t as_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t as_s_card = PerfLog::get().get_total("S WIDTH");

            //Pairwise
            PerfLog::get().clear();
            cycles_count_start();
            auto pw_A = Pairwise(problem, 1e-5, -1);
            double pw_seconds = (double) cycles_count_stop().time;
            double pw_fa = problem.eval(pw_A);
            int64_t pw_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t pw_s_card = PerfLog::get().get_total("S WIDTH");

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n << ",";
            std::cout << std::setw(2*fw) << mnp_seconds << ",";
            std::cout << std::setw(2*fw) << fw_seconds << ",";
            std::cout << std::setw(2*fw) << as_seconds << ",";
            std::cout << std::setw(2*fw) << pw_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_iterations << ",";
            std::cout << std::setw(2*fw) << fw_iterations << ",";
            std::cout << std::setw(2*fw) << as_iterations << ",";
            std::cout << std::setw(2*fw) << pw_iterations << ",";
            std::cout << std::setw(2*fw) << (double) mnp_s_card / (double) mnp_iterations << ",";
            std::cout << std::setw(2*fw) << (double) as_s_card / (double) as_iterations << ",";
            std::cout << std::setw(2*fw) << (double) pw_s_card / (double) pw_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_fa - fw_fa + as_fa - pw_fa << ",";
            std::cout << std::endl;
        }
    }
}

template<class DT>
void mnp_bvh()
{
    int64_t start = 1000;
    int64_t end = 10000;
    int64_t inc = 1000;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking MNP and Simplicial Decomposition" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "MNP_|A|"; 
    std::cout << std::setw(2*fw) << "MNP_F(A)";
    std::cout << std::setw(2*fw) << "BVH_F(A)";
    std::cout << std::setw(2*fw) << "MNP_T";
    std::cout << std::setw(2*fw) << "BVH_T";
    std::cout << std::setw(2*fw) << "MNP_N";
    std::cout << std::setw(2*fw) << "BVH_N";
    std::cout << std::setw(2*fw) << "MNP_C";
    std::cout << std::setw(2*fw) << "BVH_C";
    std::cout << std::setw(2*fw) << "MNP_|S|";
    std::cout << std::setw(2*fw) << "BVH_|S|";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            //LogDet<DT> problem(n);
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);
            //problem.Groups(16, 0.25, 1e-5);
            //Deep<DT> problem(n);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-10);
            double mnp_fa = problem.eval(mnp_A);
            double mnp_seconds = (double) cycles_count_stop().time;
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_s_card = PerfLog::get().get_total("S WIDTH");

            //BVH
            PerfLog::get().clear();
            cycles_count_start();
            auto bvh_A = bvh(problem, 1e-5, 1e-10, 1e-8);
            double bvh_fa = problem.eval(bvh_A);
            double bvh_seconds = (double) cycles_count_stop().time;
            int64_t bvh_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t bvh_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t bvh_s_card = PerfLog::get().get_total("S WIDTH");

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << mnp_fa;
            std::cout << std::setw(2*fw) << bvh_fa;
            std::cout << std::setw(2*fw) << mnp_seconds << ",";
            std::cout << std::setw(2*fw) << bvh_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_iterations << ",";
            std::cout << std::setw(2*fw) << bvh_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_minor_cycles << ",";
            std::cout << std::setw(2*fw) << bvh_minor_cycles << ",";
            std::cout << std::setw(2*fw) << (double) mnp_s_card / (double) mnp_iterations;
            std::cout << std::setw(2*fw) << (double) bvh_s_card / (double) bvh_iterations;
            std::cout << std::endl;
        }
    }
}

template<class DT>
void mnp_order_k()
{
    int64_t start = 100;
    int64_t end = 1000;
    int64_t inc = 100;
    int64_t n_reps = 20;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking MNP and Simplicial Decomposition" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(2*fw) << "MNP F(A)";
    std::cout << std::setw(2*fw) << "Spec F(A)";
    std::cout << std::setw(2*fw) << "MNP T";
    std::cout << std::setw(2*fw) << "Spec T";
    std::cout << std::setw(2*fw) << "MNP N";
    std::cout << std::setw(2*fw) << "Spec N";
    std::cout << std::setw(2*fw) << "MNP C";
    std::cout << std::setw(2*fw) << "Spec C";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            //LogDet<DT> problem(n);
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-10, 1e-10);
            double mnp_seconds = (double) cycles_count_stop().time;
            double mnp_fa = problem.eval(mnp_A);
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_speculate_A = mnp_order_k(problem, 1e-10, 1e-10);
            double mnp_speculate_seconds = (double) cycles_count_stop().time;
            double mnp_speculate_fa = problem.eval(mnp_speculate_A);

            int64_t mnp_speculate_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_speculate_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(2*fw) << mnp_fa;
            std::cout << std::setw(2*fw) << mnp_speculate_fa;
            std::cout << std::setw(2*fw) << mnp_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_speculate_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_speculate_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_minor_cycles << ",";
            std::cout << std::setw(2*fw) << mnp_speculate_minor_cycles << ",";
            std::cout << std::endl;

        }
    }
}

template<class DT>
void frank_wolfe_mincut_err_vs_time()
{
    int64_t n = 100;

    std::vector<std::vector<double>> times;
    std::vector<std::vector<double>> dualities;

    //Initialize min norm point problem
    PerfLog::get().clear();
    MinCut<DT> problem(n);
    problem.seed = 3785515132;
    problem.WattsStrogatz(16, 0.25);

    //Run everything once to warm it up
    mnp(problem, 1e-5, 1e-5);
    //FrankWolfe(problem, 1e-5);
    AwaySteps(problem, 1e-5, -1);
    Pairwise(problem, 1e-5, -1);

    //MNP
    PerfLog::get().clear();
    std::cout << std::setw(fw) << "\"MNP T\"" << ", " << std::setw(fw) << "\"MNP D\"" << ", ";
    mnp(problem, 1e-5, 1e-5);
    dualities.emplace_back(PerfLog::get().get_sequence("MNP DUALITY"));
    times.emplace_back(PerfLog::get().get_sequence("MNP CUMMULATIVE TIME"));
       
    //Vanilla FW 
   /* PerfLog::get().clear();
    std::cout << std::setw(fw) << "\"FW T\"" << ", " << std::setw(fw) << "\"FW D\"" << ", ";
    FrankWolfe(problem, 1e-5);
    dualities.emplace_back(PerfLog::get().get_sequence("FW DUALITY"));
    times.emplace_back(PerfLog::get().get_sequence("FW CUMMULATIVE TIME"));*/

    //Away Steps FW
    PerfLog::get().clear();
    std::cout << std::setw(fw) << "\"AS T\"" << ", " << std::setw(fw) << "\"AS D\"" << ", ";
    AwaySteps(problem, 1e-5, -1);
    dualities.emplace_back(PerfLog::get().get_sequence("AS DUALITY"));
    times.emplace_back(PerfLog::get().get_sequence("AS CUMMULATIVE TIME"));
    for(int64_t pruning = 16; pruning < 4096; pruning *= 2) {
        std::cout << std::setw(fw) << "\"ASP" + std::to_string(pruning) + " T\"" << ", " << std::setw(fw) << "\"ASP" + std::to_string(pruning) + " D\"" << ", ";
        PerfLog::get().clear();
        AwaySteps(problem, 1e-5, pruning);
        dualities.emplace_back(PerfLog::get().get_sequence("AS DUALITY"));
        times.emplace_back(PerfLog::get().get_sequence("AS CUMMULATIVE TIME"));
    }
    
    //PW FW
    std::cout << std::setw(fw) << "\"PW T\"" << ", " << std::setw(fw) << "\"PW D\"" << ", ";
    Pairwise(problem, 1e-5, -1);
    dualities.emplace_back(PerfLog::get().get_sequence("PW DUALITY"));
    times.emplace_back(PerfLog::get().get_sequence("PW CUMMULATIVE TIME"));
    for(int64_t pruning = 16; pruning < 4096; pruning *= 2) {
        std::cout << std::setw(fw) << "\"PWP" + std::to_string(pruning) + " T\"" << ", " << std::setw(fw) << "\"PWP" + std::to_string(pruning) + " D\"" << ", ";
        PerfLog::get().clear();
        Pairwise(problem, 1e-5, pruning);
        dualities.emplace_back(PerfLog::get().get_sequence("PW DUALITY"));
        times.emplace_back(PerfLog::get().get_sequence("PW CUMMULATIVE TIME"));
    }
    std::cout << std::endl;

    size_t max_len = 0;
    for(uint64_t i = 0; i < times.size(); i++) max_len = std::max(max_len, times[i].size());
    for(uint64_t j = 0; j < max_len; j++) {
        for(uint64_t i = 0; i < times.size(); i++) {
            assert(times[i].size() == dualities[i].size());
            if(j < times[i].size()) { 
                std::cout << std::setw(fw) << times[i][j] / 3.6e9 << ", " << std::setw(fw) << dualities[i][j] << ", ";
            } else {
                std::cout << std::setw(fw) << " " << ", " << std::setw(fw) << " " << ", ";
            }
        }
        std::cout << std::endl;
    }
}

extern int number_extreme_point;

void test_versus_fujishige()
{
    int64_t start = 100;
    int64_t end = 10000;
    int64_t inc = 100;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(2*fw) << "F cycles"; 
    std::cout << std::setw(2*fw) << "T cycles"; 
    std::cout << std::setw(fw) << "|A|"; 
    std::cout << std::setw(2*fw) << "F MNP F(A)"; 
    std::cout << std::setw(2*fw) << "T MNP F(A)"; 
    std::cout << std::setw(2*fw) << "F"; 
    std::cout << std::setw(2*fw) << "T"; 
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            MinCut<double> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //Fujishige FW
            number_extreme_point = 0;
            cycles_count_start();
            auto fw_A = run_isotani_and_fujishige(problem);
            double fw_seconds = (double) cycles_count_stop().time;
            double fw_fa = problem.eval(fw_A);
            std::cout << std::setw(2*fw) << number_extreme_point;

            //MNP
            PerfLog::get().set_total("ITERATIONS", 0);
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-10, 1e-10);
            double mnp_seconds = (double) cycles_count_stop().time;
            double mnp_fa = problem.eval(mnp_A);
            std::cout << std::setw(2*fw) << PerfLog::get().get_total("ITERATIONS"); 

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << fw_fa;
            std::cout << std::setw(2*fw) << mnp_fa;
            std::cout << std::setw(2*fw) << fw_seconds;
            std::cout << std::setw(2*fw) << mnp_seconds;
            std::cout << std::endl;
        }
    }
}

#include "minimizers/mnp_fw.h"
template<class DT>
void mnp_fw()
{
    int64_t start = 1000;
    int64_t end = 10000;
    int64_t inc = 1000;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking MNP and MNP_FW" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "MNP_|A|"; 
    std::cout << std::setw(2*fw) << "MNP_F(A)";
    std::cout << std::setw(2*fw) << "MNP_FW_F(A)";
    std::cout << std::setw(2*fw) << "MNP_T";
    std::cout << std::setw(2*fw) << "MNP_FW_T";
    std::cout << std::setw(2*fw) << "MNP_N";
    std::cout << std::setw(2*fw) << "MNP_FW_N";
    std::cout << std::setw(2*fw) << "MNP_C";
    std::cout << std::setw(2*fw) << "MNP_FW_C";
    std::cout << std::setw(2*fw) << "MNP_|S|";
    std::cout << std::setw(2*fw) << "MNP_FW_|S|";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-10);
            double mnp_seconds = (double) cycles_count_stop().time;
            double mnp_fa = problem.eval(mnp_A);
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_s_card = PerfLog::get().get_total("S WIDTH");

            //BVH
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_fw_A = mnp_fw(problem, 1e-5, 1e-10);
            double mnp_fw_seconds = (double) cycles_count_stop().time;
            double mnp_fw_fa = problem.eval(mnp_fw_A);
            int64_t mnp_fw_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_fw_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_fw_s_card = PerfLog::get().get_total("S WIDTH");

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << mnp_fa;
            std::cout << std::setw(2*fw) << mnp_fw_fa;
            std::cout << std::setw(2*fw) << mnp_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_fw_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_fw_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_minor_cycles << ",";
            std::cout << std::setw(2*fw) << mnp_fw_minor_cycles << ",";
            std::cout << std::setw(2*fw) << (double) mnp_s_card / (double) mnp_iterations;
            std::cout << std::setw(2*fw) << (double) mnp_fw_s_card / (double) mnp_fw_iterations;
            std::cout << std::endl;
        }
    }
}

template<class DT, class GEN, class DIST>
void mnp_deep(GEN &gen, DIST &dist, const std::vector<int64_t> layers, const std::string desc)
{
    int64_t start = 1000;
    int64_t end = 10000;
    int64_t inc = 1000;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking MNP for Deep Submodular Functions" << std::endl;
    std::cout << std::setw(25) << desc << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n" << ","; 
    std::cout << std::setw(fw) << "MNP_|A|" << ","; 
    std::cout << std::setw(2*fw) << "MNP_F(A)" << ",";
    std::cout << std::setw(2*fw) << "MNP_T" << ",";
    std::cout << std::setw(2*fw) << "MNP_N" << ",";
    std::cout << std::setw(2*fw) << "MNP_C" << ",";
    std::cout << std::setw(2*fw) << "MNP_|S|" << ",";
    std::cout << std::setw(2*fw) << "mhat test" << ",";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize deep submodular problem
            Deep<double> deep(n+2, layers);
            deep.init_weights(gen, dist);
//            deep.rectify = [](double x){ return std::sqrt(x); };
            deep.rectify = [](double x){ return std::min(x, 1.0); };

            PlusModular<double, Deep<double>> deep_plus_modular(n+2, std::move(deep), dist);
            STConstrain<double, PlusModular<double, Deep<double>>> problem(n, deep_plus_modular);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-10);
            double mnp_fa = problem.eval(mnp_A);
            double mnp_seconds = (double) cycles_count_stop().time;
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_s_card = PerfLog::get().get_total("S WIDTH");

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n << ",";
            std::cout << std::setw(fw) << cardinality << ",";
            std::cout << std::setw(2*fw) << mnp_fa << ",";
            std::cout << std::setw(2*fw) << mnp_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_minor_cycles << ",";
            std::cout << std::setw(2*fw) << (double) mnp_s_card / (double) mnp_iterations << ",";
            
            std::vector<bool> S(n);
            std::random_device rd;
            std::mt19937 gen{rd()};
            std::bernoulli_distribution dist(0.3);
            std::generate(S.begin(), S.end(), [&dist, &gen](){ return dist(gen); });
            std::cout << std::setw(2*fw) << problem.m_hat(S) - problem.eval(S);
            std::cout << std::endl;
        }
    }
}

#include "set_fn/hypergraph_cut.h"
template<class DT>
void mnp_hyper(int64_t r)
{
    int64_t start = 1000;
    int64_t end = 5000;
    int64_t inc = 1000;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking MNP for Hypergraph cuts. r = " << r << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << std::setw(fw) << "n" << ","; 
    std::cout << std::setw(fw) << "MNP_|A|" << ","; 
    std::cout << std::setw(2*fw) << "MNP_F(A)" << ",";
    std::cout << std::setw(2*fw) << "MNP_T" << ",";
    std::cout << std::setw(2*fw) << "MNP_N" << ",";
    std::cout << std::setw(2*fw) << "MNP_C" << ",";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            HyperCut<DT> cut(n+2);
            cut.GeneralizedWattsStrogatz(16, r, 0.25);
            STConstrain<double, HyperCut<double>> problem(n, cut);

            for(auto & e: problem.submodular.edges) {
                if(std::find(e.v.begin(), e.v.end(), problem.s) != e.v.end() || std::find(e.v.begin(), e.v.end(), problem.t) != e.v.end()) {
                    e.w += 10.0;
                }
            }
            problem.recalculate_baseline();
            std::vector<bool> empty(n);
            std::fill(empty.begin(), empty.end(), false);
            assert(problem.eval(empty) == 0.0);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-10);
            double mnp_seconds = (double) cycles_count_stop().time;
            double mnp_fa = problem.eval(mnp_A);
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n << ",";
            std::cout << std::setw(fw) << cardinality << ",";
            std::cout << std::setw(2*fw) << mnp_fa << ",";
            std::cout << std::setw(2*fw) << mnp_seconds << ",";
            std::cout << std::setw(2*fw) << mnp_iterations << ",";
            std::cout << std::setw(2*fw) << mnp_minor_cycles << ",";
            std::cout << std::endl;
        }
    }
}

int main() 
{
    run_validation_suite();

    fw = 0; //Make excel-readable
    //fw = 8; //Make human-readable

    std::random_device rd;
    std::mt19937 gen(rd());
    
    double p = 0.19;
    int64_t layer_size = 10;
    std::bernoulli_distribution bern(p);
    std::vector<int64_t> layers;
    for(int64_t i = 1; i < 10; i++) {
        layers.push_back(layer_size);
        mnp_deep<double, std::mt19937, std::bernoulli_distribution>(gen, bern, layers, "Bernoulli Distribution p = " + std::to_string(p) + " " + std::to_string(i) + " layer size " + std::to_string(layer_size)); 
    }
    exit(1);

    mnp_deep<double, std::mt19937, std::bernoulli_distribution>(gen, bern, layers, "Bernoulli Distribution"); 
//    mnp_deep<double, std::mt19937, std::uniform_real_distribution<double>>(gen, uniform, layers, "Uniform Distribution"); 

    //Some hypergraph stuff.
    for(int64_t r = 1; r <= 8; r *= 2) {
        mnp_hyper<double>(r);
    }



    for(int i = 4; i < 64; i+=4) {
        layers.clear();
        for(int j = 0; j < 4; j++){
            layers.push_back(i);
        }
        mnp_deep<double, std::mt19937, std::bernoulli_distribution>(gen, bern, layers, 
                std::string() + "Bernoulli Distribution. 4 Layers. Layer size " + std::to_string(i)); 
    }


    layers.clear();
    for(int i = 0; i < 16; i++) {
        layers.push_back(16);
        mnp_deep<double, std::mt19937, std::bernoulli_distribution>(gen, bern, layers, 
                std::string() + "Bernoulli Distribution. " + std::to_string(i) + " Layers."); 
    }

    exit(1);
    run_benchmark_suite();
    mkl_free_buffers(); 
}
