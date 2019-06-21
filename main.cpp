#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include <random>

#include "set_fn/submodular.h"
#include "set_fn/graph_cut.h"
#include "set_fn/log_det.h"
#include "set_fn/iwata_test.h"
#include "set_fn/coverage.h"
#include "set_fn/deep.h"

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

template<class DT>
void benchmark_logdet(DT eps, DT tol)
{
    int64_t start = 4000;
    int64_t end = 10000;
    int64_t inc = 500;
    int64_t n_reps = 10;


    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking log det" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 8;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "|A|"; 
    std::cout << std::setw(2*fw) << "seconds";
    std::cout << std::setw(2*fw) <<  "major";
#ifdef SLOW_GREEDY
    std::cout << std::setw(2*fw) << "slowseconds";
    std::cout << std::setw(2*fw) << "slowmajor";
#endif
    std::cout << std::setw(2*fw) <<  "minor";
    std::cout << std::setw(2*fw) <<  "add col %";
    std::cout << std::setw(2*fw) <<  "del col %";
    std::cout << std::setw(2*fw) <<  "del col qr %";
    std::cout << std::setw(2*fw) <<  "solve %";
    std::cout << std::setw(2*fw) <<  "vector %";
    std::cout << std::setw(2*fw) <<  "greedy %";
    std::cout << std::setw(2*fw) <<  "total %";
    std::cout << std::setw(2*fw) <<  "MVM MB/S";
    std::cout << std::setw(2*fw) <<  "TRSV MB/S";
    std::cout << std::setw(2*fw) <<  "del cols MB/S";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            int64_t max_iter = 1e5;
            //Initialize min norm point problem
            LogDet<DT> problem(n);
#ifdef SLOW_GREEDY
            SlowLogDet<DT> slow_problem(problem);
#endif
        
            //Initial condition    
            Vector<DT> wA(n);
            wA.fill_rand();

            //Time problem
            cycles_count_start();
            auto A = mnp(problem, wA, eps, tol);
            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

#ifdef PRINT_HIST
            auto col_hist = PerfLog::get().get_hist("NUM COLUMNS");
            for(int i = 0; i < col_hist.buckets.size(); i++) {
                std::cout << std::setw(8) << col_hist.min + col_hist.bucket_size * i;
            }
            std::cout << std::endl;
            for(int i = 0; i < col_hist.buckets.size(); i++) {
                std::cout << std::setw(8) << col_hist.buckets[i];
            }
            std::cout << std::endl;
#endif

#ifdef SLOW_GREEDY
            cycles_count_start();
            mnp.minimize(slow_problem, wA, eps, tol);
            double slow_seconds = (double) cycles_count_stop().time;
#endif

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << seconds;
#ifdef SLOW_GREEDY
            std::cout << std::setw(2*fw) << slow_seconds;
#endif
            std::cout << std::setw(2*fw) << PerfLog::get().get_count("MAJOR TIME");
#ifdef SLOW_GREEDY
            std::cout << std::setw(2*fw) << PerfLog::get().get_count("MAJOR TIME");
#endif
            std::cout << std::setw(2*fw) << PerfLog::get().get_count("MINOR TIME");

            double total = 0.0;
            for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE TIME", "VECTOR TIME", "GREEDY TIME"}) {
                double percent = 100 * (double) PerfLog::get().get_total(p) / cycles;
                total += percent;
                std::cout << std::setw(2*fw) << percent;
            }
            std::cout << std::setw(2*fw) << total;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("MVM BYTES")) / ((double) PerfLog::get().get_total("MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("TRSV BYTES")) / ((double) PerfLog::get().get_total("TRSV TIME"));
            if(PerfLog::get().get_total("REMOVE COLS QR TIME") > 0) {
                std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("REMOVE COLS QR BYTES")) / ((double) PerfLog::get().get_total("REMOVE COLS QR TIME"));
            }
            else {
                std::cout << std::setw(2*fw) << 0;
            }
            std::cout << std::endl;
        }
    }
}


template<class DT>
void benchmark_iwata(DT eps, DT tol)
{
    int64_t start = 1000;
    int64_t end = 100000;
    int64_t inc = 1000;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking Iwata's test function" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 8;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "|A|"; 
    std::cout << std::setw(2*fw) << "seconds";
    std::cout << std::setw(2*fw) <<  "major";
    std::cout << std::setw(2*fw) <<  "minor";
    std::cout << std::setw(2*fw) <<  "add col %";
    std::cout << std::setw(2*fw) <<  "del col %";
    std::cout << std::setw(2*fw) <<  "del col qr %";
    std::cout << std::setw(2*fw) <<  "solve %";
    std::cout << std::setw(2*fw) <<  "vector %";
    std::cout << std::setw(2*fw) <<  "greedy %";
    std::cout << std::setw(2*fw) <<  "total %";
    std::cout << std::setw(2*fw) <<  "MVM MB/S";
    std::cout << std::setw(2*fw) <<  "TRSV MB/S";
    std::cout << std::setw(2*fw) <<  "del cols MB/S";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            int64_t max_iter = 1e5;

            //Initialize min norm point problem
            IwataTest<DT> problem(n);

            //Initial condition    
            Vector<DT> wA(n);
            wA.fill_rand();
            
            //Time problem
            cycles_count_start();
            auto A = mnp(problem, wA, eps, tol);

            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

#ifdef PRINT_HIST
            std::cout << "Num columns" << std::endl;
            PerfLog::get().print_hist("NUM COLUMNS");
            std::cout << std::endl;
            std::cout << "Columns removed" << std::endl;
            PerfLog::get().print_hist("COLUMNS REMOVED");
#endif

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << seconds;
            std::cout << std::setw(2*fw) << PerfLog::get().get_count("MAJOR TIME");
            std::cout << std::setw(2*fw) << PerfLog::get().get_count("MINOR TIME");
            double total = 0.0;
     //       for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE TIME", "VECTOR TIME", "GREEDY TIME"}) {
            for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE1 TIME", "SOLVE2 TIME" "VECTOR TIME", "GREEDY TIME"}) {
                double percent = 100 * (double) PerfLog::get().get_total(p) / cycles;
                total += percent;
                std::cout << std::setw(2*fw) << percent;
            }
            std::cout << std::setw(2*fw) << total;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("ADD COL MVM BYTES")) / ((double) PerfLog::get().get_total("ADD COL MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("ADD COL TRSV BYTES")) / ((double) PerfLog::get().get_total("ADD COL TRSV TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE1 MVM BYTES")) / ((double) PerfLog::get().get_total("SOLVE1 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE1 TRSV1 BYTES")) / ((double) PerfLog::get().get_total("SOLVE1 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE1 TRSV2 BYTES")) / ((double) PerfLog::get().get_total("SOLVE1 TRSV2 TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE2 MVM BYTES")) / ((double) PerfLog::get().get_total("SOLVE2 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE2 TRSV1 BYTES")) / ((double) PerfLog::get().get_total("SOLVE2 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE2 TRSV2 BYTES")) / ((double) PerfLog::get().get_total("SOLVE2 TRSV2 TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("MVM BYTES")) / ((double) PerfLog::get().get_total("MVM TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("TRSV BYTES")) / ((double) PerfLog::get().get_total("TRSV TIME"));
            if(PerfLog::get().get_total("REMOVE COLS QR TIME") > 0) {
                std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("REMOVE COLS QR BYTES")) / ((double) PerfLog::get().get_total("REMOVE COLS QR TIME"));
            }
            else {
                std::cout << std::setw(2*fw) << 0;
            }
            std::cout << std::endl;
        }
    }
}

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

    int fw = 8;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(2*fw) << "MNP T";
    std::cout << std::setw(2*fw) << "FrankWolfe T";
    std::cout << std::setw(2*fw) << "AwaySteps T";
    std::cout << std::setw(2*fw) << "Pairwise T";
    std::cout << std::setw(2*fw) << "MNP N";
    std::cout << std::setw(2*fw) << "FrankWolfe N";
    std::cout << std::setw(2*fw) << "AwaySteps N";
    std::cout << std::setw(2*fw) << "Pairwise N";
    std::cout << std::setw(2*fw) << "MNP |S|";
    std::cout << std::setw(2*fw) << "AS |S|";
    std::cout << std::setw(2*fw) << "PW |S|";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            int64_t max_iter = 1e6;

            //Initialize min norm point problem
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-5);
            double mnp_fa = problem.eval(mnp_A);
            double cycles = (double) cycles_count_stop().cycles;
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
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(2*fw) << mnp_seconds;
            std::cout << std::setw(2*fw) << fw_seconds;
            std::cout << std::setw(2*fw) << as_seconds;
            std::cout << std::setw(2*fw) << pw_seconds;
            std::cout << std::setw(2*fw) << mnp_iterations;
            std::cout << std::setw(2*fw) << fw_iterations;
            std::cout << std::setw(2*fw) << as_iterations;
            std::cout << std::setw(2*fw) << pw_iterations;
            std::cout << std::setw(2*fw) << (double) mnp_s_card / (double) mnp_iterations;
            std::cout << std::setw(2*fw) << (double) as_s_card / (double) as_iterations;
            std::cout << std::setw(2*fw) << (double) pw_s_card / (double) pw_iterations;
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

    int fw = 8;
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
            int64_t max_iter = 1e6;

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
            double cycles = (double) cycles_count_stop().cycles;
            double mnp_seconds = (double) cycles_count_stop().time;
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_s_card = PerfLog::get().get_total("S WIDTH");

            //BVH
            PerfLog::get().clear();
            cycles_count_start();
            auto bvh_A = bvh(problem, 1e-5, 1e-10);
            double bvh_fa = problem.eval(bvh_A);
            cycles = (double) cycles_count_stop().cycles;
            double bvh_seconds = (double) cycles_count_stop().time;
            int64_t bvh_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t bvh_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t bvh_s_card = PerfLog::get().get_total("S WIDTH");
            int64_t bvh_recompute_dr = PerfLog::get().get_total("RECOMPUTE DR");

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

    int fw = 8;
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
            int64_t max_iter = 1e6;

            //Initialize min norm point problem
            //LogDet<DT> problem(n);
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-10, 1e-10);
            double mnp_fa = problem.eval(mnp_A);
            double cycles = (double) cycles_count_stop().cycles;
            double mnp_seconds = (double) cycles_count_stop().time;
            int64_t mnp_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_s_card = PerfLog::get().get_total("S WIDTH");

            //MNP
            PerfLog::get().clear();
            cycles_count_start();
            auto mnp_speculate_A = mnp_order_k(problem, 1e-10, 1e-10);
            double mnp_speculate_fa = problem.eval(mnp_speculate_A);

            cycles = (double) cycles_count_stop().cycles;
            double mnp_speculate_seconds = (double) cycles_count_stop().time;
            int64_t mnp_speculate_iterations = PerfLog::get().get_total("ITERATIONS");
            int64_t mnp_speculate_minor_cycles = PerfLog::get().get_total("MINOR CYCLES");
            int64_t mnp_speculate_s_card = PerfLog::get().get_total("S WIDTH");

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
    int fw = 11;

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
    for(int i = 0; i < times.size(); i++) max_len = std::max(max_len, times[i].size());
    for(int j = 0; j < max_len; j++) {
        for(int i = 0; i < times.size(); i++) {
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

    int fw = 8;
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
            std::cout << std::setw(fw) << n;
            int64_t max_iter = 1e6;

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
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << fw_fa;
            std::cout << std::setw(2*fw) << mnp_fa;
            std::cout << std::setw(2*fw) << fw_seconds;
            std::cout << std::setw(2*fw) << mnp_seconds;
            std::cout << std::endl;
        }
    }
}

template<class DT>
void test_greedy_maximize()
{
    int64_t start = 8;
    int64_t end = 2048;
    int64_t inc = 8;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking greedy maximization" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 8;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "|A1|"; 
    std::cout << std::setw(fw) << "|A2|"; 
    std::cout << std::setw(2*fw) << "maximize 1 F(A)"; 
    std::cout << std::setw(2*fw) << "maximize 2 F(A)"; 
    std::cout << std::setw(2*fw) << "maximize 1";
    std::cout << std::setw(2*fw) << "maximize 2";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            LogDet<DT> problem(n);

            //unfused
            cycles_count_start();
            auto A1 = problem.greedy_maximize1();
            double seconds1 = (double) cycles_count_stop().time;
            double fA1 = problem.eval(A1);
            
            //fused
            cycles_count_start();
            auto A2 = problem.greedy_maximize2();
            double seconds2 = (double) cycles_count_stop().time;
            double fA2 = problem.eval(A2);

            int64_t cardinality = 0;
            int64_t cardinality2 = 0;
            for(int i = 0; i < n; i++) {
                if(A1[i]) cardinality++;
                if(A2[i]) cardinality2++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(fw) << cardinality2;
            std::cout << std::setw(2*fw) << fA1;
            std::cout << std::setw(2*fw) << fA2;
            std::cout << std::setw(2*fw) << seconds1;
            std::cout << std::setw(2*fw) << seconds2;
            std::cout << std::endl;
            if(std::abs(fA1 - fA2) > 1e-10)
                exit(1);
        }
    }
}

int main() 
{
    run_validation_suite();
    mnp_bvh<double>();
    exit(1);

    //Test Simplicical Decomposition
    mnp_bvh<double>();
    exit(1);

    //Error vs time experiments
    frank_wolfe_mincut_err_vs_time<double>();
    exit(1);

    //Compare with Fujishige's implementation
    test_versus_fujishige();
    exit(1);
    
    mkl_free_buffers(); 
    exit(1);
    frank_wolfe_wolfe_mincut<double>();

    test_greedy_maximize<double>();
    run_benchmark_suite();

    benchmark_logdet<double>(1e-10, 1e-10);
    benchmark_iwata<double>(1e-10, 1e-10);
}
