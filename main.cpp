#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include <random>
#include "submodular.h"
#include "minimizers/mnp.h"
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
void benchmark_mincut(DT eps, DT tol)
{
    int64_t start = 500;
    int64_t end = 50000;
    int64_t inc = 2;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
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
    std::cout << std::setw(2*fw) <<  "solve1 %";
    std::cout << std::setw(2*fw) <<  "solve2 %";
    std::cout << std::setw(2*fw) <<  "vector %";
    std::cout << std::setw(2*fw) <<  "greedy %";
    std::cout << std::setw(2*fw) <<  "total %";
    std::cout << std::setw(2*fw) <<  "add MVM MF/S";
    std::cout << std::setw(2*fw) <<  "add TRSV MF/S";
    std::cout << std::setw(2*fw) <<  "s1 MVM MF/S";
    std::cout << std::setw(2*fw) <<  "s1 TRSV1 MF/S";
    std::cout << std::setw(2*fw) <<  "s1 TRSV2 MF/S";
    std::cout << std::setw(2*fw) <<  "s2 MVM MF/S";
    std::cout << std::setw(2*fw) <<  "s2 TRSV1 MF/S";
    std::cout << std::setw(2*fw) <<  "s2 TRSV2 MF/S";
    std::cout << std::setw(2*fw) <<  "del cols MB/S";
    std::cout << std::setw(2*fw) <<  "edmonds MB/s";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i *= inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            int64_t max_iter = 1e6;

            //Initialize min norm point problem
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);
//            problem.Geometric(.05);
#ifdef SLOW_GREEDY
            SlowMinCut<DT> slow_problem(problem);
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
            std::cout << "Num columns" << std::endl;
            PerfLog::get().print_hist("NUM COLUMNS");
            std::cout << std::endl;
            std::cout << "Columns removed" << std::endl;
            PerfLog::get().print_hist("COLUMNS REMOVED");
#endif

#ifdef SLOW_GREEDY
            cycles_count_start();
            mnp(slow_problem, wA, eps, tol);
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
            //for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE TIME", "VECTOR TIME", "GREEDY TIME"}) {
            for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE1 TIME", "SOLVE2 TIME", "VECTOR TIME", "GREEDY TIME"}) {
                double percent = 100 * (double) PerfLog::get().get_total(p) / cycles;
                total += percent;
                std::cout << std::setw(2*fw) << percent;
            }
            std::cout << std::setw(2*fw) << total;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("ADD COL MVM FLOPS")) / ((double) PerfLog::get().get_total("ADD COL MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("ADD COL TRSV FLOPS")) / ((double) PerfLog::get().get_total("ADD COL TRSV TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE1 MVM FLOPS")) / ((double) PerfLog::get().get_total("SOLVE1 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE1 TRSV1 FLOPS")) / ((double) PerfLog::get().get_total("SOLVE1 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE1 TRSV2 FLOPS")) / ((double) PerfLog::get().get_total("SOLVE1 TRSV2 TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE2 MVM FLOPS")) / ((double) PerfLog::get().get_total("SOLVE2 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE2 TRSV1 FLOPS")) / ((double) PerfLog::get().get_total("SOLVE2 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("SOLVE2 TRSV2 FLOPS")) / ((double) PerfLog::get().get_total("SOLVE2 TRSV2 TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("MVM FLOPS")) / ((double) PerfLog::get().get_total("MVM TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("TRSV FLOPS")) / ((double) PerfLog::get().get_total("TRSV TIME"));
            if(PerfLog::get().get_total("REMOVE COLS QR TIME") > 0) {
                std::cout << std::setw(2*fw) << 3.6e3 * ((double) PerfLog::get().get_total("REMOVE COLS QR BYTES")) / ((double) PerfLog::get().get_total("REMOVE COLS QR TIME"));
            }
            else {
                std::cout << std::setw(2*fw) << 0;
            }

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) sizeof(double) * 16 * n * PerfLog::get().get_count("MARGINAL GAIN TIME")) / ((double) PerfLog::get().get_total("MARGINAL GAIN TIME"));

            for(auto p : {  "SOLVE1 MVM TIME", "SOLVE1 TRSV1 TIME", "SOLVE1 TRSV2 TIME"}) {
                double percent = 100 * (double) PerfLog::get().get_total(p) / (double) PerfLog::get().get_total("SOLVE1 TIME");
                std::cout << std::setw(2*fw) << percent;
            }
            for(auto p : {  "SOLVE2 MVM TIME", "SOLVE2 TRSV1 TIME", "SOLVE2 TRSV2 TIME"}) {
                double percent = 100 * (double) PerfLog::get().get_total(p) / (double) PerfLog::get().get_total("SOLVE2 TIME");
                std::cout << std::setw(2*fw) << percent;
            }
            for(auto p : {  "ADD COL MVM TIME", "ADD COL TRSV TIME"} ) {
                double percent = 100 * (double) PerfLog::get().get_total(p) / (double) PerfLog::get().get_total("ADD COL TIME");
                std::cout << std::setw(2*fw) << percent;
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
    int64_t start = 8;
    int64_t end = 2048;
    int64_t inc = 8;
    int64_t n_reps = 200;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 8;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "|A|"; 
    std::cout << std::setw(2*fw) << "MNP F(A)"; 
    std::cout << std::setw(2*fw) << "FrankWolfe F(A)"; 
    std::cout << std::setw(2*fw) << "AwaySteps F(A)"; 
    std::cout << std::setw(2*fw) << "Pairwise F(A)"; 
    std::cout << std::setw(2*fw) << "MNP";
    std::cout << std::setw(2*fw) << "FrankWolfe";
    std::cout << std::setw(2*fw) << "AwaySteps";
    std::cout << std::setw(2*fw) << "Pairwise";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            int64_t max_iter = 1e6;

            //Initialize min norm point problem
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);

            //MNP
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-5, 1e-5);
            double mnp_fa = problem.eval(mnp_A);
            double cycles = (double) cycles_count_stop().cycles;
            double mnp_seconds = (double) cycles_count_stop().time;
            
            //Vanilla FW
            cycles_count_start();
            auto fw_A = FrankWolfe(problem, 1e-5);
            double fw_seconds = (double) cycles_count_stop().time;
            double fw_fa = problem.eval(fw_A);

            //Away Steps FW
            cycles_count_start();
            auto as_A = AwaySteps(problem, 1e-5);
            double as_seconds = (double) cycles_count_stop().time;
            double as_fa = problem.eval(as_A);

            //Pairwise
            cycles_count_start();
            auto pw_A = Pairwise(problem, 1e-5, mnp_fa);
            double pw_seconds = (double) cycles_count_stop().time;
            double pw_fa = problem.eval(pw_A);

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(mnp_A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << mnp_fa;
            std::cout << std::setw(2*fw) << fw_fa;
            std::cout << std::setw(2*fw) << as_fa;
            std::cout << std::setw(2*fw) << pw_fa;
            std::cout << std::setw(2*fw) << mnp_seconds;
            std::cout << std::setw(2*fw) << fw_seconds;
            std::cout << std::setw(2*fw) << as_seconds;
            std::cout << std::setw(2*fw) << pw_seconds;
            std::cout << std::endl;
        }
    }
}

extern int number_extreme_point;

void test_versus_fujishige()
{
    int64_t start = 8;
    int64_t end = 10000;
    int64_t inc = 8;
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
            PerfLog::get().set_total("MAJOR CYCLES", 0);
            cycles_count_start();
            auto mnp_A = mnp(problem, 1e-10, 1e-10);
            double mnp_fa = problem.eval(mnp_A);
            double mnp_seconds = (double) cycles_count_stop().time;
            std::cout << std::setw(2*fw) << PerfLog::get().get_total("MAJOR CYCLES"); 

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
    test_versus_fujishige();
    exit(1);

    frank_wolfe_wolfe_mincut<double>();
    test_greedy_maximize<double>();
    run_benchmark_suite();

    benchmark_mincut<double>(1e-10, 1e-10);
    benchmark_logdet<double>(1e-10, 1e-10);
    benchmark_iwata<double>(1e-10, 1e-10);

}
