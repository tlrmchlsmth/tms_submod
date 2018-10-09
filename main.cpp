#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include <random>
#include "submodular.h"
#include "minimizer.h"
#include "perf/perf.h"
#include "test/validate.h"
#include "test/bench.h"

#include "vector.h"
#include "matrix.h"
#include "util.h"

//#define SLOW_GREEDY
#define PRINT_HIST

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
            PerfLog log;

            //Initialize min norm point problem
            LogDet<DT> problem(n);
#ifdef SLOW_GREEDY
            PerfLog slow_log;
            SlowLogDet<DT> slow_problem(problem);
#endif
        
            //Initial condition    
            Vector<DT> wA(n);
            wA.fill_rand();
            bool done;

            //Time problem
            MinNormPoint<DT> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, wA, &done, max_iter, eps, tol, false, &log);
            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

#ifdef PRINT_HIST
            auto col_hist = log.get_hist("NUM COLUMNS");
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
            mnp.minimize(slow_problem, wA, &done, max_iter, eps, tol, false, &slow_log);
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
            std::cout << std::setw(2*fw) << log.get_count("MAJOR TIME");
#ifdef SLOW_GREEDY
            std::cout << std::setw(2*fw) << slow_log.get_count("MAJOR TIME");
#endif
            std::cout << std::setw(2*fw) << log.get_count("MINOR TIME");

            double total = 0.0;
            for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE TIME", "VECTOR TIME", "GREEDY TIME"}) {
                double percent = 100 * (double) log.get_total(p) / cycles;
                total += percent;
                std::cout << std::setw(2*fw) << percent;
            }
            std::cout << std::setw(2*fw) << total;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("MVM BYTES")) / ((double) log.get_total("MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("TRSV BYTES")) / ((double) log.get_total("TRSV TIME"));
            if(log.get_total("REMOVE COLS QR TIME") > 0) {
                std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("REMOVE COLS QR BYTES")) / ((double) log.get_total("REMOVE COLS QR TIME"));
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
            PerfLog log;

            //Initialize min norm point problem
            MinCut<DT> problem(n);
            problem.WattsStrogatz(16, 0.25);
//            problem.Geometric(.05);
#ifdef SLOW_GREEDY
            PerfLog slow_log;
            SlowMinCut<DT> slow_problem(problem);
#endif

            //Initial condition    
            Vector<DT> wA(n);
            wA.fill_rand();
            bool done;
            
            //Time problem
            MinNormPoint<DT> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, wA, &done, max_iter, eps, tol, false, &log);

            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

#ifdef PRINT_HIST
            std::cout << "Num columns" << std::endl;
            log.print_hist("NUM COLUMNS");
            std::cout << std::endl;
            std::cout << "Columns removed" << std::endl;
            log.print_hist("COLUMNS REMOVED");
#endif

#ifdef SLOW_GREEDY
            cycles_count_start();
            mnp.minimize(slow_problem, wA, &done, max_iter, eps, tol, false, &slow_log);
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
            std::cout << std::setw(2*fw) << log.get_count("MAJOR TIME");
#ifdef SLOW_GREEDY
            std::cout << std::setw(2*fw) << slow_log.get_count("MAJOR TIME");
#endif
            std::cout << std::setw(2*fw) << log.get_count("MINOR TIME");
            double total = 0.0;
            //for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE TIME", "VECTOR TIME", "GREEDY TIME"}) {
            for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE1 TIME", "SOLVE2 TIME", "VECTOR TIME", "GREEDY TIME"}) {
                double percent = 100 * (double) log.get_total(p) / cycles;
                total += percent;
                std::cout << std::setw(2*fw) << percent;
            }
            std::cout << std::setw(2*fw) << total;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("ADD COL MVM FLOPS")) / ((double) log.get_total("ADD COL MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("ADD COL TRSV FLOPS")) / ((double) log.get_total("ADD COL TRSV TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE1 MVM FLOPS")) / ((double) log.get_total("SOLVE1 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE1 TRSV1 FLOPS")) / ((double) log.get_total("SOLVE1 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE1 TRSV2 FLOPS")) / ((double) log.get_total("SOLVE1 TRSV2 TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE2 MVM FLOPS")) / ((double) log.get_total("SOLVE2 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE2 TRSV1 FLOPS")) / ((double) log.get_total("SOLVE2 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE2 TRSV2 FLOPS")) / ((double) log.get_total("SOLVE2 TRSV2 TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("MVM FLOPS")) / ((double) log.get_total("MVM TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("TRSV FLOPS")) / ((double) log.get_total("TRSV TIME"));
            if(log.get_total("REMOVE COLS QR TIME") > 0) {
                std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("REMOVE COLS QR BYTES")) / ((double) log.get_total("REMOVE COLS QR TIME"));
            }
            else {
                std::cout << std::setw(2*fw) << 0;
            }

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) sizeof(double) * 16 * n * log.get_count("MARGINAL GAIN TIME")) / ((double) log.get_total("MARGINAL GAIN TIME"));

            for(auto p : {  "SOLVE1 MVM TIME", "SOLVE1 TRSV1 TIME", "SOLVE1 TRSV2 TIME"}) {
                double percent = 100 * (double) log.get_total(p) / (double) log.get_total("SOLVE1 TIME");
                std::cout << std::setw(2*fw) << percent;
            }
            for(auto p : {  "SOLVE2 MVM TIME", "SOLVE2 TRSV1 TIME", "SOLVE2 TRSV2 TIME"}) {
                double percent = 100 * (double) log.get_total(p) / (double) log.get_total("SOLVE2 TIME");
                std::cout << std::setw(2*fw) << percent;
            }
            for(auto p : {  "ADD COL MVM TIME", "ADD COL TRSV TIME"} ) {
                double percent = 100 * (double) log.get_total(p) / (double) log.get_total("ADD COL TIME");
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
            PerfLog log;

            //Initialize min norm point problem
            IwataTest<DT> problem(n);

            //Initial condition    
            Vector<DT> wA(n);
            wA.fill_rand();
            bool done;
            
            //Time problem
            MinNormPoint<DT> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, wA, &done, max_iter, eps, tol, false, &log);

            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

#ifdef PRINT_HIST
            std::cout << "Num columns" << std::endl;
            log.print_hist("NUM COLUMNS");
            std::cout << std::endl;
            std::cout << "Columns removed" << std::endl;
            log.print_hist("COLUMNS REMOVED");
#endif

            int64_t cardinality = 0;
            for(int i = 0; i < n; i++) {
                if(A[i]) cardinality++;
            }
            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << cardinality;
            std::cout << std::setw(2*fw) << seconds;
            std::cout << std::setw(2*fw) << log.get_count("MAJOR TIME");
            std::cout << std::setw(2*fw) << log.get_count("MINOR TIME");
            double total = 0.0;
     //       for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE TIME", "VECTOR TIME", "GREEDY TIME"}) {
            for(auto p : { "ADD COL TIME", "REMOVE COLS TIME", "REMOVE COLS QR TIME", "SOLVE1 TIME", "SOLVE2 TIME" "VECTOR TIME", "GREEDY TIME"}) {
                double percent = 100 * (double) log.get_total(p) / cycles;
                total += percent;
                std::cout << std::setw(2*fw) << percent;
            }
            std::cout << std::setw(2*fw) << total;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("ADD COL MVM BYTES")) / ((double) log.get_total("ADD COL MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("ADD COL TRSV BYTES")) / ((double) log.get_total("ADD COL TRSV TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE1 MVM BYTES")) / ((double) log.get_total("SOLVE1 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE1 TRSV1 BYTES")) / ((double) log.get_total("SOLVE1 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE1 TRSV2 BYTES")) / ((double) log.get_total("SOLVE1 TRSV2 TIME"));

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE2 MVM BYTES")) / ((double) log.get_total("SOLVE2 MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE2 TRSV1 BYTES")) / ((double) log.get_total("SOLVE2 TRSV1 TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("SOLVE2 TRSV2 BYTES")) / ((double) log.get_total("SOLVE2 TRSV2 TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("MVM BYTES")) / ((double) log.get_total("MVM TIME"));
//            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("TRSV BYTES")) / ((double) log.get_total("TRSV TIME"));
            if(log.get_total("REMOVE COLS QR TIME") > 0) {
                std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("REMOVE COLS QR BYTES")) / ((double) log.get_total("REMOVE COLS QR TIME"));
            }
            else {
                std::cout << std::setw(2*fw) << 0;
            }
            std::cout << std::endl;
        }
    }
}


void benchmark_mnp_vs_brsmnp()
{
    int64_t start = 100;
    int64_t end = 10000;
    int64_t inc = 100;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 10;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "b"; 
    std::cout << std::setw(2*fw) <<  "mean_s_mnp";
    std::cout << std::setw(2*fw) <<  "mean_s_brsmnp";
    std::cout << std::setw(2*fw) <<  "major_mnp";
    std::cout << std::setw(2*fw) <<  "major_brsmnp";
    std::cout << std::setw(2*fw) <<  "minor_mnp";
    std::cout << std::setw(2*fw) <<  "minor_brsmnp";
    std::cout << std::setw(2*fw) <<  "F A_mnp";
    std::cout << std::setw(2*fw) <<  "F A_brs_mnp";

    std::cout << std::endl;

    int64_t max_iter = 1e5;
    double tolerance = 1e-10;
    double eps = 1e-5;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;
        int64_t b = 512;

        for(int64_t r = 0; r < n_reps; r++) {
            PerfLog mnp_log;
            PerfLog brsmnp_log;

            //Initialize min norm point problem
            MinCut<double> problem(n);
            problem.WattsStrogatz(16, 0.25);
            
            //Initial Condidion
            Vector<double> wA(n);
            wA.fill_rand();
            bool done;
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();

            auto A_mnp = mnp.minimize(problem, wA, &done, max_iter, eps, tolerance, false, &mnp_log);
            double cycles = (double) cycles_count_stop().cycles;
            double mnp_time = cycles_count_stop().time;
            double FA_mnp = problem.eval(A_mnp);

            //Time problem
            BRSMinNormPoint<double> brsmnp(b);
            cycles_count_start();
            auto A_brsmnp = brsmnp.minimize(problem, wA, &done, max_iter, eps, tolerance, false, &brsmnp_log);
            cycles = (double) cycles_count_stop().cycles;
            double brsmnp_time = cycles_count_stop().time;
            double FA_brsmnp = problem.eval(A_brsmnp);

            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << b;
            std::cout << std::setw(2*fw) << mnp_time;
            std::cout << std::setw(2*fw) << brsmnp_time;
            std::cout << std::setw(2*fw) << mnp_log.get_count("MAJOR TIME");
            std::cout << std::setw(2*fw) << brsmnp_log.get_count("MAJOR TIME");
            std::cout << std::setw(2*fw) << mnp_log.get_count("MINOR TIME");
            std::cout << std::setw(2*fw) << brsmnp_log.get_count("MINOR TIME");
            std::cout << std::setw(2*fw) << FA_mnp;
            std::cout << std::setw(2*fw) << FA_brsmnp;
            std::cout << std::endl;
        }
    }
}

int main() 
{
    run_validation_suite();

//    benchmark_mincut<double>(1e-10, 1e-10);
    benchmark_mnp_vs_brsmnp();
    exit(1);
    benchmark_logdet<double>(1e-10, 1e-10);
    benchmark_iwata<double>(1e-10, 1e-10);
/*
    benchmark_mincut<float>(1e-5, 1e-5);
    benchmark_iwata<float>(1e-5, 1e-5);
    benchmark_logdet<float>(1e-5, 1e-5);
*/
    run_benchmark_suite();
}
