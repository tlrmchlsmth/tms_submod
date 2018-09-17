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
//#define PRINT_HIST

void benchmark_logdet()
{
    int64_t start = 500;
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
            LogDet<double> problem(n);
#ifdef SLOW_GREEDY
            PerfLog slow_log;
            SlowLogDet<double> slow_problem(problem);
#endif
        
            //Initial condition    
            Vector<double> wA(n);
            wA.fill_rand();
            bool done = false;

            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, wA, &done, max_iter, 1e-10, 1e-10, false, &log);
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
            done = false;
            cycles_count_start();
            mnp.minimize(slow_problem, wA, &done, max_iter, 1e-10, 1e-10, false, &slow_log);
            double slow_seconds = (double) cycles_count_stop().time;
#endif

            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << A.size();
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

void benchmark_mincut()
{
    int64_t start = 500;
    int64_t end = 10000;
    int64_t inc = 500;
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
            int64_t max_iter = 1e6;
            PerfLog log;

            //Initialize min norm point problem
            MinCut<double> problem(n);
            problem.WattsStrogatz(16, 0.25);
//            problem.Geometric(.05);
#ifdef SLOW_GREEDY
            PerfLog slow_log;
            SlowMinCut<double> slow_problem(problem);
#endif

            //Initial condition    
            Vector<double> wA(n);
            wA.fill_rand();
            bool done = false;
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, wA, &done, max_iter, 1e-10, 1e-10, false, &log);

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
            done = false;
            cycles_count_start();
            mnp.minimize(slow_problem, wA, &done, max_iter, 1e-10, 1e-10, false, &slow_log);
            double slow_seconds = (double) cycles_count_stop().time;
#endif

            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << A.size();
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

void benchmark_iwata()
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
            IwataTest<double> problem(n);

            //Initial condition    
            Vector<double> wA(n);
            wA.fill_rand();
            bool done = false;
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, wA, &done, max_iter, 1e-10, 1e-10, false, &log);

            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

#ifdef PRINT_HIST
            std::cout << "Num columns" << std::endl;
            log.print_hist("NUM COLUMNS");
            std::cout << std::endl;
            std::cout << "Columns removed" << std::endl;
            log.print_hist("COLUMNS REMOVED");
#endif

            std::cout << std::setw(fw) << n;
            std::cout << std::setw(fw) << A.size();
            std::cout << std::setw(2*fw) << seconds;
            std::cout << std::setw(2*fw) << log.get_count("MAJOR TIME");
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

int main() 
{
    run_validation_suite();
    run_benchmark_suite();

    benchmark_iwata();
    benchmark_logdet();
    benchmark_mincut();


    //benchmark_mnp_vs_brsmnp();



//    BRSMinNormPoint<double> brsmnp(8);
//    MinCut<double> max_flow_problem(10);
//    max_flow_problem.WattsStrogatz(16, 0.25);
//    brsmnp.minimize(max_flow_problem, 1e-10, 1e-10, true, NULL); 


/*
    std::cout << "===========================================================" << std::endl;
    std::cout << "Running some examples" << std::endl;
    std::cout << "===========================================================" << std::endl;

    MinNormPoint<double> mnp;
    std::cout << "Log Det problem\n";
    LogDet<double> logdet_problem(100);
    mnp.minimize(logdet_problem, 1e-10, 1e-10, true, NULL);

    std::cout << "Min cut problem\n";
    MinCut<double> max_flow_problem(1000, 15, 0.5, 0.05);
    mnp.minimize(max_flow_problem, 1e-10, 1e-10, true, NULL);

    std::cout << "Cardinality problem\n";
    IDivSqrtSize<double> F(500);
    mnp.minimize(F, 1e-10, 1e-10, true, NULL); 
*/




}

#if 0
void benchmark_mnp_vs_brsmnp()
{
    int64_t start = 4;
    int64_t end = 64;
    int64_t inc = 4;
    int64_t n_reps = 10;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 10;
    std::cout << std::setw(fw) << "n"; 
    std::cout << std::setw(fw) << "b"; 
    std::cout << std::setw(2*fw) <<  "mean_s_mnp";
    std::cout << std::setw(2*fw) <<  "mean_s_brsmnp";
    //std::cout << std::setw(2*fw) <<  "med_s_mnp";
    //std::cout << std::setw(2*fw) <<  "med_s_brsmnp";
    std::cout << std::setw(2*fw) <<  "major_mnp";
    std::cout << std::setw(2*fw) <<  "major_brsmnp";
    std::cout << std::setw(2*fw) <<  "minor_mnp";
    std::cout << std::setw(2*fw) <<  "minor_brsmnp";
    std::cout << std::setw(2*fw) <<  "F A_mnp";
    std::cout << std::setw(2*fw) <<  "F A_brs_mnp";

//    std::cout << std::setw(2*fw) <<  "BLAS_s_mnp %";
//    std::cout << std::setw(2*fw) <<  "BLAS_s_brsmnp %";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = 100;
        int64_t b = i;

        std::vector<double> cpu_cycles_mnp;
        std::vector<double> major_cycles_mnp;
        std::vector<double> minor_cycles_mnp;
        std::vector<double> blas_s_mnp;

        std::vector<double> cpu_cycles_brsmnp;
        std::vector<double> major_cycles_brsmnp;
        std::vector<double> minor_cycles_brsmnp;
        std::vector<double> blas_s_brsmnp;

        for(int64_t r = 0; r < n_reps; r++) {
            PerfLog mnp_log;
            PerfLog brsmnp_log;

            //Initialize min norm point problem
            MinCut<double> problem(n);
            problem.WattsStrogatz(16, 0.25);
            //problem.Geometric(0.05);
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            auto A_mnp = mnp.minimize(problem, 1e-10, 1e-5, false, &mnp_log);
            double cycles = (double) cycles_count_stop().cycles;
            double mnp_time = cycles_count_stop().time;
            cpu_cycles_mnp.push_back(cycles_count_stop().time);
//            major_cycles_mnp.push_back(mnp_log.get_count("MAJOR TIME"));
//            minor_cycles_mnp.push_back(mnp_log.get_count("MINOR TIME"));
            double FA_mnp = problem.eval(A_mnp);

            //Time problem
            BRSMinNormPoint<double> brsmnp(b);
            cycles_count_start();
            auto A_brsmnp = brsmnp.minimize(problem, 1e-10, 1e-5, false, &brsmnp_log);
            cycles = (double) cycles_count_stop().cycles;
            cpu_cycles_mnp.push_back(cycles_count_stop().time);
            double brsmnp_time = cycles_count_stop().time;
//            major_cycles_brsmnp.push_back(brsmnp_log.get_count("MAJOR TIME"));
//            minor_cycles_brsmnp.push_back(brsmnp_log.get_count("MINOR TIME"));
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
/*
        std::cout << std::setw(fw) << n;
        std::cout << std::setw(2*fw) << mean(cpu_cycles);
        std::cout << std::setw(2*fw) << median(cpu_cycles);
        std::cout << std::setw(2*fw) << mean(major_cycles);
        std::cout << std::setw(2*fw) << mean(minor_cycles);
        std::cout << std::setw(2*fw) << 100 * mean(mvm_percent);
        std::cout << std::setw(2*fw) << 100 * mean(trsv_percent);
        std::cout << std::setw(2*fw) << 100 * mean(remove_cols_percent);
        std::cout << std::setw(2*fw) << 100 * mean(eval_f_percent);
        std::cout << std::setw(2*fw) << 100 * mean(greedy_percent);
        std::cout << std::setw(2*fw) << 3.6e3 * mean(mvm_bw);
        std::cout << std::setw(2*fw) << 3.6e3 * mean(trsv_bw);
        std::cout << std::setw(2*fw) << 3.6e3 * mean(remove_cols_bw);
        std::cout << std::endl;*/
    }
}
#endif
