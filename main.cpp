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

template<class DT>
double time_problem_with_lemon(MinCut<DT>& problem)
{
    ListDigraph g;
    ListDigraph::ArcMap<DT> capacity(g);
    int64_t n = problem.adjacency.height();

    //Add interior nodes.
    std::vector<int> node_ids;
    node_ids.reserve(problem.adjacency.height());
    for(int i = 0; i < n; i++) {
        auto node = g.addNode();
        node_ids.push_back(g.id(node));
    }

    //Add interior edges
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(problem.adjacency(i,j) > 0.0) {
                auto arc = g.addArc(g.nodeFromId(node_ids[i]), g.nodeFromId(node_ids[j]));
                capacity[arc] = problem.adjacency(i,j);
            }
        }
    }

    //Add edges from source
    ListDigraph::Node source = g.addNode();
    for(int i = 0; i < n; i++) {
        if(problem.edges_from_source(i) > 0.0) {
           auto arc = g.addArc(source, g.nodeFromId(node_ids[i]));
            capacity[arc] = problem.edges_from_source(i);
        }
    }

    //Add edges to sink
    ListDigraph::Node sink = g.addNode();
    for(int i = 0; i < n; i++) {
        if(problem.edges_to_sink(i) > 0.0) {
            auto arc = g.addArc(g.nodeFromId(node_ids[i]), sink);
            capacity[arc] = problem.edges_to_sink(i);
        }
    }

    Preflow<ListDigraph, ListDigraph::ArcMap<double>> lemon_prob(g, capacity, source, sink);
    //EdmondsKarp<ListDigraph, ListDigraph::ArcMap<double>> lemon_prob(g, capacity, source, sink);
    cycles_count_start();
    lemon_prob.run();
    auto elapsed = cycles_count_stop().time;
    //std::cout << "lemon prob result " << lemon_prob.flowValue() - problem.baseline << std::endl;
    return elapsed;
}

void benchmark_log_det()
{
    int64_t start = 500;
    int64_t end = 4000;
    int64_t inc = 500;
    int64_t n_reps = 5;


    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking log det" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 8;
    std::cout << std::setw(fw) << "n";
    std::cout << std::setw(2*fw) << "mean |A|";
    std::cout << std::setw(2*fw) << "seconds";
    std::cout << std::setw(2*fw) << "slowseconds";
    std::cout << std::setw(2*fw) << "major cycles";
    std::cout << std::setw(2*fw) << "slowmajor cycles";
    std::cout << std::setw(2*fw) << "minor cycles";
    std::cout << std::setw(2*fw) << "mvm %";
    std::cout << std::setw(2*fw) << "trsv %";
    std::cout << std::setw(2*fw) << "remove cols %";
    std::cout << std::setw(2*fw) <<  "eval f %";
    std::cout << std::setw(2*fw) <<  "greedy %";
    std::cout << std::setw(2*fw) <<  "MVM MB/S";
    std::cout << std::setw(2*fw) <<  "TRSV MB/S";
    std::cout << std::setw(2*fw) <<  "Remove cols MB/S";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        for(int64_t r = 0; r < n_reps; r++) {
            PerfLog log;

            //Initialize min norm point problem
            LogDet<double> problem(n);

            PerfLog slow_log;
            SlowLogDet<double> slow_problem(n, problem.Cov);
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            auto A = mnp.minimize(problem, 1e-10, 1e-5, false, &log);
            double cycles = (double) cycles_count_stop().cycles;
            double seconds = (double) cycles_count_stop().time;

            cycles_count_start();
            mnp.minimize(slow_problem, 1e-10, 1e-5, false, &slow_log);
            double slow_seconds = (double) cycles_count_stop().time;

            std::cout << std::setw(fw) << n;
            std::cout << std::setw(2*fw) << A.size();
            std::cout << std::setw(2*fw) << seconds;
            std::cout << std::setw(2*fw) << slow_seconds;
            std::cout << std::setw(2*fw) << log.get_count("MAJOR TIME");
            std::cout << std::setw(2*fw) << slow_log.get_count("MAJOR TIME");
            std::cout << std::setw(2*fw) << log.get_count("MINOR TIME");
            std::cout << std::setw(2*fw) << 100 * (double) log.get_total("MVM TIME") / cycles;
            std::cout << std::setw(2*fw) << 100 * (double) log.get_total("TRSV TIME") / cycles;
            std::cout << std::setw(2*fw) << 100 * (double) log.get_total("REMOVE COLS QR TIME") / cycles;
            std::cout << std::setw(2*fw) << 100 * (double) log.get_total("EVAL F TIME") / cycles;
            std::cout << std::setw(2*fw) << 100 * (double) log.get_total("GREEDY TIME") / cycles;

            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("MVM BYTES")) / ((double) log.get_total("MVM TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("TRSV BYTES")) / ((double) log.get_total("TRSV TIME"));
            std::cout << std::setw(2*fw) << 3.6e3 * ((double) log.get_total("REMOVE COLS QR BYTES")) / ((double) log.get_total("REMOVE COLS QR TIME"));
            std::cout << std::endl;
        }
    }
}

void benchmark_min_cut()
{
    int64_t start = 100;
    int64_t end = 1000;
    int64_t inc = 100;
    int64_t n_reps = 5;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking min cut" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 10;
    std::cout << std::setw(fw) << "n" << std::setw(2*fw) << "mean (s)" << std::setw(2*fw) << "median (s)";
    std::cout << std::setw(2*fw) <<  "major cycles";
    std::cout << std::setw(2*fw) <<  "minor cycles";
    std::cout << std::setw(2*fw) <<  "mvm %";
    std::cout << std::setw(2*fw) <<  "trsv %";
    std::cout << std::setw(2*fw) <<  "remove cols %";
    std::cout << std::setw(2*fw) <<  "eval f %";
    std::cout << std::setw(2*fw) <<  "greedy %";
    std::cout << std::setw(2*fw) <<  "MVM MB/S";
    std::cout << std::setw(2*fw) <<  "TRSV MB/S";
    std::cout << std::setw(2*fw) <<  "Remove cols MB/S";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        std::vector<double> cpu_cycles;
        std::vector<double> major_cycles;
        std::vector<double> minor_cycles;

        std::vector<double> mvm_percent;
        std::vector<double> trsv_percent;
        std::vector<double> remove_cols_percent;
        std::vector<double> eval_f_percent;
        std::vector<double> greedy_percent;

        std::vector<double> mvm_bw;
        std::vector<double> trsv_bw;
        std::vector<double> remove_cols_bw;
        

        for(int64_t r = 0; r < n_reps; r++) {
            PerfLog log;

            //Initialize min norm point problem
            MinCut<double> problem(n);
            problem.WattsStrogatz(16, 0.25);
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            mnp.minimize(problem, 1e-10, 1e-6, false, &log);
            double cycles = (double) cycles_count_stop().cycles;
            cpu_cycles.push_back(cycles_count_stop().time);

            major_cycles.push_back(log.get_count("MAJOR TIME"));
            minor_cycles.push_back(log.get_count("MINOR TIME"));
            
            mvm_percent.push_back((double) log.get_total("MVM TIME") / cycles);
            trsv_percent.push_back((double) log.get_total("TRSV TIME") / cycles);
            remove_cols_percent.push_back((double) log.get_total("REMOVE COLS QR TIME") / cycles);
            eval_f_percent.push_back((double) log.get_total("EVAL F TIME") / cycles);
            greedy_percent.push_back((double) log.get_total("GREEDY TIME") / cycles);

            mvm_bw.push_back(((double) log.get_total("MVM BYTES")) / ((double) log.get_total("MVM TIME")));
            trsv_bw.push_back(((double) log.get_total("TRSV BYTES")) / ((double) log.get_total("TRSV TIME")));
            if(log.get_total("REMOVE COLS QR TIME") > 0)
                remove_cols_bw.push_back(((double) log.get_total("REMOVE COLS QR BYTES")) / ((double) log.get_total("REMOVE COLS QR TIME")));
        }

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
        std::cout << std::endl;
    }
}

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

int main() 
{
    run_validation_suite();
    run_benchmark_suite();
    benchmark_log_det();

    benchmark_min_cut();
    benchmark_mnp_vs_brsmnp();



//    BRSMinNormPoint<double> brsmnp(8);
//    MinCut<double> max_flow_problem(10);
//    max_flow_problem.WattsStrogatz(16, 0.25);
//    brsmnp.minimize(max_flow_problem, 1e-10, 1e-5, true, NULL); 



    std::cout << "===========================================================" << std::endl;
    std::cout << "Running some examples" << std::endl;
    std::cout << "===========================================================" << std::endl;
/*
    MinNormPoint<double> mnp;
    std::cout << "Log Det problem\n";
    LogDet<double> logdet_problem(100);
    mnp.minimize(logdet_problem, 1e-10, 1e-5, true, NULL);

    std::cout << "Min cut problem\n";
    MinCut<double> max_flow_problem(1000, 15, 0.5, 0.05);
    mnp.minimize(max_flow_problem, 1e-10, 1e-5, true, NULL);

    std::cout << "Cardinality problem\n";
    IDivSqrtSize<double> F(500);
    mnp.minimize(F, 1e-10, 1e-10, true, NULL); 
*/




}
