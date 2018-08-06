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

void benchmark_max_flow()
{
    int64_t start = 50;
    int64_t end = 5000;
    int64_t inc = 50;
    int64_t n_reps = 10;
    double connectivity = 0.2; 


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
//    std::cout << std::setw(2*fw) << "Major cycles" << std::setw(2*fw) << "Minor Cycles";
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
        

        for(int64_t r = 0; r < n_reps; r++) {
            PerfLog log;

            //Initialize min norm point problem
            MinCut<double> problem(n, 16, 0.5, 0.05);
            
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
/*
            std::cout << "*****************" << std::endl;
            log.print("TIME", cycles);
            std::cout << "*****************" << std::endl;
*/
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
        std::cout << std::endl;
    }
}

int main() {
    run_validation_suite();
    run_benchmark_suite();

    benchmark_max_flow();

    MinNormPoint<double> mnp;

    std::cout << "Min cut problem\n";
    MinCut<double> problem(1000, 0.2);
    mnp.minimize(problem, 1e-10, 1e-5, true, NULL);

    std::cout << "Cardinality problem\n";
    IDivSqrtSize<double> F(500);
    mnp.minimize(F, 1e-10, 1e-10, true, NULL); 
}
