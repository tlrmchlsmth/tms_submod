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

#include <lemon/list_graph.h>
#include <lemon/preflow.h>
#include <lemon/edmonds_karp.h>
using namespace lemon;

template<class DT>
DT mean(std::vector<DT> v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
template<class DT>
DT stdev(std::vector<DT> v) {
    DT mu = mean(v);
    std::vector<DT> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mu](DT x) { return x - mu; });
    DT sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    DT stdev = std::sqrt(sq_sum / v.size());
    return stdev;
}
template<class DT>
DT median(std::vector<DT> v) {
    std::sort(v.begin(), v.end());
    DT median = v[v.size()/2];
    if(v.size() % 2 == 0) {
        median = (median + v[v.size()/2 - 1]) / 2.0;
    }
    return median;
}

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
    int64_t start = 16;
    int64_t end = 512;
    int64_t inc = 16;
    int64_t n_reps = 10;
    double connectivity = 0.2; 

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

    int fw = 10;
    std::cout << std::setw(fw) << "n" << std::setw(2*fw) << "mean (s)" << std::setw(2*fw) << "median (s)" << std::setw(2*fw) <<  "stdev (s)";
    std::cout << std::setw(2*fw) << "Major cycles" << std::setw(2*fw) << "Minor Cycles";
    std::cout << std::setw(2*fw) << "Lemon cycles";
    std::cout << std::endl;

    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        std::uniform_int_distribution<> dist(0,n-1);
        std::vector<double> cpu_cycles;
        std::vector<double> major_cycles;
        std::vector<double> minor_cycles;
        std::vector<double> lemon_cpu_cycles;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize min norm point problem
            MinCut<double> problem(n, 16, 0.5, 0.05);
            
            //Time problem
            MinNormPoint<double> mnp;
            cycles_count_start();
            mnp.minimize(problem, 1e-10, 1e-6, false);
            cpu_cycles.push_back(cycles_count_stop().time);
            major_cycles.push_back(mnp.major_cycles);
            minor_cycles.push_back(mnp.minor_cycles);

            //Initialize Preflow problem 
            lemon_cpu_cycles.push_back(time_problem_with_lemon(problem));
        }

        std::cout << std::setw(fw) << n;
        std::cout << std::setw(2*fw) << mean(cpu_cycles);
        std::cout << std::setw(2*fw) << median(cpu_cycles);
        std::cout << std::setw(2*fw) << stdev(cpu_cycles);
        std::cout << std::setw(2*fw) << median(major_cycles);
        std::cout << std::setw(2*fw) << median(minor_cycles);
        std::cout << std::setw(2*fw) << mean(lemon_cpu_cycles);
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
    mnp.minimize(problem, 1e-10, 1e-5, true);

    std::cout << "Cardinality problem\n";
    IDivSqrtSize<double> F(500);
    mnp.minimize(F, 1e-10, 1e-10, true); 
}
