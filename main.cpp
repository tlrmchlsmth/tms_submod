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

void benchmark_max_flow()
{
    int64_t start = 128;
    int64_t end = 1024;
    int64_t inc = 128;
    int64_t n_reps = 10;
    double connectivity = 0.2; 

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

    int fw = 10;
    std::cout << std::setw(fw) << "n" << std::setw(2*fw) << "mean (s)" << std::setw(2*fw) << "median (s)" << std::setw(2*fw) <<  "stdev (s)" <<  std::setw(fw) << "Major cycles" << std::setw(fw) << "Minor Cycles" << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        std::uniform_int_distribution<> dist(0,n-1);
        std::vector<double> cpu_cycles;
        std::vector<double> major_cycles;
        std::vector<double> minor_cycles;

        for(int64_t r = 0; r < n_reps; r++) {
            //Initialize problem
            //MinCut problem(n, connectivity);
            MinCut problem(n, 16, 0.5, 0.05);
            
            //Time problem
            cycles_count_start();
            MinNormPoint<double> mnp;
            mnp.minimize(problem, 1e-10, 1e-5, false);
            cpu_cycles.push_back(cycles_count_stop().time);
            major_cycles.push_back(mnp.major_cycles);
            minor_cycles.push_back(mnp.minor_cycles);
        }

        std::cout << std::setw(fw) << n;
        std::cout << std::setw(2*fw) << mean(cpu_cycles);
        std::cout << std::setw(2*fw) << median(cpu_cycles);
        std::cout << std::setw(2*fw) << stdev(cpu_cycles);
        std::cout << std::setw(fw) << median(major_cycles);
        std::cout << std::setw(fw) << median(minor_cycles);
        std::cout << std::endl;
    }
}


int main() {
    run_validation_suite();
    run_benchmark_suite();

//    std::cout << "Min cut problem\n";
//    MinCut problem(10, 0.2);
//    std::unordered_set<int64_t> V1 = problem.get_set();
//    min_norm_point(problem, V1, 1e-10, 1e-10);

    benchmark_max_flow();

    MinNormPoint<double> mnp;

    std::cout << "Min cut problem\n";
    MinCut problem(1000, 0.2);
    mnp.minimize(problem, 1e-10, 1e-5, true);

    std::cout << "Cardinality problem\n";
    IDivSqrtSize F(500);
    mnp.minimize(F, 1e-10, 1e-10, true); 
}
