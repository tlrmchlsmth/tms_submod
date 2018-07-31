#include <iostream>
#include <random>
#include <list>
#include <set>
#include <algorithm>

#include "../vector.h"
#include "../matrix.h"
#include "../perf/perf.h"
#include "../util.h"

void benchmark_remove_cols()
{
    int64_t start = 256;
    int64_t end = 1024;
    int64_t inc = 256;
    int64_t n_reps = 10;
    double percent_to_remove = 0.1; 

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking remove columns" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 10;
    std::cout << std::setw(fw) << "n";
    std::cout << std::setw(2*fw) << "mean 1" << std::setw(2*fw) << "mean 2" << std::setw(2*fw) << "mean 3" << std::setw(2*fw) << "mean 4";
    std::cout << std::setw(2*fw) << "BW 1" << std::setw(2*fw) << "BW 2" << std::setw(2*fw) << "BW 3" << std::setw(2*fw) << "BW 4";
    std::cout << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;
        int64_t nb = 48; 
        int64_t task_size = 64;

        std::vector<double> cycles1; std::vector<double> cycles2; std::vector<double> cycles3; std::vector<double> cycles4;
        cycles1.reserve(n_reps); cycles2.reserve(n_reps); cycles3.reserve(n_reps); cycles4.reserve(n_reps);

        for(int64_t r = 0; r < n_reps; r++) {
            std::list<int64_t> cols_to_remove = get_cols_to_remove(n, percent_to_remove);

            //Create R to remove columns from
            Matrix<double> R(n,n);
            Matrix<double> R1(n,n); Matrix<double> R2(n,n); Matrix<double> R3(n,n); Matrix<double> R4(n,n);
            Vector<double> t(n);
            Matrix<double> T(nb,n);
            Matrix<double> V(cols_to_remove.size(),n);
            Matrix<double> ws(nb, n);
            R.fill_rand();
            R.qr(t);
            R.set_subdiagonal(0.0);
            R1.copy(R); R2.copy(R); R3.copy(R); R3.copy(R);
            
            //Time various implementations of remove_cols_incremental_QR
            cycles_count_start();
            R1.remove_cols_incremental_qr_householder(cols_to_remove, t);
            cycles1.push_back(cycles_count_stop().time);

            cycles_count_start();
            R2.remove_cols_incremental_qr_blocked_householder(cols_to_remove, t, nb);
            cycles2.push_back(cycles_count_stop().time);

            cycles_count_start();
            R3.remove_cols_incremental_qr_kressner(cols_to_remove, T, V, nb, ws);
            cycles3.push_back(cycles_count_stop().time);

            cycles_count_start();
            R.remove_cols_incremental_qr_tasks_kressner(R4, cols_to_remove, T, V, task_size, nb, ws);
            cycles4.push_back(cycles_count_stop().time);
        }

        std::cout << std::setw(fw) << n;
        std::cout << std::setw(2*fw) << mean(cycles1) << std::setw(2*fw) << mean(cycles2) << std::setw(2*fw) << mean(cycles3) << std::setw(2*fw) << mean(cycles4);
        std::cout << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles1) / 1e9 
            << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles2) / 1e9  
            << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles3) / 1e9  
            << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles4) / 1e9;
        std::cout << std::endl;
    }
}

void run_benchmark_suite()
{
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    std::cout << "Running benchmarks." << std::endl;
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    benchmark_remove_cols();
}
