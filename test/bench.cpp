#include <iostream>
#include <random>
#include <list>
#include <set>
#include <algorithm>

#include "../vector.h"
#include "../matrix.h"
#include "../perf/perf.h"
#include "../util.h"
#include "../submodular.h"

void benchmark_gemm()
{
    int64_t start = 256;
    int64_t end = 1024;
    int64_t inc = 256;
    int64_t n_reps = 3;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking GEMM" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 20;
    std::cout << std::setw(fw) << "n" << std::setw(fw) << "GFLOPS";
    std::cout << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        std::vector<double> cycles1;
        cycles1.reserve(n_reps);

        for(int64_t r = 0; r < n_reps; r++) {
            //Create R to remove columns from
            Matrix<double> A(n,n);
            Matrix<double> B(n,n);
            Matrix<double> C(n,n);
            A.fill_rand();
            B.fill_rand();
            C.fill_rand();

            //Time various implementations of remove_cols_incremental_QR
            cycles_count_start();
            C.mmm(1.0, A, B, 0.0);
            cycles1.push_back(cycles_count_stop().time);
        }

        std::cout << std::setw(fw) << n;
        std::cout << std::setw(fw) << 2*n*n*n / mean(cycles1) / 1e9;
        std::cout << std::endl;
    }
}

void benchmark_remove_cols()
{
    int64_t start = 128;
    int64_t end = 1024;
    int64_t inc = start;
    int64_t n_reps = 3;

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking remove columns" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 10;
    std::cout << std::setw(fw) << "n";
    std::cout << std::setw(2*fw) << "cols removed";
    std::cout << std::setw(2*fw) << "mean HH" << std::setw(2*fw) << "mean HH BLK F" << std::setw(2*fw) << "mean KR" << std::setw(2*fw) << "mean KR BLK";
    std::cout << std::setw(2*fw) << "MB/s HH" << std::setw(2*fw) << "MB/s HH BLK F" << std::setw(2*fw) << "MB/s KR" << std::setw(2*fw) << "MB/s KR BLK";
    std::cout << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;
        int n_to_remove = 10;
        int64_t nb = 48; 
        int64_t task_size = 64;

        std::vector<double> cycles1; std::vector<double> cycles2; std::vector<double> cycles3; std::vector<double> cycles4;
        cycles1.reserve(n_reps); cycles2.reserve(n_reps); cycles3.reserve(n_reps); cycles4.reserve(n_reps);

        for(int64_t r = 0; r < n_reps; r++) {
//            std::list<int64_t> cols_to_remove = get_cols_to_remove(n, percent_to_remove);
            std::list<int64_t> cols_to_remove;
            for(int i = 0; i < n_to_remove; i++)
                cols_to_remove.push_back(i);
             

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
            R.remove_cols_incremental_qr_kressner(R3, cols_to_remove, T, V, nb, ws);
            cycles3.push_back(cycles_count_stop().time);

            cycles_count_start();
            R.remove_cols_incremental_qr_tasks_kressner(R4, cols_to_remove, T, V, task_size, nb, ws);
            cycles4.push_back(cycles_count_stop().time);
        }

        std::cout << std::setw(fw) << n;
        std::cout << std::setw(2*fw) << n_to_remove;
        std::cout << std::setw(2*fw) << mean(cycles1) << std::setw(2*fw) << mean(cycles2) << std::setw(2*fw) << mean(cycles3) << std::setw(2*fw) << mean(cycles4);
        std::cout << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles1) / 1e6 
            << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles2) / 1e6  
            << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles3) / 1e6  
            << std::setw(2*fw) << sizeof(double) * n * n / mean(cycles4) / 1e6;
        std::cout << std::endl;
    }
}

void benchmark_logdet_marginal_gains()
{
    int64_t start = 128;
    int64_t end = 1024;
    int64_t inc = 128;
    int64_t n_reps = 3;

    std::cout << "===========================================================" << std::endl;
    std::cout << "Benchmarking LogDet Marginal Gains" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int fw = 20;
    std::cout << std::setw(fw) << "n";
    std::cout << std::setw(fw) << "GFLOPS 1";
    std::cout << std::setw(fw) << "GFLOPS 2";
    std::cout << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;

        LogDet<double> fast(n);
        SlowLogDet<double> slow(n);

        std::vector<int64_t> perm(n);
        for(int64_t i = 0; i < n; i++) perm[i] = i;
        Vector<double> p1(n);
        Vector<double> p2(n);

        std::vector<double> cycles1;
        std::vector<double> cycles2;

        for(int64_t r = 0; r < n_reps; r++) {
            scramble(perm);

            cycles_count_start();
            fast.eval(perm, p1);
            cycles1.push_back(cycles_count_stop().time);

            cycles_count_start();
            slow.eval(perm, p2);
            cycles2.push_back(cycles_count_stop().time);
        }

        std::cout << std::setw(fw) << n;
        std::cout << std::setw(fw) << n*n*n/3.0 / mean(cycles1) / 1e9;
        std::cout << std::setw(fw) << n*n*n/3.0 / mean(cycles2) / 1e9;
        std::cout << std::endl;
    }
}

void run_benchmark_suite()
{
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    std::cout << "Running benchmarks." << std::endl;
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    benchmark_gemm();
    benchmark_remove_cols();
    benchmark_logdet_marginal_gains();
}

