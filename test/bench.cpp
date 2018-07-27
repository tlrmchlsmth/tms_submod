#include <iostream>
#include <random>
#include <list>
#include <set>
#include <algorithm>

#include "../vector.h"
#include "../matrix.h"
#include "../perf/perf.h"


template<class RNG, class DIST>
std::list<int64_t> get_cols_to_delete(int64_t m, double percent_to_delete, RNG &gen, DIST& dist)
{
    int64_t n_cols_to_delete = std::round(m * percent_to_delete);
    std::set<int64_t> cols_to_delete;

    while(cols_to_delete.size() < n_cols_to_delete) {
        cols_to_delete.insert(dist(gen));
    }

    std::list<int64_t> to_ret;
    for(auto it = cols_to_delete.begin(); it != cols_to_delete.end(); ++it) {
        to_ret.push_back(*it);
    }

    return to_ret;
}

void benchmark_delete_cols()
{
    int64_t start = 16;
    int64_t end = 256;
    int64_t inc = 16;
    int64_t n_reps = 10;
    double percent_to_delete = 0.1; 

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

    int fw = 10;
    std::cout << std::setw(fw) << "m" << std::setw(fw) << "n" << std::setw(fw) << "task sz" << std::setw(fw) << "nb" << std::setw(2*fw) << "mean (s)" << std::setw(2*fw) <<   "stdev (s)" << std::setw(2*fw) << "BW" << std::endl;
    for(int64_t i = start; i <= end; i += inc) {
        int64_t n = i;
        int64_t m = n;
        int64_t nb = 32; 
        int64_t task_size = 128;

        std::uniform_int_distribution<> dist(0,n-1);
        std::vector<double> cycles;
        cycles.reserve(n_reps);

        for(int64_t r = 0; r < n_reps; r++) {
            std::list<int64_t> cols_to_delete = get_cols_to_delete(n, percent_to_delete, gen, dist);

            //1. Create random S
            Matrix<double> S(m,n);
            S.fill_rand(gen, normal);

            //2. Perform a QR factorization of S
//            Matrix<double> RT(n,m);
//            auto R = RT.transposed();
            Matrix<double> R(m,n);
            Vector<double> t(n);
            Matrix<double> T(nb,n);
            Matrix<double> V(cols_to_delete.size(),n);
            Matrix<double> ws(nb, n);
            Matrix<double> Rinit(m,n);
            R.copy(S);
            R.qr(t);
            Rinit.copy(R);
            
            auto R0 = R.submatrix(0,0,n,n);
            auto Rinit0 = Rinit.submatrix(0,0,n,n);

            //3. Call delete_cols_incremental_QR, timing it.
            cycles_count_start();
        //    S.remove_cols(cols_to_delete);
        //    R0.blocked_remove_cols_incremental_qr(cols_to_delete, t, nb);
            Rinit0.remove_cols_incremental_qr_tasks_kressner(R0, cols_to_delete, T, V, task_size, nb, ws);
            //R0.kressner_remove_cols_incremental_qr(cols_to_delete, T, V, nb, ws);
            //R.remove_cols_incremental_qr(cols_to_delete, t);
            cycles.push_back(cycles_count_stop().time);
        }

        double mean = std::accumulate(cycles.begin(), cycles.end(), 0.0) / cycles.size();
        std::vector<double> diff(cycles.size());
        std::transform(cycles.begin(), cycles.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / cycles.size());
        std::cout << std::setw(fw) << m;
        std::cout << std::setw(fw) << n;
        std::cout << std::setw(fw) << task_size;
        std::cout << std::setw(fw) << nb;
        std::cout << std::setw(2*fw) << mean;
        std::cout << std::setw(2*fw) << stdev;
        std::cout << std::setw(2*fw) << sizeof(double) * (2*n*n) / mean / 1e6  << " MB/s" << std::endl;
    }
}

void run_benchmark_suite()
{
    benchmark_delete_cols();
}
