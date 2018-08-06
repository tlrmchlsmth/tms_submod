#include <set>
#include <list>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cassert>
#include <algorithm>

#include "ipps.h"
#include "mkl.h"

std::list<int64_t> get_cols_to_remove(int64_t m, double percent_to_remove)
{
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<> dist(0,m-1);

    int64_t n_cols_to_remove = std::round(m * percent_to_remove);
    std::set<int64_t> cols_to_remove;

    while(cols_to_remove.size() < n_cols_to_remove) {
        cols_to_remove.insert(dist(gen));
    }   

    std::list<int64_t> to_ret;
    for(auto it = cols_to_remove.begin(); it != cols_to_remove.end(); ++it) {
        to_ret.push_back(*it);
    }   

    return to_ret;
}

void print_err(double err, int64_t w) 
{
    if(std::abs(err) < 1e-6) {
        std::cout << "\033[1;32m" << std::setw(w) << err << "\033[0m";
    } else if(std::abs(err) < 1e-4) {
        std::cout << "\033[1;33m" << std::setw(w) << err << "\033[0m";
    } else {
        std::cout << "\033[1;31m" << std::setw(w) << err << "\033[0m";
    }
}
void print_err(float err, int64_t w) {
    if(std::abs(err) < 1e-4) {
        std::cout << "\033[1;32m" << std::setw(w) << err << "\033[0m";
    } else if(std::abs(err) < 1e-2) {
        std::cout << "\033[1;33m" << std::setw(w) << err << "\033[0m";
    } else {
        std::cout << "\033[1;31m" << std::setw(w) << err << "\033[0m";
    }
}


//#define BLAS_HOUSE        
void house_apply(int64_t m, int64_t n, double * v, int64_t stride, double tau, double* X, int64_t x_rs, int64_t x_cs) {
    #pragma omp parallel for
    for(int j = 0; j < n; j++) {
        //BLAS VERSION
#ifdef BLAS_HOUSE
        double vt_x = X[j*x_cs] + cblas_ddot(_len-1, &v[1], stride, &X[1 + j*x_cs], x_rs);
        double alpha = tau * vt_x;
        X[0 + j*x_cs] -= alpha;
        cblas_daxpy(m-1, -alpha, &v[1], stride, &X[x_rs + j*x_cs], x_rs);
#else
        //IPP version
        assert(x_rs == 1);
        double vt_x;
        ippsDotProd_64f(&v[1], &X[1 + j*x_cs], m-1, &vt_x);
        vt_x += X[j*x_cs];
        double alpha = tau * vt_x;
        X[j*x_cs] -= alpha;
        ippsAddProductC_64f(&v[1], -alpha, &X[1 + j*x_cs], m-1);
#endif
    }
}
