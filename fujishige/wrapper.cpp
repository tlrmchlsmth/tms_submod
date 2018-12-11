#include <iostream>
#include "../submodular.h"

extern "C" {
    int func_fw     (int n, double sf(int *p, int j, int n), double *x);
    int func_smin   (int n, double sf(int *p, int j, int n), double *x, int* s);
    int func_smax   (int n, double sf(int *p, int j, int n), double *x, int* s);
}

static SubmodularFunction<double>* fujishige_wrapper_f;
static Vector<double>* fujishige_wrapper_ws;
static std::vector<int64_t>* fujishige_wrapper_perm;

double fujishige_wrapper_oracle(int* p, int j, int n) {
    assert(j > 0);
    if(j == 1) {
        for(int64_t i = 0; i < n; i++) {
            (*fujishige_wrapper_perm)[i] = p[i+1]-1;
        }

        fujishige_wrapper_f->gains((*fujishige_wrapper_perm), (*fujishige_wrapper_ws));
        double sum = 0.0;

        for(int64_t i = 0; i < n; i++) {
            sum += (*fujishige_wrapper_ws)((*fujishige_wrapper_perm)[i]);
            (*fujishige_wrapper_ws)((*fujishige_wrapper_perm)[i]) = sum;
        }
    }
    return (*fujishige_wrapper_ws)((*fujishige_wrapper_perm)[j-1]);
}

std::vector<bool> run_isotani_and_fujishige(SubmodularFunction<double>& F) {
    //Initialize workspace for oracle
    Vector<double> ws(F.n);
    std::vector<int64_t> p(F.n);

    //Setup global variables for oracle
    fujishige_wrapper_f = &F;
    fujishige_wrapper_ws = &ws;
    fujishige_wrapper_perm = &p;

    double x [F.n+1];
    func_fw(F.n, fujishige_wrapper_oracle, x);

    int smin[F.n+1];
    int jmin = func_smin(F.n, fujishige_wrapper_oracle, x, smin);
    std::vector<bool> A(F.n, false);
    for(int i = 1; i <= jmin; i++) {
        A[smin[i]-1] = true;
    }
    //exit(1);
    return A;
}
