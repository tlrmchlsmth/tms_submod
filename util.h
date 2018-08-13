#ifndef UTIL_H
#define UTIL_H

#include <vector>

template<class DT>
DT mean(const std::vector<DT> v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
template<class DT>
DT stdev(const std::vector<DT> v) {
    DT mu = mean(v);
    std::vector<DT> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mu](DT x) { return x - mu; });
    DT sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    DT stdev = std::sqrt(sq_sum / v.size());
    return stdev;
}
template<class DT>
DT median(const std::vector<DT> v_in) {
    auto v = v_in;
    std::sort(v.begin(), v.end());
    DT median = v[v.size()/2];
    if(v.size() % 2 == 0) {
        median = (median + v[v.size()/2 - 1]) / 2.0;
    }
    return median;
}

std::list<int64_t> get_cols_to_remove(int64_t m, double percent_to_remove);

void print_err(double err, int64_t w);
void print_err(float err, int64_t w);

void house_apply(int64_t m, int64_t n, double * v, int64_t stride, double tau, double* X, int64_t x_rs, int64_t x_cs);

template<class DT>
DT house_gen(int64_t m, DT* x, int64_t stride) 
{
    DT chi1 = x[0];
    DT nrm_x2_sqr = 0.0;
    
    for(int64_t i = 1; i < m; i++) {
        DT xi = x[i*stride];
        nrm_x2_sqr += xi * xi;
    }
    DT nrm_x  = sqrt(chi1*chi1 + nrm_x2_sqr);

    DT tau = 0.5;
    if(nrm_x2_sqr == 0) {
        x[0] = -chi1;
        return tau;
    }

    DT alpha = -sgn(chi1) * nrm_x;
    DT mult = 1.0 / (chi1 - alpha);
    
    for(int64_t i = 1; i < m; i++) {
        x[i] *= mult;
    }

    tau = 1.0 /  (0.5 + 0.5 * nrm_x2_sqr * mult * mult);
    x[0] = alpha;
    return tau;
}
#endif
