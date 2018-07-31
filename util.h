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

#endif
