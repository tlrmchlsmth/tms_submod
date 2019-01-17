#ifndef SUM_SUBMODULAR_H
#define SUM_SUBMODULAR_H

#include "submodular.h"
#include "../la/vector.h"

template<class DT>
class SumSubmodulars : public SubmodularFunction<DT> {
public:
    std::vector<std::unique_ptr<SubmodularFunction<DT>>> _fns;

    SumSubmodulars(int64_t n) : SubmodularFunction<DT>(n) {}

    SumSubmodulars(int64_t n, std::vector<std::unique_ptr<SubmodularFunction<DT>>> fns) : SubmodularFunction<DT>(n) {
        _fns = std::move(fns);
    }

    DT eval(const std::vector<bool>& A) {
        DT val = 0.0;
        for( auto&& f : _fns ) {
            val += f->eval(A);
        }
        return val;
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& p) {
        Vector<DT> tmp(p.length());
        p.set_all(0.0);
        for( auto&& f : _fns) {
            f->gains(perm, tmp);
            p.axpy(1.0, tmp);
        }
    }
};
#endif
