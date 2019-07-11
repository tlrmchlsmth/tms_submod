#ifndef PLUS_MODULAR_H
#define PLUS_MODULAR_H

#include "submodular.h"
#include "../la/vector.h"

template<class DT, class SFN>
class PlusModular : public SubmodularFunction<DT> {
public:
    int64_t n;
    SFN submodular;
    Vector<DT> modular;

    PlusModular(int64_t n_in) :
        SubmodularFunction<DT>(n_in), n(n_in), submodular(n_in), modular(n_in)
    {
        std::random_device rd;
        std::mt19937 gen{rd()};

        //Generate modular function
        std::normal_distribution<> modular_dist(1.0, 10.0);
        modular.fill_rand(gen, modular_dist);

        //Scale it so F(V) = 0
        std::vector<bool> V(n);
        std::fill(V.begin(), V.end(), true); 
        modular.scale(-submodular.eval(V) / modular.sum());
    }

    template<class DIST>
    PlusModular(int64_t n_in, SFN fn_in, DIST d) :
        SubmodularFunction<DT>(n_in), n(n_in), submodular(fn_in), modular(n_in)
    {
        std::random_device rd;
        std::mt19937 gen{rd()};

        //Generate modular function
        modular.fill_rand(gen, d);
        
        //Scale it so F(V) = 0
        std::vector<bool> V(n);
        std::fill(V.begin(), V.end(), true); 
        modular.scale(-submodular.eval(V) / modular.sum());
    }

    PlusModular(int64_t n_in, SFN fn_in) : 
        SubmodularFunction<DT>(n_in), n(n_in), submodular(fn_in), modular(n_in)
    { }

    DT eval(const std::vector<bool>& A) 
    {
        DT modular_val = 0.0;
        for(int64_t i = 0; i < n; i++) {
            if(A[i]) modular_val += modular(i);
        }
        return modular_val + submodular.eval(A);
    }

    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& x) {
        submodular.gains(perm, x);
        x.axpy(1.0, modular);
    }
};

#endif
