#ifndef IWATA_TEST_H
#define IWATA_TEST_H

template<class DT>
class IwataTest final : public SubmodularFunction<DT> {
public:
    int64_t n;

    IwataTest(int64_t n) : SubmodularFunction<DT>(n), n(n) {};

    DT eval(const std::vector<bool>& A) override 
    {
        int64_t cardinality = 0;
        DT val = 0.0;
        _Pragma("omp parallel for reduction(+:val, cardinality)")
        for(int64_t i = 0; i < n; i++) {
            if(A[i]) {
                cardinality++;
                val -= 5*i - 2*n;
            }
        }

        val += cardinality * (n-cardinality);
        return val;
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) override
    {
        _Pragma("omp parallel for")
        for(int64_t i = 0; i < n; i++) {
            //Cardinality term
            x(perm[i]) = n - 2*i - 1;
            //Index term
            x(perm[i]) -= 5*perm[i] - 2*n;
        }
    }
};

#endif
