#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <vector>
#include <functional>

using namespace std;

class FV2toR {
    public:
        virtual double eval(std::vector<int64_t>& A) = 0;
        virtual double eval(std::vector<int64_t>& A, std::function<bool(int64_t)> condition) = 0;
};

class IDivSqrtSize : public FV2toR {
    public:
    double eval(std::vector<int64_t>& A) {
        double val = 0.0;
        for(auto i : A) {
            val += i / sqrt(A.size());
        }
        return val;
    }
    double eval(std::vector<int64_t>& A, std::function<bool(int64_t)> condition) {
        double val = 0.0;
        int n = 0;
        for(auto i : A) {
            if (condition(i)) {
                val += i;
                n += 1;
            }
        }
        
        if( val == 0) 
            return 0;
        return val / sqrt(n);
    }
};

#endif
