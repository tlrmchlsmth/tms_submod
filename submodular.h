#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <functional>

#include "vector.h"
#include "matrix.h"

template<class DT>
class FV2toR {
protected:
    //Workspace for the greedy algorithm
    std::unordered_set<int64_t> A;
    std::vector<int64_t> permutation;

public:
    FV2toR(int64_t n) 
    {
        A.reserve(n);
        permutation.reserve(n);
        for(int i = 0; i < n; i++) 
            permutation.push_back(i);
    }
    virtual DT eval(const std::unordered_set<int64_t>& A) = 0;
    virtual DT eval(const std::unordered_set<int64_t>& A, std::function<bool(int64_t)> condition) = 0;
    virtual std::unordered_set<int64_t> get_set() = 0;
    virtual DT eval(const std::unordered_set<int64_t>& A, DT FA, int64_t b) {
        std::unordered_set<int64_t> Ab = A;
        Ab.insert(b);
        DT FAb = this->eval(Ab);
        return FAb;
    }

    void polyhedron_greedy(double alpha, const Vector<DT>& weights, Vector<DT>& xout) 
    {
        //sort weights
        if (alpha > 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) > weights(b); } );
        } else if (alpha < 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) < weights(b); } );
        }

        DT FA_old = 0.0;
        for(int i = 0; i < xout.length(); i++) {
            DT FA = eval(A, FA_old, permutation[i]);
            xout(permutation[i]) = FA - FA_old;
            A.insert(permutation[i]);
            FA_old = FA;
        }
        A.clear();
    }
};

template<class DT>
class IDivSqrtSize : public FV2toR<DT> {
public:
    int64_t _n;
    IDivSqrtSize(int64_t n) : _n(n), FV2toR<DT>(n) {}

    DT eval(const std::unordered_set<int64_t>& A) {
        DT val = 0.0;
        for(auto i : A) {
            val += i / sqrt(A.size());
        }
        return val;
    }

    DT eval(const std::unordered_set<int64_t>& A, std::function<bool(int64_t)> condition) {
        DT val = 0.0;
        int n = 0;
        for(auto i : A) {
            if (condition(i)) {
                val += i;
                n += 1;
            }
        }

        if(n != 0) val = val / sqrt(n); 
        return val;
    }

    std::unordered_set<int64_t> get_set() {
        std::unordered_set<int64_t> V;
        V.reserve(_n);
        for(int i = 0; i < _n; i++) 
            V.insert(i);
        return V;
    }
};


//submodular function for a flow network
//1 source and 1 sink, 2 groups
template<class DT>
class MinCut : public FV2toR<DT> {
public:
    Matrix<DT> adjacency;
    Vector<DT> edges_from_source;
    Vector<DT> edges_to_sink;
    int64_t _n;
    DT baseline;
    
    //Generate a nonsymmetric random graph.
    MinCut(int64_t n, double connectivity_factor) :
        FV2toR<DT>(n),
        _n(n), adjacency(Matrix<DT>(n,n)), baseline(0.0),
        edges_from_source(Vector<DT>(n)),
        edges_to_sink(Vector<DT>(n)) 
    {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        adjacency.set_all(0.0);
        edges_from_source.set_all(0.0);
        edges_to_sink.set_all(0.0);

        //Setup edges for source and sink nodes
        for(int i = 0; i < n/2; i++) {
            if(dist(gen) < connectivity_factor) {
                edges_from_source(i) = dist(gen);
                baseline += edges_from_source(i);
            }
        }
        for(int i = n/2; i < n; i++) {
            if(dist(gen) < connectivity_factor)
                edges_to_sink(i) = dist(gen);

        }

        //Setup edges for other nodes
        /*
        for(int j = 0; j < n; j++) {
            for(int i = 0; i < n; i++) {
                adjacency(i,j) = 0.0;
                if(dist(gen) < connectivity_factor)
                    adjacency(i,j) = dist(gen);
            }
        }*/

        for(int j = 0; j < n/2; j++) {
            for(int i = 0; i < n/2; i++) {
                if(dist(gen) < connectivity_factor)
                    adjacency(i,j) = dist(gen);
            }
        }

        for(int j = n/2; j < n; j++) {
            for(int i = n/2; i < n; i++) {
                if(dist(gen) < connectivity_factor)
                    adjacency(i,j) = dist(gen);
            }
        }

        //Setup edges between groups
        std::uniform_real_distribution<double> zero_to_quarter_n(0.0, n/4);
        std::uniform_real_distribution<double> zero_to_half_n(0.0, n/2);
        int64_t n_connections_between_groups = (int64_t) zero_to_quarter_n(gen);
        for(int64_t i = 0; i < n_connections_between_groups; i++) {
            int64_t x = zero_to_half_n(gen);
            int64_t y = zero_to_half_n(gen);
            adjacency(x,y + n/2) = dist(gen);
        }
    }

    //Generate a nonsymmetric random graph.
    MinCut(int64_t n, int64_t m,  double cfa, double cfb) : 
        FV2toR<DT>(n),
        _n(n), adjacency(Matrix<DT>(n,n)), baseline(0.0),
        edges_from_source(Vector<DT>(n)),
        edges_to_sink(Vector<DT>(n)) 
    {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        adjacency.set_all(0.0);
        edges_from_source.set_all(0.0);
        edges_to_sink.set_all(0.0);


        //Setup edges from source nodes
        int64_t k = n / m; //number of groups
        if(k < 2){
            k = 2;
            m = n / 2;
        }

        for(int64_t i = 0; i < m; i++) {
            if(dist(gen) / sqrt(m) < cfa) {
                edges_from_source(i) = dist(gen);
                baseline += edges_from_source(i);
            }
        }
        for(int64_t i = m; i < 2*m; i++) {
            if(dist(gen) < cfb) {
                edges_from_source(i) = dist(gen);
                baseline += edges_from_source(i);
            }
        }

        //Setup edges within graph
        for(int64_t p = 0; p < k; p++) {
            for(int i = 0; i < m; i++) {
                for(int j = 0; j < m; j++) {
                    //Create edges within group
                    if(dist(gen) < cfa) {
                        adjacency(i + p*m, j + p*m) = dist(gen);
                    }

                    //Create edge with previous group
                    if(p > 0 && dist(gen) < cfb) {
                        adjacency(i + p*m, j + (p-1)*m) = dist(gen);
                    }

                    //Create edge with next group
                    if(p < k-1 && dist(gen) < cfb) {
                        adjacency(i + p*m, j + (p+1)*m) = dist(gen);
                    }

                }
            }
        }

        //Setup edges to sink nodes.
        for(int64_t i = 0; i < m; i++) {
            if(dist(gen) / sqrt(m) < cfa) {
                edges_to_sink(n-i-1) = dist(gen);
            }
        }
        
        for(int64_t i = m; i < 2*m; i++) {
            if(dist(gen) < cfb) {
                edges_to_sink(n-i) = dist(gen);
            }
        }
    }

    DT eval(const std::unordered_set<int64_t>& A) {
        DT val = 0.0;
        for(auto a : A){
            //Edges within graph
            for(int64_t i = 0; i < _n; i++) {
                if(a != i && A.count(i) == 0) {
                    val += adjacency(a,i);
                }
            }
            //Edge to sink
            val += edges_to_sink(a);
        }

        //Edges from source
        for(int64_t i = 0; i < _n; i++) {
            if(A.count(i) == 0) {
                val += edges_from_source(i);
            }
        }
        
        return val - baseline;
    }

    DT eval(const std::unordered_set<int64_t>& A, std::function<bool(int64_t)> condition) {
        DT val = 0.0;
        for(auto a : A){
            if(!condition(a)) continue;

            //Edges within graph
            for(int64_t i = 0; i < _n; i++) {
                if(a != i && (A.count(i) == 0 || !condition(i))) {
                    val += adjacency(a, i);
                }
            }

            //Edge to sink
            val += edges_to_sink(a);
        }
        
        //Edges from source
        for(int64_t i = 0; i < _n; i++) {
            if(!condition(i) || A.count(i) == 0) {
                val += edges_from_source(i);
            }
        }

        //return val;
        return val - baseline;
    }

    DT eval(const std::unordered_set<int64_t>& A, DT FA, int64_t b) {

        //Gain from adding b
        DT gain = 0.0;
        for(int64_t i = 0; i < adjacency.height(); i++) {
            if(b != i && A.count(i) == 0) {
                gain += adjacency(b, i);
            }
        }
        gain += edges_to_sink(b);

        //Loss from adding b
        DT loss = 0.0;
        for(auto a : A){
            loss -= adjacency(a, b);
        }
        loss -= edges_from_source(b);

        return FA + gain + loss;
    }

    std::unordered_set<int64_t> get_set() {
        std::unordered_set<int64_t> V;
        int64_t n = adjacency.height();
        V.reserve(n);
        for(int i = 0; i < n; i++) 
            V.insert(i);
        return V;
    }
};


#endif
