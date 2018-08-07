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
    virtual DT eval(const std::unordered_set<int64_t>& A) const = 0;
    virtual std::unordered_set<int64_t> get_set() const = 0;
    virtual DT eval(const std::unordered_set<int64_t>& A, DT FA, int64_t b) const  {
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
    int64_t size;
    IDivSqrtSize(int64_t n) : size(n), FV2toR<DT>(n) {}

    DT eval(const std::unordered_set<int64_t>& A) const {
        DT val = 0.0;
        for(auto i : A) {
            val += i / sqrt(A.size());
        }
        return val;
    }

    std::unordered_set<int64_t> get_set() const {
        std::unordered_set<int64_t> V;
        V.reserve(size);
        for(int i = 0; i < size; i++) 
            V.insert(i);
        return V;
    }
};

template<class DT>
class Edge {
public:
    int64_t index;
    DT weight;
    Edge(int64_t i, DT w) : index(i), weight(w) {}
};


//submodular function for a flow network
//1 source and 1 sink, 2 groups
template<class DT>
class MinCut : public FV2toR<DT> {
public:
    //Each node has a list of edges in and a list of edges out.
    //Each edge has an index and a weight, and we will have the source and sink node have index n and n+1, respectively.
    std::vector<std::vector<Edge<DT>>> adj_in;
    std::vector<std::vector<Edge<DT>>> adj_out;
    int64_t size;
    DT baseline;

    MinCut(int64_t n) : FV2toR<DT>(n), size(n), baseline(0.0) {}
    
    //Generate a nonsymmetric random graph.
    MinCut(int64_t n, int64_t m,  double cfa, double cfb) : FV2toR<DT>(n), size(n), baseline(0.0) {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        //Initialize adjacency lists
        for(int64_t i = 0; i < n+2; i++) {
            adj_in.emplace_back(std::vector<Edge<DT>>());
            adj_out.emplace_back(std::vector<Edge<DT>>());
        }

        //Setup edges from source nodes
        int64_t k = n / m; //number of groups
        if(k < 2){
            k = 2;
            m = n / 2;
        }

        for(int64_t i = 0; i < m; i++) {
            if(dist(gen) / sqrt(m) < cfa) {
                double weight = dist(gen);
                adj_in[i].emplace_back(Edge<DT>(n, weight));
                adj_out[n].emplace_back(Edge<DT>(i, weight));
                baseline += weight;
            }
        }
        for(int64_t i = m; i < 2*m; i++) {
            if(dist(gen) < cfb) {
                double weight = dist(gen);
                adj_in[i].emplace_back(Edge<DT>(n, weight));
                adj_out[n].emplace_back(Edge<DT>(i, weight));
                baseline += weight;
            }
        }

        //Setup edges within graph
        for(int64_t p = 0; p < k; p++) {
            for(int i = 0; i < m; i++) {
                for(int j = 0; j < m; j++) {
                    //Create edges within group
                    if(i != j && dist(gen) < cfa) {
                        double weight = dist(gen);
                        adj_out[i+p*m].emplace_back(Edge<DT>(j+p*m, weight));
                        adj_in[j+p*m].emplace_back(Edge<DT>(i+p*m, weight));
                    }

                    //Create edge with previous group
                    if(p > 0 && dist(gen) < cfb) {
                        double weight = dist(gen);
                        adj_out[i+p*m].emplace_back(Edge<DT>(j+(p-1)*m, weight));
                        adj_in[j+(p-1)*m].emplace_back(Edge<DT>(i+p*m, weight));
                    }

                    //Create edge with next group
                    if(p < k-1 && dist(gen) < cfb) {
                        double weight = dist(gen);
                        adj_out[i+p*m].emplace_back(Edge<DT>(j+(p+1)*m, weight));
                        adj_in[j+(p+1)*m].emplace_back(Edge<DT>(i+p*m, weight));
                    }

                }
            }
        }

        //Setup edges to sink nodes.
        for(int64_t i = 0; i < m; i++) {
            if(dist(gen) / sqrt(m) < cfa) {
                double weight = dist(gen);
                adj_out[n-i-1].emplace_back(Edge<DT>(n+1, weight));
                adj_in[n+1].emplace_back(Edge<DT>(n-i-1, weight));
            }
        }
        
        for(int64_t i = m; i < 2*m; i++) {
            if(dist(gen) < cfb) {
                double weight = dist(gen);
                adj_out[n-i].emplace_back(Edge<DT>(n+1, weight));
                adj_in[n+1].emplace_back(Edge<DT>(n-i, weight));
            }
        }
    }

    void WattsStrogatz(int64_t k, double beta) 
    {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_real_distribution<double> uniform_node(0.0, size+2);
        std::uniform_real_distribution<double> uniform_node2(0.0, size);

        //Initialize adjacency lists
        adj_in.clear();
        adj_out.clear();
        for(int64_t i = 0; i < size+2; i++) {
            adj_in.emplace_back(std::vector<Edge<DT>>());
            adj_out.emplace_back(std::vector<Edge<DT>>());
        }

        //Pick the source and sink node randomly.  
        int64_t source = (int64_t) uniform_node2(gen);
        int64_t sink = (int64_t) uniform_node2(gen);
        while(source == sink) {
            sink = (int64_t) uniform_node2(gen);
        }

        for(int64_t i = 0; i < size+2; i++) {
            for(int64_t p = 1; p < k/2; p++) {
                double weight = dist(gen); // Weight of edge
                if(dist(gen) >= beta) {
                    //Connect to neighbor
                    if(i + p < size+2) {
                        if(i != sink && i+p != source) {
                            adj_out[i].emplace_back(Edge<DT>(i+p, weight));
                            adj_in[i+p].emplace_back(Edge<DT>(i, weight));
                        }
                        if(i != source && i+p != sink) {
                            adj_out[i+p].emplace_back(Edge<DT>(i, weight));
                            adj_in[i].emplace_back(Edge<DT>(i+p, weight));
                        }
                    }
                } 
                else {
                    //Otherwise rewire edge randomly (but not to self)
                    int64_t new_neighbor = (int64_t) uniform_node(gen);
                    while(new_neighbor == i) {
                        new_neighbor = (int64_t) uniform_node(gen);
                    }
                    if(i != sink && new_neighbor != source) {
                        adj_out[i].emplace_back(Edge<DT>(new_neighbor, weight));
                        adj_in[new_neighbor].emplace_back(Edge<DT>(i, weight));
                    }
                    if(i != source && new_neighbor != sink) {
                        adj_out[new_neighbor].emplace_back(Edge<DT>(i, weight));
                        adj_in[i].emplace_back(Edge<DT>(new_neighbor, weight));
                    }
                }
            }
        }


        //Swap locations of source, sink and last 2 nodes
        std::swap(adj_out[source], adj_out[size]);
        std::swap(adj_in[source], adj_in[size]);

        std::swap(adj_out[sink], adj_out[size+1]);
        std::swap(adj_in[sink], adj_in[size+1]);

        for(int64_t i = 0; i < size+2; i++){
            for(int64_t e = 0; e < adj_out[i].size(); e++) {
                if(adj_out[i][e].index == source) { std::cout << "ERROR!" << std::endl;}

                if(adj_out[i][e].index == sink) {
                    adj_out[i][e].index = size+1; 
                } else if(adj_out[i][e].index == size) {
                    adj_out[i][e].index = source;
                } else if(adj_out[i][e].index == size+1) { 
                    adj_out[i][e].index = sink;
                }
            }
            for(int64_t e = 0; e < adj_in[i].size(); e++) {
                if(adj_in[i][e].index == sink) { std::cout << "ERROR!" << std::endl;}

                if(adj_in[i][e].index == source)  {
                    adj_in[i][e].index = size;
                } else if(adj_in[i][e].index == size) {
                    adj_in[i][e].index = source; 
                } else if(adj_in[i][e].index == size+1) {
                    adj_in[i][e].index = sink;
                }
            }
        }
        
        //Establish baseline
        baseline = 0.0;
        for(auto a : adj_out[size]) {
            baseline += a.weight;
        }
    }

    DT eval(const std::unordered_set<int64_t>& A) const {
        DT val = 0.0;
        for(auto a : A) {
            for(auto b : adj_out[a]) {
                if(A.count(b.index) == 0)
                    val += b.weight;
            }
        }
        for(auto b : adj_out[size]) {
            if(A.count(b.index) == 0)
                val += b.weight;
        }

        return val - baseline;
    }

    DT eval(const std::unordered_set<int64_t>& A, DT FA, int64_t b) const {

        //Gain from adding b
        DT gain = 0.0;
        for(int64_t i = 0; i < adj_out[b].size(); i++) {
            if(A.count(adj_out[b][i].index) == 0)
                gain += adj_out[b][i].weight;
        }

        //Loss from adding b
        DT loss = 0.0;
        for(int64_t i = 0; i < adj_in[b].size(); i++) {
            if(adj_in[b][i].index == size || A.count(adj_in[b][i].index) != 0)
                loss -= adj_in[b][i].weight;
        }

        return FA + gain + loss;
    }

    std::unordered_set<int64_t> get_set() const {
        std::unordered_set<int64_t> V;
        V.reserve(size);
        for(int i = 0; i < size; i++) 
            V.insert(i);
        return V;
    }
};


#endif
