#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <functional>

#include "vector.h"
#include "matrix.h"
#include "perf_log.h"


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

    void polyhedron_greedy(double alpha, const Vector<DT>& weights, Vector<DT>& xout, PerfLog* log) 
    {
        int64_t start_a = rdtsc();
        //sort weights
        if (alpha > 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) > weights(b); } );
        } else if (alpha < 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) < weights(b); } );
        }
        if(log) {
            log->log("SORT TIME", rdtsc() - start_a);
        }

        int64_t start_b = rdtsc();
        DT FA_old = 0.0;
        for(int i = 0; i < xout.length(); i++) {
            DT FA = eval(A, FA_old, permutation[i]);
            xout(permutation[i]) = FA - FA_old;
            A.insert(permutation[i]);
            FA_old = FA;
        }
        A.clear();
        if(log) {
            log->log("MARGINAL GAIN TIME", rdtsc() - start_b);
        }
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

private:
    void init_adj_lists() 
    {
        adj_in.clear();
        adj_out.clear();
        for(int64_t i = 0; i < size+2; i++) {
            adj_in.emplace_back(std::vector<Edge<DT>>());
            adj_out.emplace_back(std::vector<Edge<DT>>());
        }
    }
    
    void connect_undirected(int64_t i, int64_t j, double weight)
    {
        assert(i != j);
        assert(!std::any_of(adj_out[j].begin(), adj_out[j].end(), [=](Edge<DT> e){return e.index == i;})) ;
        assert(!std::any_of(adj_out[i].begin(), adj_out[i].end(), [=](Edge<DT> e){return e.index == j;})) ;
        assert(!std::any_of(adj_in[j].begin(), adj_in[j].end(), [=](Edge<DT> e){return e.index == i;})) ;
        assert(!std::any_of(adj_in[i].begin(), adj_in[i].end(), [=](Edge<DT> e){return e.index == j;})) ;

        //Edge from i to j
        adj_out[i].emplace_back(Edge<DT>(j, weight));
        adj_in [j].emplace_back(Edge<DT>(i, weight));

        //Edge from j to i
        adj_out[j].emplace_back(Edge<DT>(i, weight));
        adj_in [i].emplace_back(Edge<DT>(j, weight));
    }

    //Utility routine to randomly select source and sink nodes
    void select_source_and_sink() 
    {
        //Select source and sink nodes randomly, but not the nodes at size or size+1,
        //so I don't have to handle the special cases
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> uniform_node(0.0, size);
        int64_t source = (int64_t) uniform_node(gen);
        int64_t sink = (int64_t) uniform_node(gen);
        while(source == sink) {
            sink = (int64_t) uniform_node(gen);
        }
        assert(source >= 0 && source < size && sink >= 0 && sink < size);

        //Swap locations of source, sink and last 2 nodes
        std::swap(adj_out[source], adj_out[size]);
        std::swap(adj_in [source], adj_in [size]);

        std::swap(adj_out[sink], adj_out[size+1]);
        std::swap(adj_in [sink], adj_in [size+1]);

        //Clear out incoming edges of source and outgoing edges of sink
        adj_in[size].clear();
        adj_out[size+1].clear();

        
        //Fix up the rest of the adjacency lists
        for(int64_t i = 0; i < size+2; i++){
            //Remove outgoing edges to source node and incoming edges from sink node
            adj_out[i].erase(std::remove_if(adj_out[i].begin(), adj_out[i].end(), [=](Edge<DT> e){ return e.index == source; }), adj_out[i].end());
            adj_in [i].erase(std::remove_if(adj_in [i].begin(), adj_in [i].end(), [=](Edge<DT> e){ return e.index == sink;   }), adj_in [i].end());

            //Redirect edges to their new sources and destinations
            for(int64_t e = 0; e < adj_out[i].size(); e++) {
                if(adj_out[i][e].index == sink) {
                    adj_out[i][e].index = size+1; 
                } else if(adj_out[i][e].index == size) {
                    adj_out[i][e].index = source;
                } else if(adj_out[i][e].index == size+1) { 
                    adj_out[i][e].index = sink;
                }
            }
            for(int64_t e = 0; e < adj_in[i].size(); e++) {
                if(adj_in[i][e].index == source)  {
                    adj_in[i][e].index = size;
                } else if(adj_in[i][e].index == size) {
                    adj_in[i][e].index = source; 
                } else if(adj_in[i][e].index == size+1) {
                    adj_in[i][e].index = sink;
                }
            }
        }
    }

    void sanity_check()
    {
        Matrix<double> sums(size+2, 5); sums.set_all(0.0);
        Vector<double> sum_in_a = sums.subcol(1);
        Vector<double> sum_in_b = sums.subcol(2);
        Vector<double> sum_out_a= sums.subcol(3);
        Vector<double> sum_out_b= sums.subcol(4);
        for(int64_t i = 0; i < size+2; i++)
            sums(i, 0) = i;

        for(int64_t i = 0; i < size+2; i++)
        {
            for(auto a: adj_in[i]) {
                sum_in_a(i) += a.weight;
                sum_out_b(a.index) += a.weight;
            }
            for(auto a: adj_out[i]) {
                sum_out_a(i) += a.weight;
                sum_in_b(a.index) += a.weight;
            }
        }
        sum_in_a.axpy(-1.0, sum_in_b);
        sum_out_a.axpy(-1.0, sum_out_b);

        if(sum_in_a.norm2() > 1e-5 || sum_out_a.norm2() > 1e-5) {
            std::cout << "Graph is invalid. Exiting." << std::endl;
            exit(1);
        }
    } 


public: 
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
   
        this->init_adj_lists();
        
        //Connect each node to K nearest neighbors.
        //With a beta % chance, rewire edge randomly
        for(int64_t i = 0; i < size+2; i++) {
            for(int64_t p = 1; p < k/2 && i+p < size+2; p++) {
                int64_t new_neighbor = i+p;
        
                if(dist(gen) < beta) {
                    int64_t new_neighbor = (int64_t) uniform_node(gen);
                    int64_t attempts = 0;
                    while(new_neighbor == i || std::any_of(adj_out[i].begin(), adj_out[i].end(), [=](Edge<DT> e){return e.index == new_neighbor;})) 
                    {
                        new_neighbor = (int64_t) uniform_node(gen);
                        attempts++;

                        if(attempts > 1000) {
                            std::cerr << "Warning: Gave up on rewiring edge randomly" << std::endl;
                            new_neighbor = i+p;
                            break;
                        }
                    }
                }
                
                this->connect_undirected(i, new_neighbor, dist(gen));         
            }
        }
        this->select_source_and_sink();

        //Establish baseline
        baseline = 0.0;
        for(auto a : adj_out[size]) {
            baseline += a.weight;
        }

        this->sanity_check();
    }

    //Place vertices randomly on the unit square and connect if their distance is less than d
    void Geometric(double d) 
    {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        this->init_adj_lists(); 

        std::vector<double> x_coords(size+2);
        std::vector<double> y_coords(size+2);
        for(int64_t i = 0; i < size+2; i++) {
            x_coords[i] = dist(gen);
            y_coords[i] = dist(gen);
        }

        for(int64_t i = 0; i < size+2; i++) {
            for(int64_t j = i+1; j < size+2; j++) {
                double x_dist = x_coords[i] - x_coords[j];
                double y_dist = y_coords[i] - y_coords[j];
                double euclidean = sqrt(x_dist * x_dist + y_dist * y_dist);
                if(euclidean < d)
                    this->connect_undirected(i, j, dist(gen));         
            }
        }
        this->select_source_and_sink();

        //Establish baseline
        baseline = 0.0;
        for(auto a : adj_out[size]) {
            baseline += a.weight;
        }

        this->sanity_check();
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
