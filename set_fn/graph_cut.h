#ifndef GRAPHCUT_H
#define GRAPHCUT_H

#include "submodular.h"
#include <vector>
#include "../la/vector.h"

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
class MinCut : public SubmodularFunction<DT> {
public:
    //Each node has a list of edges in and a list of edges out.
    //Each edge has an index and a weight, and we will have the source and sink node have index n and n+1, respectively.
    std::vector<std::vector<Edge<DT>>> adj_in;
    std::vector<std::vector<Edge<DT>>> adj_out;
    int64_t n;
    DT baseline;
    unsigned seed;

    MinCut(int64_t n_in) : SubmodularFunction<DT>(n_in), n(n_in), baseline(0.0) {
        std::random_device rd;
        seed = rd();
    }
    void initialize_default()
    {
        WattsStrogatz(16, 0.25);
    }

private:
    void init_adj_lists() 
    {
        adj_in.clear();
        adj_out.clear();
        for(int64_t i = 0; i < n+2; i++) {
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
    void select_source_and_sink(std::mt19937 gen) 
    {
        //Select source and sink nodes randomly, but not the nodes at n or n+1,
        //so I don't have to handle the special cases
        std::uniform_int_distribution<int64_t> uniform_node(0, n-1);

        int64_t source = uniform_node(gen);
        int64_t sink = uniform_node(gen);
        while(source == sink) {
            sink = uniform_node(gen);
        }
        assert(source >= 0 && source < n && sink >= 0 && sink < n);

        //Swap locations (in memory) of source, sink and last 2 nodes
        std::swap(adj_out[source], adj_out[n]);
        std::swap(adj_in [source], adj_in [n]);

        std::swap(adj_out[sink], adj_out[n+1]);
        std::swap(adj_in [sink], adj_in [n+1]);

        //Clear out incoming edges of source and outgoing edges of sink
        adj_in[n].clear();
        adj_out[n+1].clear();
       
        double weight_factor = 4.0; 
        //scale outgoing weights of source and incoming weights of sink
        for(int64_t j = 0; j < adj_out[n].size(); j++) {
            adj_out[n][j].weight *= weight_factor;
        }
        for(int64_t j = 0; j < adj_in[n+1].size(); j++) {
            adj_in[n+1][j].weight *= weight_factor;
        }

        //Fix up the rest of the adjacency lists
        for(int64_t i = 0; i < n+2; i++){
            //Remove outgoing edges to source node and incoming edges from sink node
            adj_out[i].erase(std::remove_if(adj_out[i].begin(), adj_out[i].end(), [=](Edge<DT> e){ return e.index == source; }), adj_out[i].end());
            adj_in [i].erase(std::remove_if(adj_in [i].begin(), adj_in [i].end(), [=](Edge<DT> e){ return e.index == sink;   }), adj_in [i].end());

            //Redirect edges to their new sources and destinations
            for(int64_t e = 0; e < adj_out[i].size(); e++) {
                if(adj_out[i][e].index == sink) {
                    adj_out[i][e].index = n+1; 
                    adj_out[i][e].weight *= weight_factor;
                } else if(adj_out[i][e].index == n) {
                    adj_out[i][e].index = source;
                } else if(adj_out[i][e].index == n+1) { 
                    adj_out[i][e].index = sink;
                }
            }
            for(int64_t e = 0; e < adj_in[i].size(); e++) {
                if(adj_in[i][e].index == source)  {
                    adj_in[i][e].index = n;
                    adj_in[i][e].weight *= weight_factor;
                } else if(adj_in[i][e].index == n) {
                    adj_in[i][e].index = source; 
                } else if(adj_in[i][e].index == n+1) {
                    adj_in[i][e].index = sink;
                }
            }
        }
    }

    void sanity_check()
    {
        Vector<double> sum_in_a(n+2); sum_in_a.set_all(0.0);
        Vector<double> sum_in_b(n+2); sum_in_b.set_all(0.0);
        Vector<double> sum_out_a(n+2); sum_out_a.set_all(0.0);
        Vector<double> sum_out_b(n+2); sum_out_b.set_all(0.0);

        for(int64_t i = 0; i < n+2; i++)
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
    void WattsStrogatz(int64_t k, double beta) 
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.1, 1.0);
        std::uniform_int_distribution<int64_t> uniform_node(0, n+1);
   
        this->init_adj_lists();
        
        //Connect each node to K nearest neighbors.
        //With a beta % chance, rewire edge randomly
        for(int64_t i = 0; i < n+2; i++) {
            for(int64_t p = 1; p < k/2 && i+p < n+2; p++) {
                int64_t new_neighbor = i+p;
        
                if(dist(gen) < beta) {
                    int64_t new_neighbor = uniform_node(gen);
                    int64_t attempts = 0;
                    while(new_neighbor == i || std::any_of(adj_out[i].begin(), adj_out[i].end(), [=](Edge<DT> e){return e.index == new_neighbor;})) 
                    {
                        new_neighbor = uniform_node(gen);
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
        this->select_source_and_sink(gen);

        //Establish baseline
        baseline = 0.0;
        for(auto a : adj_out[n]) {
            baseline += a.weight;
        }

        this->sanity_check();
    }

    //Place vertices randomly on the unit square and connect if their distance is less than d
    void Geometric(double d) 
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_real_distribution<double> weight_dist(0.1, 1.0);

        this->init_adj_lists(); 

        std::vector<double> x_coords(n+2);
        std::vector<double> y_coords(n+2);
        for(int64_t i = 0; i < n+2; i++) {
            x_coords[i] = dist(gen);
            y_coords[i] = dist(gen);
        }

        for(int64_t i = 0; i < n+2; i++) {
            for(int64_t j = i+1; j < n+2; j++) {
                double x_dist = x_coords[i] - x_coords[j];
                double y_dist = y_coords[i] - y_coords[j];
                double euclidean = sqrt(x_dist * x_dist + y_dist * y_dist);
                if(euclidean < d)
                    this->connect_undirected(i, j, weight_dist(gen));         
            }
        }
        this->select_source_and_sink(gen);

        //Establish baseline
        baseline = 0.0;
        for(auto a : adj_out[n]) {
            baseline += a.weight;
        }

        this->sanity_check();
    }

    DT eval(const std::vector<bool>& A) 
    {
        DT val = 0.0;
        for(int64_t i = 0; i < n; i++) {
            if(!A[i]) continue;
            for(auto b : adj_out[i]) {
                if(b.index == n+1 || !A[b.index])
                    val += b.weight;
            }
        }
        for(auto b : adj_out[n]) {
            if(b.index == n+1 || !A[b.index])
                val += b.weight;
        }

        return val - baseline;
    }


    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        std::vector<int64_t> perm_lookup(n);
//        _Pragma("omp parallel for")
        for(int64_t i = 0; i < n; i++) {
            perm_lookup[perm[i]] = i;
        }

        //Iterate over every edge.
        //Each edge connects two nodes, a and b. We get a gain when the first of the nodes joins A and a loss when the second node joins A.
        x.set_all(0.0);
        _Pragma("omp parallel for")
        for(int64_t a = 0; a < n; a++) {
            int64_t index_a = perm_lookup[a];
            for(auto edge : adj_out[a]) {
                int64_t b = edge.index;
                if(b == n+1) {
                    x(a) += edge.weight;
                }
                else {
                    int64_t index_b = perm_lookup[b];

                    if(index_a < index_b) {
                        x(a) += edge.weight;
                    } else {
                        x(a) -= edge.weight;
                    }
                }
            }
        }

        for(auto edge : adj_out[n]) {
            if(edge.index != n+1) x(edge.index) -= edge.weight;
        }
    }
};

template<class DT>
class SlowMinCut : public SubmodularFunction<DT> {
public:
    std::vector<std::vector<Edge<DT>>> adj_in;
    std::vector<std::vector<Edge<DT>>> adj_out;
    int64_t n;
    DT baseline;

    SlowMinCut(const MinCut<DT>& other) : SubmodularFunction<DT>(other.n), n(other.n), adj_in(other.adj_in), adj_out(other.adj_out), baseline(other.baseline) { }
    SlowMinCut(int64_t n_in) : SubmodularFunction<DT>(n_in), n(n_in)
    {
        MinCut<DT> other(n);
        other.initialize_default();
        adj_in = other.adj_in;
        adj_out = other.adj_out;
        baseline = other.baseline;
    }

    DT eval(const std::vector<bool>& A) 
    {
        DT val = 0.0;
        for(int64_t i = 0; i < n; i++) {
            if(!A[i]) continue;
            for(auto b : adj_out[i]) {
                if(b.index == n+1 || !A[b.index])
                    val += b.weight;
                //if(!A[b.index])
            }
        }
        for(auto b : adj_out[n]) {
            if(b.index == n+1 || !A[b.index])
                val += b.weight;
        }

        return val - baseline;
    }

    DT gain(const std::vector<bool>& A, int64_t b) 
    {
        //Gain from adding b
        DT gain = 0.0;
        for(int64_t i = 0; i < adj_out[b].size(); i++) {
            if(!A[adj_out[b][i].index])
                gain += adj_out[b][i].weight;
        }

        //Loss from adding b
        DT loss = 0.0;
        for(int64_t i = 0; i < adj_in[b].size(); i++) {
            if(adj_in[b][i].index == n || A[adj_in[b][i].index])
                loss -= adj_in[b][i].weight;
        }

        return gain + loss;
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        _Pragma("omp parallel") 
        {
            int64_t t_id = omp_get_thread_num();
            int64_t nt = omp_get_num_threads();

            int64_t n_per_thread = (n - 1) / nt + 1;
            int64_t start = n_per_thread * t_id;
            int64_t end = std::min(start + n_per_thread, n);

            //Each thread must maintain its own set A
            std::vector<bool> A(perm.size());
            std::fill(A.begin(), A.end(), 0);
            for(int64_t i = 0; i < std::min(start, n); i++) A[perm[i]] = 1;

            for(int64_t j = start; j < end; j++) {
                int64_t b = perm[j];
                x(b) = gain(A, b);
                A[b] = 1;
            }
        }
    }
};
#endif
