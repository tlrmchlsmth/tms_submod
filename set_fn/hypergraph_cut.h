#ifndef HYPERGRAPHCUT_H
#define HYPERGRAPHCUT_H

#include "submodular.h"
#include <vector>
#include "../la/vector.h"

template<class DT>
class HyperEdge {
public:
    std::vector<int64_t> v;
    DT w;
    HyperEdge(std::vector<int64_t> v_in, DT w_in) : v(v_in), w(w_in) {}
};


template<class DT>
class HyperCut final : public SubmodularFunction<DT> {
public:
    int64_t n;
    std::vector<HyperEdge<DT>> edges;

    HyperCut(int64_t n_in) : SubmodularFunction<DT>(n_in), n(n_in) {
        GeneralizedWattsStrogatz(4, 2, 0.25);
    }

    void GeneralizedWattsStrogatz(int64_t k, int64_t r, double beta) 
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> weight_dist(0.0, 1.0);
        std::uniform_real_distribution<double> connect_dist(0.0, 1.0);
        std::uniform_int_distribution<int64_t> uniform_node(0, n-1);

        edges.reserve(k*n);
        //Each vertex is connected to k nearest neighbors
        //Each hyper edge connects to r vertices
        for(int64_t i = 0; i < n; i++) {
            for(int64_t p = 1; p < k/2; p++) {

                //Construct hyperedge. Connect to current vertex and adjacent vertices
                std::vector<int64_t> vertices;
                vertices.reserve(r);
                vertices.push_back(i);

                for(int64_t q = 0; q < r; q++) {
                    vertices.push_back((i + p + q) % n);
                }

                edges.emplace_back(vertices, weight_dist(gen));
            }
        }

        //With probability beta, rewire each vertex in edge
        for(auto e : edges) {
            for(uint64_t i = 0; i < e.v.size(); i++) {
                if(connect_dist(gen) < beta) {
                    int64_t new_v = e.v[i];
                    int64_t k = 0;
                    do {
                        new_v = uniform_node(gen);
                        if(k++ > 100) {
                            //Give up rewiring.
                            new_v = e.v[i];
                            break;
                        }
                    } while(std::find(e.v.begin(), e.v.end(), new_v) != e.v.end());
                    e.v[i] = new_v;
                }
            }
        }
    }


    DT eval(const std::vector<bool>& A) override
    {
        DT val = 0.0;

        #pragma omp parallel for reduction(+ : val)
        for(uint64_t j = 0; j < edges.size(); j++) {
            bool is_cut = false;
            for(uint64_t i = 1; i < edges[j].v.size(); i++) {
                if(A[edges[j].v[i]] != A[edges[j].v[0]]) {
                    is_cut = true;
                    break;
                }
            }
            if(is_cut) val += edges[j].w;
        }

        return val;
    }


    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) override
    {
        std::vector<int64_t> perm_lookup(n);
        _Pragma("omp parallel for")
        for(int64_t i = 0; i < n; i++) {
            perm_lookup[perm[i]] = i;
        }

        x.set_all(0.0);

        //Iterate over every  edge.
        #pragma omp parallel for
        for(auto it = edges.begin(); it < edges.end(); it++) {
            auto & e = *it;

            //There is a gain when the first vertex is added,
            //and a loss when the last vertex is added
            int64_t first = e.v[0];
            int64_t last = e.v[0];
            for(auto v : e.v) {
                if(perm_lookup[v] <= perm_lookup[first]) first = v;
                if(perm_lookup[v] >= perm_lookup[last]) last = v;
            }
            
            #pragma omp atomic
            x(first) += e.w;

            #pragma omp atomic
            x(last) -= e.w;
        }
    }
};

#endif
