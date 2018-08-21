#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>

#include "vector.h"
#include "matrix.h"
#include "perf_log.h"


template<class DT>
class SubmodularFunction {
public:
    //Workspace for the greedy algorithm
    std::unordered_set<int64_t> A;
    std::vector<int64_t> permutation;

    SubmodularFunction(int64_t n) 
    {
        A.reserve(n);
        permutation.reserve(n);
        for(int i = 0; i < n; i++) 
            permutation.push_back(i);
    }
    virtual DT eval(const std::unordered_set<int64_t>& A) = 0;
    virtual std::unordered_set<int64_t> get_set() const = 0;
    virtual DT eval(const std::unordered_set<int64_t>& A, DT FA, int64_t b) {
        std::unordered_set<int64_t> Ab = A;
        Ab.insert(b);
        DT FAb = this->eval(Ab);
        return FAb;
    }

    virtual void polyhedron_greedy(double alpha, const Vector<DT>& weights, Vector<DT>& xout, PerfLog* plog) 
    {
        int64_t start_a = rdtsc();
        //sort weights
        if (alpha > 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) > weights(b); } );
        } else if (alpha < 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return weights(a) < weights(b); } );
        }
        if(plog) {
            plog->log("SORT TIME", rdtsc() - start_a);
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
        if(plog) {
            plog->log("MARGINAL GAIN TIME", rdtsc() - start_b);
            plog->log("GREEDY TIME", rdtsc() - start_a);
        }
    }
};

//m: # of states. n: number of variables
template<class DT>
class LogDet : public SubmodularFunction<DT> {
public:
    int64_t n;

    Matrix<DT> cov;     //Covariance matrix

    Matrix<DT> KV;      //Cholesky factorization of the covariance matrix of S
//    Matrix<DT> KA_base;      //Workspace for the cholesky factorization of the covariance matrix of the random variables in set A
//    Matrix<DT> KAc_base;     //Workspace for the cholesky factorization of the covariance matrix of the random variables not in set A

    DT baseline;     //Log determinant of KV

//    Matrix<DT> T_ws;
//    Matrix<DT> V_ws;
//    Matrix<DT> ws;

    DT log_determinant(Matrix<DT>& A)
    {
        DT val = 0.0;
        for(int64_t i = 0; i < std::min(A.height(), A.width()); i++) {
            val += log(A(i,i)*A(i,i));
        }
        return val;
    }

    LogDet(int64_t n_in) : SubmodularFunction<DT>(n_in),
                                         n(n_in), cov(n_in, n_in), KV(n_in, n_in),
                                         baseline(0.0)
    {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> diag_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> dist(-.01, .01);

        //Create a covariance matrix. 
        //Every element must be positive and it must be symmetric positive definite
        //
        //We accomplish this by:
        //  1. Start with a random upper triangular matrix
        //  2. Multiply it with its transpose
        //  2. Hit it with an orthogonal similarity transformation
        //
        KV.fill_rand(gen, dist);
        for(int i = 0; i < n; i++) {
            KV(i,i) = diag_dist(gen);
        }
        KV.set_subdiagonal(0.0);
        auto KVT = KV.transposed();
        cov.mmm(1.0, KVT, KV, 0.0);

        //Our orthogonal transformation is a single reflection
        /*for(int i = 0; i < 1; i++) {
            Vector<DT> v(n);
            v.fill_rand(gen, dist);
            v.scale(1.0 / v.norm2());
            v.house_apply(v(0), cov);
            auto cov_T = cov.transposed();
            v.house_apply(v(0), cov_T);
        }*/

        //Get cholesky factorization of the covariance matrix
        
//        KV.copy(cov);
//        KV.chol('U');
//        KV.set_subdiagonal(0.0);

        //Doublecheck
        Vector<double> y(n_in);
        y.fill_rand();
        Vector<double> cov_y(n_in);
        cov.mvm(1.0, y, 0.0, cov_y); 
        auto KT = KV.transposed();
        Vector<double> Ky(n_in);
        Vector<double> KTKy(n_in);
        KV.mvm(1.0, y, 0.0, Ky); 
        KT.mvm(1.0, Ky, 0.0, KTKy);
        KTKy.axpy(-1.0, cov_y);
//        std::cout << "KV" << std::endl;
//        KV.print(); std::cout << std::endl;
        assert(KTKy.norm2() < 1e-8);
       
        //Check 
        baseline = log_determinant(KV);
    }

    DT eval(const std::unordered_set<int64_t>& A) {
        if(A.size() == 0 || A.size() == n) return 0.0; //Fast-path so stuff doesn't break with empty matrices

        //Copy rows and columns in A into KA
//        auto KA = KA_base.submatrix(0,0,A.size(),A.size());
        auto KA = KV.submatrix(0,0,A.size(),A.size());
        KA.set_all(0.0);
        int64_t kaj = 0;
        for(int64_t j = 0; j < n; j++) {
            if(A.count(j) == 0) continue;
            int64_t kai = 0;
            for(int64_t i = 0; i <= j; i++) {
                if(A.count(i) == 0) continue;
                KA(kai, kaj) = cov(i,j);
                kai++;
            }
            kaj ++;
        }
        KA.chol('U');
        DT log_det_ka = log_determinant(KA);

        auto KAc = KV.submatrix(0,0,n-A.size(),n-A.size());
        KAc.set_all(0.0);
        kaj = 0;
        for(int64_t j = 0; j < n; j++) {
            if(A.count(j) > 0) continue;
            int64_t kai = 0;
            for(int64_t i = 0; i <= j; i++) {
                if(A.count(i) > 0) continue;
                KAc(kai, kaj) = cov(i,j);
                kai++;
            }
            kaj ++;
        }
        KAc.chol('U');
        DT log_det_kac = log_determinant(KAc);
        return log_det_ka + log_det_kac - baseline;

        //TODO: probably there is a crossover point where the size of A is small where the following is beneficial
/*
        std::list<int64_t> cols_to_remove_KA;
        std::list<int64_t> cols_to_remove_KAc;
        for(int64_t i = 0; i < n; i++) {
            if(A.count(i) == 0) cols_to_remove_KA.push_back(i);
            else cols_to_remove_KAc.push_back(i);
        }

        KV.remove_cols_incremental_qr_tasks_kressner(KA, cols_to_remove_KA, T_ws, V_ws, 64, 32, ws);
        auto KAc = KAc_base.submatrix(0,0,n,n);
        KV.remove_cols_incremental_qr_tasks_kressner(KAc, cols_to_remove_KAc, T_ws, V_ws, 64, 32, ws);
        return log_determinant(KA) + log_determinant(KAc) - baseline;
        */
    }

    std::unordered_set<int64_t> get_set() const {
        std::unordered_set<int64_t> V;
        V.reserve(n);
        for(int i = 0; i < n; i++) 
            V.insert(i);
        return V;
    }


    //incremental version of polyhedron greedy
    //Slow, uses l2 blas and here for posterity only
    void polyhedron_greedy_inc(double alpha, const Vector<DT>& weights, Vector<DT>& xout, PerfLog* plog) 
    {
        //sort weights
        int64_t start_a = rdtsc();
        if (alpha > 0.0) std::sort(SubmodularFunction<DT>::permutation.begin(), SubmodularFunction<DT>::permutation.end(), 
                [&](int64_t a, int64_t b){ return weights(a) > weights(b); } );
        else std::sort(SubmodularFunction<DT>::permutation.begin(), SubmodularFunction<DT>::permutation.end(), 
                [&](int64_t a, int64_t b){ return weights(a) < weights(b); } );
        if(plog) plog->log("SORT TIME", rdtsc() - start_a);

        for(auto a: SubmodularFunction<DT>::permutation) {
            std::cout << a << ", ";
        } std::cout << std::endl;

        int64_t start_b = rdtsc();
       
        //Get initial KA 
        Vector<DT> t_a(n);
        auto KA = KV.submatrix(0,0,1,1);
        KA(0,0) = std::sqrt(cov(SubmodularFunction<DT>::permutation[0], SubmodularFunction<DT>::permutation[0]));

        //Get initial KAc 
        Matrix<DT> T_ac(2, n);
        auto KAc = KAc_base.submatrix(0,0,n,n);
        KAc.copy(KV);
        KAc.remove_column_iqr_givens(SubmodularFunction<DT>::permutation[0], T_ac, 64);

        //Create list of column ids to remove for KAc
        std::vector<int64_t> shifted_cols_to_remove = SubmodularFunction<DT>::permutation;
        for(int64_t i = 0; i < n; i++) {
            int64_t index = shifted_cols_to_remove[i];
            for(int64_t j = i+1; j < n; j++) {
                if(index < shifted_cols_to_remove[j])
                    shifted_cols_to_remove[j] -= 1;
            }
        }

        //Evaluate initial FA
        DT FA_old = log_determinant(KA) + log_determinant(KAc) - baseline;
        xout(SubmodularFunction<DT>::permutation[0]) = FA_old;

        //Evaluate the rest of FAs
        for(int64_t i = 1; i < n-1; i++) {
            //
            //Add column to KA
            //
            auto c1 = KV.subcol(0, i, i);
            int64_t cov_i = SubmodularFunction<DT>::permutation[i]; 
            for(int64_t j = 0; j < i; j++) {
                c1(j) = cov(cov_i, SubmodularFunction<DT>::permutation[j]);
            }
            KA.transpose(); KA.trsv(CblasLower, c1); KA.transpose();
            DT mu = sqrt(std::abs(cov(cov_i, cov_i) - c1.dot(c1)));
            KA.enlarge_n(1); KA.enlarge_m(1);
            KA(i,i) = mu;
            
            // 
            //Remove column from KAc
            //
            KAc.remove_column_iqr_givens(shifted_cols_to_remove[i], T_ac, 64);

            //Evaluate
            DT FA = log_determinant(KA) + log_determinant(KAc) - baseline;
            xout(cov_i) = FA - FA_old;

            FA_old = FA;
        }

        if(plog) {
            plog->log("GREEDY TIME", rdtsc() - start_a);
            plog->log("MARGINAL GAIN TIME", rdtsc() - start_b);
        }
    }

    void polyhedron_greedy(double alpha, const Vector<DT>& weights, Vector<DT>& xout, PerfLog* plog) 
    {
        //sort weights
        int64_t start_a = rdtsc();
        if (alpha > 0.0) std::sort(SubmodularFunction<DT>::permutation.begin(), SubmodularFunction<DT>::permutation.end(), 
                [&](int64_t a, int64_t b){ return weights(a) > weights(b); } );
        else std::sort(SubmodularFunction<DT>::permutation.begin(), SubmodularFunction<DT>::permutation.end(), 
                [&](int64_t a, int64_t b){ return weights(a) < weights(b); } );
        if(plog) plog->log("SORT TIME", rdtsc() - start_a);


        int64_t start_b = rdtsc();

        //Permute rows and columns of covariance matrix
        std::vector<int64_t> perm = SubmodularFunction<DT>::permutation;
        auto LA = KV.submatrix(0,0,n,n);
        LA.copy_permute_rc(cov, perm);
        
        //Do cholestky factorization of forward permuted matrix
        LA.chol('U');
        for(int64_t i = 0; i < n; i++) {
            xout(perm[i]) = log(LA(i,i) * LA(i,i));
        }

        //Reverse permutation of rows and columns of covariance matrix
        auto rev = perm;
        std::reverse(rev.begin(), rev.end());
        LA.copy_permute_rc(cov, rev);

        //Do cholesky factorization of reverse permuted matrix
        LA.chol('U');
        for(int64_t i = 0; i < n; i++) {
            xout(perm[i]) -= log(LA(n-i-1, n-i-1)* LA(n-i-1, n-i-1));
        }

        if(plog) {
            plog->log("GREEDY TIME", rdtsc() - start_a);
            plog->log("MARGINAL GAIN TIME", rdtsc() - start_b);
        }
    }
};

template<class DT>
class IDivSqrtSize : public SubmodularFunction<DT> {
public:
    int64_t size;
    IDivSqrtSize(int64_t n) : size(n), SubmodularFunction<DT>(n) {}

    DT eval(const std::unordered_set<int64_t>& A) {
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
class MinCut : public SubmodularFunction<DT> {
public:
    //Each node has a list of edges in and a list of edges out.
    //Each edge has an index and a weight, and we will have the source and sink node have index n and n+1, respectively.
    std::vector<std::vector<Edge<DT>>> adj_in;
    std::vector<std::vector<Edge<DT>>> adj_out;
    int64_t size;
    DT baseline;

    MinCut(int64_t n) : SubmodularFunction<DT>(n), size(n), baseline(0.0) {}

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
    MinCut(int64_t n, int64_t m,  double cfa, double cfb) : SubmodularFunction<DT>(n), size(n), baseline(0.0) {
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

    DT eval(const std::unordered_set<int64_t>& A) {
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

    DT eval(const std::unordered_set<int64_t>& A, DT FA, int64_t b) {

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
