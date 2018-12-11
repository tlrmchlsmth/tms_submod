#ifndef TMS_SUBMOD_FN_H
#define TMS_SUBMOD_FN_H

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>

#include "la/vector.h"
#include "la/matrix.h"
#include "perf_log.h"

#include <omp.h>

template<class DT>
class SubmodularFunction {
public:
    //Workspace for the greedy algorithm
    std::vector<bool> A;
    std::vector<int64_t> permutation;
    int64_t n;
    Vector<DT> ws;

    SubmodularFunction(int64_t n_in) : n(n_in), A(n_in), ws(n)
    {
        permutation.reserve(n);
        for(int i = 0; i < n; i++) 
            permutation.push_back(i);
    }
    virtual void initialize_default(){}

    virtual DT eval(const std::vector<bool>& A) = 0;

    std::vector<bool> get_set() const {
        std::vector<bool> V(n);
        std::fill(V.begin(), V.end(), 1);
        return V;
    }

    virtual DT gain(const std::vector<bool>& A, DT FA, int64_t b) {
        std::vector<bool> Ab = A;
        Ab[b] = 1;
        DT FAb = this->eval(Ab);
        return FAb - FA;
    }

    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& p) 
    {
        std::fill(A.begin(), A.end(), 0);
        DT FA_old = 0.0;
        for(int i = 0; i < p.length(); i++) {
            DT FA = gain(A, FA_old, perm[i]);
            p(perm[i]) = FA - FA_old;
            A[perm[i]] = 1;
            FA_old = FA;
        }
    }


    void polyhedron_greedy(double alpha, const Vector<DT>& x, Vector<DT>& p, PerfLog* perf_log) 
    {
        int64_t start_a = rdtsc();
        //sort x
        if (alpha > 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) > x(b); } );
        } else if (alpha < 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) < x(b); } );
        }
        if(perf_log) {
            perf_log->log_total("SORT TIME", rdtsc() - start_a);
        }

        int64_t start_b = rdtsc();
        gains(permutation, p);
        if(perf_log) {
            perf_log->log_total("MARGINAL GAIN TIME", rdtsc() - start_b);
            perf_log->log_total("GREEDY TIME", rdtsc() - start_a);
        }
    }

    double polyhedron_greedy_eval(double alpha, const Vector<DT>& x, Vector<DT>& p, PerfLog* perf_log) 
    {
        int64_t start_a = rdtsc(); //sort x
        if (alpha > 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) > x(b); } );
        } else if (alpha < 0.0) {
            std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) < x(b); } );
        }
        if(perf_log) {
            perf_log->log_total("SORT TIME", rdtsc() - start_a);
        }

        int64_t start_b = rdtsc();
        gains(permutation, p);
        if(perf_log) perf_log->log_total("MARGINAL GAIN TIME", rdtsc() - start_b);
        
        //Get current value of F(A)
        double val = 0.0;
/*      for(int64_t i = 0; i < p.length(); i++) {
            if(x(permutation[i]) >= 0.0) break;
            val += p(permutation[i]);
        }*/
        
        for(int64_t i = 0; i < x.length(); i++) {
            if(x(i) <= 0.0) val += p(i);
        }

        if(perf_log) perf_log->log_total("GREEDY TIME", rdtsc() - start_a);
        return val;
    }
};

template<class DT>
class IwataTest : public SubmodularFunction<DT> {
public:
    int64_t n;
    IwataTest(int64_t n) : SubmodularFunction<DT>(n), n(n) {};
    DT eval(const std::vector<bool>& A) {
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
    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
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

//m: # of states. n: number of variables
template<class DT>
class LogDet : public SubmodularFunction<DT> {
public:
    int64_t n;

    Matrix<DT> Cov;     //Covariance matrix
    Matrix<DT> U;       //Cholesky factorization of the covariance matrix of S

    DT log_determinant(Matrix<DT>& A)
    {
        DT val = 0.0;
        for(int64_t i = 0; i < std::min(A.height(), A.width()); i++) {
            val += log(A(i,i)*A(i,i));
        }
        return val;
    }

    LogDet(int64_t n_in) : SubmodularFunction<DT>(n_in),
                           n(n_in), Cov(n_in, n_in), U(n_in, n_in)
    {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> diag_dist(-2.0, 2.0);
        std::uniform_real_distribution<double> dist(-.01, .01);

        //Create a covariance matrix. 
        //Every element must be positive and it must be symmetric positive definite
        //
        //We accomplish this by:
        //  1. Start with a random upper triangular matrix
        //  2. Multiply it with its transpose
        U.fill_rand(gen, dist);
        for(int i = 0; i < n; i++) {
            U(i,i) = diag_dist(gen);
        }
        U.set_subdiagonal(0.0);
        auto UT = U.transposed();
        Cov.mmm(1.0, UT, U, 0.0);
    }

    std::vector<bool> greedy_maximize2() {
        std::vector<bool> A(n);
        auto U0 = LogDet<DT>::U.submatrix(0,0,0,0);
        Matrix<DT> C_base(n,n);
        auto C = C_base.submatrix(0,0,0,n);
        Vector<DT> d_base(n);

        std::list<int64_t> columns;
        for(int64_t j = 0; j < n; j++) {
            columns.push_back(j);
        }

        //Add one element at a time
        for(int64_t i = 0; i < n; i++) {
            assert(U0.height() == i);
            assert(U0.height() == C.height());
            assert(C.width() == n-i);
            
            auto d = d_base.subvector(0, n-i);
            d.set_all(0.0);
            if(i > 0) {
                auto U1 = U.submatrix(0, i, U0.height(), C.width());
                U1.copy(C);
                U0.transpose(); U0.trsm(CblasLower, CblasLeft, U1); U0.transpose();
                for(int64_t j = 0; j < U1.width(); j++) {
                    auto uj = U1.subcol(j);
                    d(j) = uj.dot(uj);
                }
            }
            int64_t j;
            std::list<int64_t>::iterator cov_j;

            //for(int64_t j = 0; j < d.length(); j++) {
            for(j = 0, cov_j = columns.begin(); j < d.length(); j++, cov_j++) {
                d(j) = sqrt(std::abs(Cov(*cov_j, *cov_j) - d(j)));
            }
            
//            int64_t best_j = d.index_of_max();
            int64_t best_j = 0; auto best_cov_j = columns.begin();
            for(j = 0, cov_j = columns.begin(); j < d.length(); j++, cov_j++) {
                if(d(j) > d(best_j)) {
                    best_j = j;
                    best_cov_j = cov_j;
                }
            }
            if(d(best_j) > 1.0) {
                A[*best_cov_j] = true;
            } else {
                break;
            }

            //Copy best column of U to where it belongs
//            if(best_j != i+1 && i > 0) {
            if(best_j != 0 && i > 0) {
                auto U1 = U.submatrix(0, i, U0.height(), C.width());
                auto to   = U1.subcol(0);
                auto from = U1.subcol(best_j);
                to.copy(from);
            }

            //Enlarge U, remove row,col from C
            U0.enlarge_m(1); U0.enlarge_n(1);
            C.enlarge_m(1);
            assert(U0.height() == C.height());

            U0(i,i) = d(best_j);
            C.remove_col(best_j);
            columns.erase(best_cov_j);
            for(j = 0, cov_j = columns.begin(); j < C.width(); j++, cov_j++) {
                C(i,j) = Cov(*best_cov_j, *cov_j);
            }
        }

        return A;
    }

    std::vector<bool> greedy_maximize1() {
        std::vector<bool> A(n);
        for(int64_t i = 0; i < n; i++) {
            A[i] = false;
        }
        auto U0 = LogDet<DT>::U.submatrix(0,0,0,0);
        Matrix<DT> C_base(n,n);
        Matrix<DT> C = C_base.submatrix(0,0,0,n);

        std::list<int64_t> columns;
        for(int64_t j = 0; j < n; j++) {
            columns.push_back(j);
        }

        //Add one element at a time
        for(int64_t i = 0; i < n; i++) {
            assert(U0.height() == i);
            assert(U0.height() == C.height());

            int64_t best_j = 0; auto best_cov_j = columns.begin();
            DT best_mu = std::numeric_limits<DT>::min();
            int64_t j;
            std::list<int64_t>::iterator cov_j;

            for(j = 0, cov_j = columns.begin(); j < C.width(); j++, cov_j++) {
                DT u1_nrm_sqr = 0.0;
                if(U0.height() > 0) { 
                    auto u1 = U.subcol(0, i+j, U0.height());
                    auto c1 = C.subcol(0, j, U0.height());
                    u1.copy(c1);
                    U0.transpose(); U0.trsv(CblasLower, u1); U0.transpose();
                    u1_nrm_sqr = u1.dot(u1);
                }
                DT mu = sqrt(std::abs(Cov(*cov_j,*cov_j) - u1_nrm_sqr));
                
                if(mu > best_mu) {
                    best_mu = mu;
                    best_j = j;
                    best_cov_j = cov_j;
                }
            }
            
            if(best_mu > 1.0) {
                A[*best_cov_j] = true;
            } else {
                break;
            }

            //Copy best column of U to where it belongs
            if(best_j != 0 && i > 0) {
                auto to   = U.subcol(0, i, U0.height());
                auto from = U.subcol(0, i+best_j, U0.height());
                to.copy(from);
            }

            assert(U0.height() == C.height());
            //Enlarge U, C, remove row,col from C
            U0.enlarge_m(1); U0.enlarge_n(1);
            C.enlarge_m(1);
            assert(U0.height() == C.height());

            U0(i,i) = best_mu;

            C.remove_col(best_j);
            columns.erase(best_cov_j);
            for(j = 0, cov_j = columns.begin(); j < C.width(); j++, cov_j++) {
                C(i,j) = Cov(*best_cov_j, *cov_j);
            }
        }

        return A;
    }

    DT eval(const std::vector<bool>& A) {
        int64_t cardinality = 0;
        for(auto a : A) { if(a) cardinality++; }
        if(cardinality == 0) return 0.0;

        // Select some rows and columns and perform Cholesky factorization
        auto Ua = U.submatrix(0,0,cardinality,cardinality);
        Ua.set_all(0.0);
        int64_t kaj = 0;
        for(int64_t j = 0; j < n; j++) {
            if(!A[j]) continue;
            int64_t kai = 0;
            for(int64_t i = 0; i <= j; i++) {
                if(!A[i]) continue;
                Ua(kai, kaj) = Cov(i,j);
                kai++;
            }
            kaj ++;
        }
        Ua.chol('U');
        DT log_det = log_determinant(Ua);

        return log_det;
    }

    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        //Permute rows and columns of covariance matrix and perform cholesky factorization
        auto Ua = U.submatrix(0,0,n,n);
        Ua.copy_permute_rc(Cov, perm);
        Ua.chol('U');
        for(int64_t i = 0; i < n; i++) {
            x(perm[i]) = log(Ua(i,i) * Ua(i,i));
        }
    }
};

template<class DT>
class SlowLogDet : public LogDet<DT> {
public:
    SlowLogDet(LogDet<DT>& from) : LogDet<DT>(from.n)
    {
        LogDet<DT>::Cov.copy(from.Cov);
    }

    SlowLogDet(int64_t n_in) : LogDet<DT>(n_in) {}

    void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        //Get initial KA 
        auto Ua = LogDet<DT>::U.submatrix(0,0,1,1);
        Ua(0,0) = std::sqrt(LogDet<DT>::Cov(perm[0], perm[0]));

        //Evaluate initial marginal gain 
        x(perm[0]) = log(Ua(0,0) * Ua(0,0));

        //Evaluate the rest of marginal gains, one at a time
        for(int64_t i = 1; i < LogDet<DT>::n; i++) {
            auto c1 = LogDet<DT>::U.subcol(0, i, i);
            for(int64_t j = 0; j < i; j++) {
                c1(j) = LogDet<DT>::Cov(perm[i], perm[j]);
            }
            Ua.transpose(); Ua.trsv(CblasLower, c1); Ua.transpose();
            DT mu = sqrt(std::abs(LogDet<DT>::Cov(perm[i], perm[i]) - c1.dot(c1)));
            Ua.enlarge_n(1); Ua.enlarge_m(1);
            Ua(i,i) = mu;
            
            x(perm[i]) = log(mu*mu);
        }
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
    int64_t n;
    DT baseline;

    MinCut(int64_t n_in) : SubmodularFunction<DT>(n_in), n(n_in), baseline(0.0) {}
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
    void select_source_and_sink() 
    {
        //Select source and sink nodes randomly, but not the nodes at n or n+1,
        //so I don't have to handle the special cases
        std::random_device rd;
        std::mt19937 gen{rd()};
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
        Matrix<double> sums(n+2, 5); sums.set_all(0.0);
        Vector<double> sum_in_a = sums.subcol(1);
        Vector<double> sum_in_b = sums.subcol(2);
        Vector<double> sum_out_a= sums.subcol(3);
        Vector<double> sum_out_b= sums.subcol(4);
        for(int64_t i = 0; i < n+2; i++)
            sums(i, 0) = i;

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
        std::random_device rd;
        std::mt19937 gen{rd()};
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
        this->select_source_and_sink();

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
        std::random_device rd;
        std::mt19937 gen{rd()};
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
        this->select_source_and_sink();

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
