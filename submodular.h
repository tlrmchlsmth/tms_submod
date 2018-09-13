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

#include <omp.h>

template<class DT>
class SubmodularFunction {
public:
    //Workspace for the greedy algorithm
    std::unordered_set<int64_t> A;
    std::vector<int64_t> permutation;
    int64_t n;

    SubmodularFunction(int64_t n_in) : n(n_in)
    {
        A.reserve(n);
        permutation.reserve(n);
        for(int i = 0; i < n; i++) 
            permutation.push_back(i);
    }
    virtual void initialize_default(){}

    virtual DT eval(const std::unordered_set<int64_t>& A) = 0;
    virtual std::unordered_set<int64_t> get_set() const = 0;

    virtual DT marginal_gain(const std::unordered_set<int64_t>& A, DT FA, int64_t b) {
        std::unordered_set<int64_t> Ab = A;
        Ab.insert(b);
        DT FAb = this->eval(Ab);
        return FAb - FA;
    }

    virtual void marginal_gains(const std::vector<int64_t>& perm, Vector<DT>& p) 
    {
        A.clear();
        DT FA_old = 0.0;
        for(int i = 0; i < p.length(); i++) {
            DT FA = marginal_gain(A, FA_old, perm[i]);
            p(perm[i]) = FA - FA_old;
            A.insert(perm[i]);
            FA_old = FA;
        }
    }

    void polyhedron_greedy(double alpha, const Vector<DT>& x, Vector<DT>& p, PerfLog* perf_log) 
    {
        int64_t start_a = rdtsc();
        //sort x
        if (alpha > 0.0) {
            std::stable_sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) > x(b); } );
        } else if (alpha < 0.0) {
            std::stable_sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) < x(b); } );
        }
        if(perf_log) {
            perf_log->log_total("SORT TIME", rdtsc() - start_a);
        }

        int64_t start_b = rdtsc();
        marginal_gains(permutation, p);
        if(perf_log) {
            perf_log->log_total("MARGINAL GAIN TIME", rdtsc() - start_b);
            perf_log->log_total("GREEDY TIME", rdtsc() - start_a);
        }
    }

    double polyhedron_greedy_eval(double alpha, const Vector<DT>& x, Vector<DT>& p, PerfLog* perf_log) 
    {
        int64_t start_a = rdtsc();
        //sort x
        if (alpha > 0.0) {
            std::stable_sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) > x(b); } );
        } else if (alpha < 0.0) {
            std::stable_sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b){ return x(a) < x(b); } );
        }
        if(perf_log) {
            perf_log->log_total("SORT TIME", rdtsc() - start_a);
        }

        int64_t start_b = rdtsc();
        marginal_gains(permutation, p);
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

//m: # of states. n: number of variables
template<class DT>
class LogDet : public SubmodularFunction<DT> {
public:
    int64_t n;

    Matrix<DT> Cov;     //Covariance matrix
    Matrix<DT> U;       //Cholesky factorization of the covariance matrix of S
   // DT baseline;        //Log determinant of covariance matrix

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

    DT eval(const std::unordered_set<int64_t>& A) {
        if(A.size() == 0) return 0.0;

        // Select some rows and columns and perform Cholesky factorization
        auto Ua = U.submatrix(0,0,A.size(),A.size());
        Ua.set_all(0.0);
        int64_t kaj = 0;
        for(int64_t j = 0; j < n; j++) {
            if(A.count(j) == 0) continue;
            int64_t kai = 0;
            for(int64_t i = 0; i <= j; i++) {
                if(A.count(i) == 0) continue;
                Ua(kai, kaj) = Cov(i,j);
                kai++;
            }
            kaj ++;
        }
        Ua.chol('U');
        DT log_det = log_determinant(Ua);

        return log_det;
    }

    std::unordered_set<int64_t> get_set() const {
        std::unordered_set<int64_t> V;
        V.reserve(n);
        for(int i = 0; i < n; i++) 
            V.insert(i);
        return V;
    }

    void marginal_gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
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

    void marginal_gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
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

#if 0
//m: # of states. n: number of variables
template<class DT>
class LogDetSymm : public SubmodularFunction<DT> {
public:
    int64_t n;

    Matrix<DT> cov;     //Covariance matrix
    Matrix<DT> KV;      //Cholesky factorization of the covariance matrix of S
    DT baseline;     //Log determinant of KV

    DT log_determinant(Matrix<DT>& A)
    {
        DT val = 0.0;
        for(int64_t i = 0; i < std::min(A.height(), A.width()); i++) {
            val += log(A(i,i)*A(i,i));
        }
        return val;
    }

    LogDetSymm(int64_t n_in) : SubmodularFunction<DT>(n_in),
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

        baseline = log_determinant(KV);

        //Our orthogonal transformation is a single reflection
        /*
        Vector<DT> v(n);
        v.fill_rand(gen, dist);
        v.scale(1.0 / v.norm2());
        v.house_apply(v(0), cov);
        auto cov_T = cov.transposed();
        v.house_apply(v(0), cov_T);
        */

        //Check 
        
        //Get cholesky factorization of the covariance matrix
        KV.copy(cov);
        KV.chol('U');
        KV.set_subdiagonal(0.0);
        DT baseline2 = log_determinant(KV);
        std::cout << baseline << "\t" << baseline2 << std::endl;

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
    }

    std::unordered_set<int64_t> get_set() const {
        std::unordered_set<int64_t> V;
        V.reserve(n);
        for(int i = 0; i < n; i++) 
            V.insert(i);
        return V;
    }

/*
    //incremental version of polyhedron greedy
    //Slow, uses l2 blas and here for posterity only
    void polyhedron_greedy_inc(double alpha, const Vector<DT>& x, Vector<DT>& x, PerfLog* perf_log) 
    {
        //sort x
        int64_t start_a = rdtsc();
        if (alpha > 0.0) std::sort(SubmodularFunction<DT>::permutation.begin(), SubmodularFunction<DT>::permutation.end(), 
                [&](int64_t a, int64_t b){ return x(a) > x(b); } );
        else std::sort(SubmodularFunction<DT>::permutation.begin(), SubmodularFunction<DT>::permutation.end(), 
                [&](int64_t a, int64_t b){ return x(a) < x(b); } );
        if(perf_log) perf_log->log_total("SORT TIME", rdtsc() - start_a);

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
        x(SubmodularFunction<DT>::permutation[0]) = FA_old;

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
            x(cov_i) = FA - FA_old;

            FA_old = FA;
        }

        if(perf_log) {
            perf_log->log_total("GREEDY TIME", rdtsc() - start_a);
            perf_log->log_total("MARGINAL GAIN TIME", rdtsc() - start_b);
        }
    }*/

    void marginal_gains(std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        //Permute rows and columns of covariance matrix
        auto Ua = U.submatrix(0,0,n,n);
        Ua.copy_permute_rc(cov, perm);
        
        //Do cholestky factorization of permuted matrix
        Ua.chol('U');
        for(int64_t i = 0; i < n; i++) {
            x(perm[i]) = log(Ua(i,i) * Ua(i,i));
        }

        //Reverse permutation of rows and columns of covariance matrix
        auto rev = perm;
        std::reverse(rev.begin(), rev.end());
        Ua.copy_permute_rc(cov, rev);

        //Do cholesky factorization of reverse permuted matrix
        Ua.chol('U');
        for(int64_t i = 0; i < n; i++) {
            x(perm[i]) -= log(Ua(n-i-1, n-i-1)* Ua(n-i-1, n-i-1));
        }
    }
};
#endif

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

    DT eval(const std::unordered_set<int64_t>& A) 
    {
        DT val = 0.0;
        for(auto a : A) {
            for(auto b : adj_out[a]) {
                if(A.count(b.index) == 0)
                    val += b.weight;
            }
        }
        for(auto b : adj_out[n]) {
            if(A.count(b.index) == 0)
                val += b.weight;
        }

        return val - baseline;
    }

    DT marginal_gain(const std::unordered_set<int64_t>& A, DT FA, int64_t b) 
    {
        //Gain from adding b
        DT gain = 0.0;
        for(int64_t i = 0; i < adj_out[b].size(); i++) {
            if(A.count(adj_out[b][i].index) == 0)
                gain += adj_out[b][i].weight;
        }

        //Loss from adding b
        DT loss = 0.0;
        for(int64_t i = 0; i < adj_in[b].size(); i++) {
            if(adj_in[b][i].index == n || A.count(adj_in[b][i].index) != 0)
                loss -= adj_in[b][i].weight;
        }

        return gain + loss;
    }

    virtual void marginal_gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        std::vector<int64_t> perm_lookup(n);
        _Pragma("omp parallel for")
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

    std::unordered_set<int64_t> get_set() const 
    {
        std::unordered_set<int64_t> V;
        V.reserve(n);
        for(int i = 0; i < n; i++) 
            V.insert(i);
        return V;
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

    DT eval(const std::unordered_set<int64_t>& A) 
    {
        DT val = 0.0;
        for(auto a : A) {
            for(auto b : adj_out[a]) {
                if(A.count(b.index) == 0)
                    val += b.weight;
            }
        }
        for(auto b : adj_out[n]) {
            if(A.count(b.index) == 0)
                val += b.weight;
        }

        return val - baseline;
    }

    void marginal_gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        _Pragma("omp parallel") 
        {

            int64_t t_id = omp_get_thread_num();
            int64_t nt = omp_get_num_threads();

            int64_t n_per_thread = (n - 1) / nt + 1;
            int64_t start = n_per_thread * t_id;
            int64_t end = std::min(start + n_per_thread, n);

            //Each thread must maintain its own set A
            std::unordered_set<int64_t> A;
            A.reserve(perm.size());
            for(int64_t i = 0; i < std::min(start, n); i++) A.insert(perm[i]);

            for(int64_t j = start; j < end; j++) {
                int64_t b = perm[j];

                //Gain from adding b
                DT gain = 0.0;
                for(int64_t i = 0; i < adj_out[b].size(); i++) {
                    if(A.count(adj_out[b][i].index) == 0)
                        gain += adj_out[b][i].weight;
                }

                //Loss from adding b
                DT loss = 0.0;
                for(int64_t i = 0; i < adj_in[b].size(); i++) {
                    if(adj_in[b][i].index == n || A.count(adj_in[b][i].index) != 0)
                        loss -= adj_in[b][i].weight;
                }

                x(b) = gain + loss;
                A.insert(b);
            }
        }
    }

    std::unordered_set<int64_t> get_set() const 
    {
        std::unordered_set<int64_t> V;
        V.reserve(n);
        for(int i = 0; i < n; i++) 
            V.insert(i);
        return V;
    }
};


#endif
