#ifndef LOG_DET_H
#define LOG_DET_H

#include "submodular.h"
#include "../la/vector.h"
#include "../la/matrix.h"


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

    DT greedy_maximize(int64_t cardinality_constraint, std::vector<bool> A) {
        auto U0 = LogDet<DT>::U.submatrix(0,0,0,0);
        Matrix<DT> C_base(n,n);
        auto C = C_base.submatrix(0,0,0,n);
        Vector<DT> d_base(n);

        std::list<int64_t> columns;
        for(int64_t j = 0; j < n; j++) {
            columns.push_back(j);
        }

        //Add one element at a time
        for(int64_t i = 0; i < cardinality_constraint && i < n; i++) {
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

            //for(int64_t j = 0; j < d.length(); j++) 
            for(j = 0, cov_j = columns.begin(); j < d.length(); j++, cov_j++) {
                d(j) = sqrt(std::abs(Cov(*cov_j, *cov_j) - d(j)));
            }
            
//            int64_t best_j = d.index_of_max();
            int64_t best_j = 0; auto best_cov_j = columns.begin();
            for(j = 0, cov_j = columns.begin(); j < d.length(); j++, cov_j++) {
                if(d(j) > d(best_j)) {
                    best_j = j;
                    best_cov_j = cov_j; }
            }
            if(d(best_j) >= 0.0) {
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

        return eval(A);
    }

    std::vector<bool> greedy_maximize_trsv() {
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
#endif
