#include <iostream>
#include <random>
#include <list>
#include <set>

#include "../../vector.h"
#include "../../matrix.h"

//#define DEBUGGING

template<class RNG, class DIST>
std::list<int64_t> get_cols_to_remove(int64_t m, double percent_to_remove, RNG &gen, DIST& dist)
{
    int64_t n_cols_to_remove = std::round(m * percent_to_remove);
    std::set<int64_t> cols_to_remove;

    while(cols_to_remove.size() < n_cols_to_remove) {
        cols_to_remove.insert(dist(gen));
    }   

    std::list<int64_t> to_ret;
    for(auto it = cols_to_remove.begin(); it != cols_to_remove.end(); ++it) {
        to_ret.push_back(*it);
    }   

    return to_ret;
}

void val_incremental_qr_remove_cols()
{
    int64_t start = 128;
    int64_t end = 1024;
    int64_t inc = 128; 
    double percent_to_remove = 0.2; 

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal(0.0, 10);

    std::cout << "================================================================================" << std::endl;
    std::cout << "Validating incremental QR Column Removal" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "m\tn\terror" << std::endl;
    for(int64_t n = start; n <= end; n += inc) {
        int64_t m = n;
        int64_t task_size = 16;
        int64_t nb = 4;

        std::uniform_int_distribution<> dist(0,n-1);
        std::vector<double> cycles;

        std::list<int64_t> cols_to_remove = get_cols_to_remove(n, percent_to_remove, gen, dist);

#ifdef DEBUGGING        
        std::cout << "Removing columns ";
        for(auto a : cols_to_remove) {
            std::cout << a << ", ";
        }
        std::cout << std::endl;
#endif

        //1. Create random S
        Matrix<double> S(m,n);
        S.fill_rand(gen, normal);

        //2. Perform a QR factorization of S
        Matrix<double> R(m,n);
        Matrix<double> Rinit(m,n);
        Matrix<double> Runb(m,n);
        Vector<double> t(n);
        Matrix<double> T(nb, n);
        Matrix<double> ws(nb, n);
        Matrix<double> V_ws(cols_to_remove.size(), n);
        R.copy(S);
        R.qr(t);
        R.set_subdiagonal(0.0);
        Runb.copy(R);
        Rinit.copy(R);
        
        //3. Delete Cols.
        S.remove_cols(cols_to_remove);
//        R.blocked_remove_cols_incremental_qr(cols_to_remove, t, nb);
        Rinit.blocked_kressner_remove_cols_incremental_qr(R, cols_to_remove, T, V_ws, task_size, nb, ws);
//        Runb.kressner_remove_cols_incremental_qr(cols_to_remove, T, V_ws, nb, ws);
        R.set_subdiagonal(0.0);
//        Runb.set_subdiagonal(0.0);
 
  //     Uncomment these, and comment out qr and updates to see if kressner remove cols_incremental_qr is shifting properly.
/*
        std::cout << "*********" << std::endl;
        Rinit.print();
        std::cout << "*********" << std::endl;
        Rinit.remove_cols(cols_to_remove);
        Rinit.transpose(); Rinit.remove_cols(cols_to_remove); Rinit.transpose();

        std::cout << "*********" << std::endl;
        Rinit.print();
        std::cout << "*********" << std::endl;

        std::cout << "*********" << std::endl;
        D.print();
        std::cout << "*********" << std::endl;

        std::cout << "*********" << std::endl;
        Rinit.axpby(1.0, D, -1.0);
        Rinit.print();
        std::cout << "*********" << std::endl;

*/
        //4. Do some checksum MVMs
        Vector<double> y(S.width());
        y.fill_rand(gen, normal);
        Vector<double> Sy(S.height());
        Vector<double> STSy(S.width());
        auto ST = S.transposed();
        S.mvm(1.0, y, 0.0, Sy); 
        ST.mvm(1.0, Sy, 0.0, STSy); 

        R.set_subdiagonal(0.0);
        Vector<double> Ry(R.height());
        Vector<double> RTRy(R.width());
        auto RT = R.transposed();
        R.mvm(1.0, y, 0.0, Ry); 
        RT.mvm(1.0, Ry, 0.0, RTRy); 

        STSy.axpy(-1.0, RTRy);
        double error = STSy.norm2();
        
        //Print off error 
       /* 
        Matrix<double> STS(S.width(),S.width());
        std::cout << "S m " << S.height() << " n " << S.width() << std::endl;
        STS.mmm(1.0, ST, S, 0.0);
        std::cout<< std::endl;
        STS.print();
        std::cout<< std::endl;

        Matrix<double> RTR(R.width(),R.width());
        RTR.mmm(1.0, RT, R, 0.0);
        std::cout<< std::endl;
        std::cout << "R m " << R.height() << " n " << R.width() << std::endl;
        RTR.print();
        std::cout<< std::endl;
     
        std::cout << "DIFF m " << R.height() << " n " << R.width() << std::endl;
        STS.axpby(1.0, RTR, -1.0);
        STS.print();
        std::cout<< std::endl;
*/
        std::cout << m << "\t" << n << "\t" << error << std::endl;
#ifdef DEBUGGING        
        exit(1);
#endif
    }

}

void run_validation_suite() 
{
    val_incremental_qr_remove_cols();
}
