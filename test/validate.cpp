#include <iostream>
#include <random>
#include <list>
#include <set>

#include "../vector.h"
#include "../matrix.h"
#include "../minimizer.h"
#include "../util.h"

#include <lemon/list_graph.h>
#include <lemon/preflow.h>
using namespace lemon;

//#define DEBUGGING

void val_incremental_qr_remove_cols()
{
    int64_t start = 128;
    int64_t end = 1024;
    int64_t inc = 128; 
    double percent_to_remove = 0.2; 


    std::cout << "===========================================================" << std::endl;
    std::cout << "Validating incremental QR Column Removal" << std::endl;
    std::cout << "===========================================================" << std::endl;
    int w = 18;
    std::cout << std::setw(w) << "m" << std::setw(w) << "n" << std::setw(w) << "error1" << std::setw(w) << "error2" << std::setw(w) << "error3" << std::setw(w) << "error4" << std::endl;
    for(int64_t n = start; n <= end; n += inc) {
        int64_t m = n;
        int64_t task_size = 16;
        int64_t nb = 32;

        std::vector<double> cycles;
        std::list<int64_t> cols_to_remove = get_cols_to_remove(n, percent_to_remove);

        //1. Create random S
        Matrix<double> S(m,n);
        S.fill_rand();

        //2. Perform a QR factorization of S
        Matrix<double> R(m,n);
        Matrix<double> R1(m,n); Matrix<double> R2(m,n); Matrix<double> R3(m,n); Matrix<double> R4(m,n);

        Vector<double> t(n);
        Matrix<double> T(nb, n);
        Matrix<double> ws(nb, n);
        Matrix<double> V_ws(cols_to_remove.size(), n);
        R.copy(S);
        R.qr(t);
        R.set_subdiagonal(0.0);
        R1.copy(R); R2.copy(R); R3.copy(R); R4.copy(R);
        
        //3. Delete Cols.
        S.remove_cols(cols_to_remove);
        R1.remove_cols_incremental_qr_householder(cols_to_remove, t);
        R2.remove_cols_incremental_qr_blocked_householder(cols_to_remove, t, nb);
        R.remove_cols_incremental_qr_kressner(R3, cols_to_remove, T, V_ws, nb, ws);
        R.remove_cols_incremental_qr_tasks_kressner(R4, cols_to_remove, T, V_ws, task_size, nb, ws);
        R1.set_subdiagonal(0.0); R2.set_subdiagonal(0.0); R3.set_subdiagonal(0.0); R4.set_subdiagonal(0.0);
        
        //4. Do some checksum MVMs
        Vector<double> y(S.width());
        y.fill_rand();
        Vector<double> Sy(S.height());
        Vector<double> STSy(S.width());
        auto ST = S.transposed();
        S.mvm(1.0, y, 0.0, Sy); 
        ST.mvm(1.0, Sy, 0.0, STSy); 

        R1.set_subdiagonal(0.0);
        Vector<double> R1y(R1.height());
        Vector<double> R1TR1y(R1.width());
        auto R1T = R1.transposed();
        R1.mvm(1.0, y, 0.0, R1y); 
        R1T.mvm(1.0, R1y, 0.0, R1TR1y); 
        R1TR1y.axpy(-1.0, STSy);
        double error1 = R1TR1y.norm2();

        R2.set_subdiagonal(0.0);
        Vector<double> R2y(R2.height());
        Vector<double> R2TR2y(R2.width());
        auto R2T = R2.transposed();
        R2.mvm(1.0, y, 0.0, R2y); 
        R2T.mvm(1.0, R2y, 0.0, R2TR2y); 
        R2TR2y.axpy(-1.0, STSy);
        double error2 = R2TR2y.norm2();

        R3.set_subdiagonal(0.0);
        Vector<double> R3y(R3.height());
        Vector<double> R3TR3y(R3.width());
        auto R3T = R3.transposed();
        R3.mvm(1.0, y, 0.0, R3y); 
        R3T.mvm(1.0, R3y, 0.0, R3TR3y); 
        R3TR3y.axpy(-1.0, STSy);
        double error3 = R3TR3y.norm2();
        
        R4.set_subdiagonal(0.0);
        Vector<double> R4y(R4.height());
        Vector<double> R4TR4y(R4.width());
        auto R4T = R4.transposed();
        R4.mvm(1.0, y, 0.0, R4y); 
        R4T.mvm(1.0, R4y, 0.0, R4TR4y); 
        R4TR4y.axpy(-1.0, STSy);
        double error4 = R4TR4y.norm2();
        
        std::cout << std::setw(w) << m << std::setw(w) << n;
        print_err(error1, w); print_err(error2, w); print_err(error3, w); print_err(error4, w);
        std::cout << std::endl;
#ifdef DEBUGGING        
        exit(1);
#endif
    }

}

void val_mincut()
{
    int64_t start = 8;
    int64_t end = 128;
    int64_t inc = 8; 

    std::cout << "===========================================================" << std::endl;
    std::cout << "Validating mincut max flow" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int w = 18;
    std::cout << std::setw(w) << "n" << std::setw(w) << "MNP solution" << std::setw(w) << "Lemon solution" << std::setw(w) << "Difference" << std::endl;
    for(int64_t n = start; n <= end; n += inc) {
        //Initialize min cut problem
//        MinCut<double> problem(n, 16, 0.5, 0.05);
        MinCut<double> problem(n);
        problem.WattsStrogatz(16, 0.25);
        
        //Solve problem via min norm point
        MinNormPoint<double> mnp;
        auto A = mnp.minimize(problem, 1e-10, 1e-6, false, NULL);
        double mnp_sol = problem.eval(A) + problem.baseline;
        
        //Solve problem with lemon
        ListDigraph g;
        ListDigraph::ArcMap<double> capacity(g);

        //Create nodes and lookup table for interior nodes.
        std::vector<int> node_ids;
        node_ids.reserve(n);
        for(int i = 0; i < n; i++) {
            auto node = g.addNode();
            node_ids.push_back(g.id(node));
        }
        ListDigraph::Node source = g.addNode();
        ListDigraph::Node sink = g.addNode();

        //Add edges from outgoing adjacency lists
        for(int i = 0; i < n; i++) {
            for(auto b : problem.adj_out[i]) {
                assert(b.index != n);
                if(b.index == n+1) {
                    //Edge to sink
                    auto arc = g.addArc(g.nodeFromId(node_ids[i]), sink);
                    capacity[arc] = b.weight;
                } else {
                    //Edge to interior node
                    auto arc = g.addArc(g.nodeFromId(node_ids[i]), g.nodeFromId(node_ids[b.index]));
                    capacity[arc] = b.weight;
                }
            }
        }

        //Add edges from source node
        for(auto b : problem.adj_out[n]) {
            if(b.index == n+1) {
                //Edge to sink
                auto arc = g.addArc(source, sink);
                capacity[arc] = b.weight;
            } else {
                //Edge to interior node
                auto arc = g.addArc(source, g.nodeFromId(node_ids[b.index]));
                capacity[arc] = b.weight;
            }
        }

        Preflow<ListDigraph, ListDigraph::ArcMap<double>> lemon_prob(g, capacity, source, sink);
        lemon_prob.run();
        double lemon_sol = lemon_prob.flowValue();
        

        std::cout << std::setw(w) << n << std::setw(w) << mnp_sol << std::setw(w) << lemon_sol;
        print_err(mnp_sol - lemon_sol, w);
        std::cout << std::endl;
#ifdef DEBUGGING        
        exit(1);
#endif
    }

}

void run_validation_suite() 
{
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    std::cout << "Running validation tests." << std::endl;
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    val_incremental_qr_remove_cols();
    val_mincut();
}
