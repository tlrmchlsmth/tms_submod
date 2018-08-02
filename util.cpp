#include <set>
#include <list>
#include <random>
#include <iostream>
#include <iomanip>

std::list<int64_t> get_cols_to_remove(int64_t m, double percent_to_remove)
{
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<> dist(0,m-1);

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

void print_err(double err, int64_t w) 
{
    if(std::abs(err) < 1e-6) {
        std::cout << "\033[1;32m" << std::setw(w) << err << "\033[0m";
    } else if(std::abs(err) < 1e-4) {
        std::cout << "\033[1;33m" << std::setw(w) << err << "\033[0m";
    } else {
        std::cout << "\033[1;31m" << std::setw(w) << err << "\033[0m";
    }
}
void print_err(float err, int64_t w) {
    if(std::abs(err) < 1e-4) {
        std::cout << "\033[1;32m" << std::setw(w) << err << "\033[0m";
    } else if(std::abs(err) < 1e-2) {
        std::cout << "\033[1;33m" << std::setw(w) << err << "\033[0m";
    } else {
        std::cout << "\033[1;31m" << std::setw(w) << err << "\033[0m";
    }
}
