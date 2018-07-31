#include <set>
#include <list>
#include <random>

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
