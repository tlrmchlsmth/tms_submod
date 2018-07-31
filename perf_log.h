#ifndef PERF_LOG_H
#define PERF_LOG_H

#include <vector>
#include <map>
#include <string>

//In the future we may add histogram/list support
class PerfTotal {
public:
    int64_t total;
    int64_t count;

    PerfTotal() : total(0), count(0) { }
    inline void log(int64_t x) {
        total += x;
        count++;
    }
};

class PerfLog {
public:
    std::map<std::string, PerfTotal> tot_classes;

    PerfLog(){}

    void add_histogram(std::string s) {
        tot_classes.emplace(s,PerfTotal());
    }

    void log(std::string s, int64_t x) {
        tot_classes[s].log(x);
    }

    int64_t get_total(std::string s) const {
        return tot_classes.at(s).total;
    }

    double get_total(std::string s, double mult) const {
        return tot_classes.at(s).total;
    }

    void print(std::string tag) {
        for(auto a : tot_classes) {
            if(a.first.find(tag) != std::string::npos) {
                std::cout << a.first << " : " << a.second.count << " : " << a.second.total << " : " << (double) a.second.total / (double) a.second.count << std::endl;
            }
        }
    }

    void print(std::string tag, int64_t total) {
        for(auto a : tot_classes) {
            if(a.first.find(tag) != std::string::npos) {
                std::cout << a.first << " : " << a.second.count  << " : " << a.second.total << " : " << (double) a.second.total / (double) a.second.count  << " : " << (double) a.second.total / (double) total << std::endl;
            }
        }
    }
};

#endif
