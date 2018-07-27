#ifndef PERF_LOG_H
#define PERF_LOG_H

#include <vector>
#include <map>
#include <string>

//Performance log. It's a set of histograms
class PerfHist {
public:
    int64_t total;
    std::vector<int64_t> buckets;
    const int64_t bucket_size;

    PerfHist(int64_t n_buckets, int64_t bs) : bucket_size(bs), buckets(n_buckets), total(0){
        for(int64_t i = 0; i < n_buckets; i++)
            buckets[i] = 0;
    }

    void log(int64_t x) {
        total += x;
        int64_t bucket = x / bucket_size;
        buckets[bucket]++;
    }
};

class PerfLog {
public:
    std::map<std::string, PerfHist> hists;

    PerfLog(){}

    void add_histogram(std::string s, int64_t n_buckets, int64_t bucket_size) {
        hists.emplace(s,PerfHist(n_buckets, bucket_size));
    }

    void log(std::string s, int64_t x) {
        hists.at(s).log(x);
    }

    int64_t get_total(std::string s) const {
        return hists.at(s).total;
    }

    double get_total(std::string s, double mult) const {
        return mult * hists.at(s).total;
    }

    void print_histogram(std::string s) {
        for(int i = 0; i < hists.at(s).buckets.size(); i++) {
            std::cout << std::setw(15) << i * hists.at(s).bucket_size;
        }
        std::cout << std::endl;
        for(int i = 0; i < hists.at(s).buckets.size(); i++) {
            std::cout << std::setw(15) << hists.at(s).buckets[i];
        }
        std::cout << std::endl;
    }

    void print_histogram(std::string s, double mult) {
        for(int i = 0; i < hists.at(s).buckets.size(); i++) {
            std::cout << std::setw(15) << i * hists.at(s).bucket_size;
        }
        std::cout << std::endl;
        for(int i = 0; i < hists.at(s).buckets.size(); i++) {
            std::cout << std::setw(15) << mult * hists.at(s).buckets[i];
        }
        std::cout << std::endl;
    }
};

#endif
