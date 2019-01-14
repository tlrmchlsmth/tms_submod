#ifndef PERF_LOG_H
#define PERF_LOG_H

#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <iostream>

class PerfTotal {
public:
    int64_t total;
    int64_t count;

    PerfTotal() : total(0), count(0) { }
    inline void log_total(int64_t x) {
        total += x;
        count++;
    }
    inline void set_total(int64_t x) {
        total = x;
        count++;
    }
};

class PerfHist {
public:
    std::vector<int64_t> buckets;
    double min;
    double max;
    double bucket_size;

    PerfHist() : min(0), max(100), bucket_size(1), buckets(100){}
    PerfHist(double min, double max, int64_t num_buckets) : min(min), max(max), bucket_size((max - min) / (double) num_buckets), buckets(num_buckets) 
    { 
        std::fill(buckets.begin(), buckets.end(), 0);
    }
    inline void log_hist(double x) {
        int64_t bucket = (x - min) / bucket_size;
        bucket = std::min(bucket, (int64_t)buckets.size() - 1);
        buckets[bucket]++;
    }
    void print(std::string s) {
        for(int i = 0; i < buckets.size(); i++) {
            std::cout << "\033[1;34m" << std::setw(8) << min + bucket_size * i << "\033[0m";
        }
        std::cout << std::endl;
        for(int i = 0; i < buckets.size(); i++) {
            std::cout << "\033[1;34m" << std::setw(8) << buckets[i] << "\033[0m";
        }
        std::cout << std::endl;
    }
};

class PerfLog {
public:
    std::map<std::string, PerfTotal> tot_classes;
    std::map<std::string, PerfHist> hist_classes;

    static PerfLog& get() 
    {
        static PerfLog instance;
        return instance;
    }

    void clear() {
        tot_classes.clear();
        hist_classes.clear();
    }

    void log_total(std::string s, int64_t x) {
        tot_classes[s].log_total(x);
    }
    void set_total(std::string s, int64_t x) {
        tot_classes[s].set_total(x);
    }


    int64_t get_total(std::string s) const {
        if(tot_classes.count(s) == 0)
            return 0;
        return tot_classes.at(s).total;
    }

    int64_t get_count(std::string s) const {
        if(tot_classes.count(s) == 0)
            return 0;
        return tot_classes.at(s).count;
    }


    void add_histogram(std::string s, double min, double max, int num_buckets) {
        hist_classes.emplace(s,PerfHist(min, max, num_buckets));
    }

    void log_hist(std::string s, double x) {
        hist_classes[s].log_hist(x);
    }

    PerfHist get_hist(std::string s) const {
        if(hist_classes.count(s) == 0)
            return PerfHist();

        auto to_ret = hist_classes.at(s);
        return to_ret;
    }

private:
    PerfLog(){}
public:
    PerfLog(PerfLog const&) = delete;
    void operator=(PerfLog const&) = delete;
};

#endif
