#ifndef PERF_LOG_H
#define PERF_LOG_H

#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <iostream>

#define LOG_FREQ 1

class PerfTotal {
public:
    int64_t total;
    int64_t count;

    PerfTotal() : total(0), count(0) { }
    void log(int64_t x) {
        total += x;
        count++;
    }
    void set(int64_t x) {
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

    PerfHist() : buckets(100), min(0), max(100), bucket_size(1) {}
    PerfHist(double min, double max, int64_t num_buckets) : buckets(num_buckets), min(min), max(max), bucket_size((max - min) / (double) num_buckets)
    { 
        std::fill(buckets.begin(), buckets.end(), 0);
    }
    void log(double x) {
        int64_t bucket = (x - min) / bucket_size;
        bucket = std::min(bucket, (int64_t) buckets.size() - 1);
        buckets[bucket]++;
    }
    void print(std::string s) {
        std::cout << s << std::endl;
        for(uint64_t i = 0; i < buckets.size(); i++) {
            std::cout << "\033[1;34m" << std::setw(8) << min + bucket_size * i << "\033[0m";
        }
        std::cout << std::endl;
        for(uint64_t i = 0; i < buckets.size(); i++) {
            std::cout << "\033[1;34m" << std::setw(8) << buckets[i] << "\033[0m";
        }
        std::cout << std::endl;
    }
};

class PerfSequence {
public:
    std::vector<double> sequence;

    PerfSequence() : sequence() { 
        sequence.reserve(1e9);
    }

    void log(double x) {
        sequence.push_back(x);
    }
};

class PerfLog {
public:
    std::map<std::string, PerfTotal> tot_classes;
    std::map<std::string, PerfHist> hist_classes;
    std::map<std::string, PerfSequence> sequence_classes;

    static PerfLog& get() 
    {
        static PerfLog instance;
        return instance;
    }

    void clear() {
        tot_classes.clear();
        hist_classes.clear();
        sequence_classes.clear();
    }

    //
    // Total logger stuff
    //
    void log_total(std::string s, int64_t x) {
        tot_classes[s].log(x);
    }
    void set_total(std::string s, int64_t x) {
        tot_classes[s].set(x);
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

    //
    // Histogram logger stuff
    //
    void add_histogram(std::string s, double min, double max, int num_buckets) {
        hist_classes.emplace(s,PerfHist(min, max, num_buckets));
    }

    void log_hist(std::string s, double x) {
        hist_classes[s].log(x);
    }

    const PerfHist get_hist(std::string s) const {
        if(hist_classes.count(s) == 0)
            return PerfHist();

        auto to_ret = hist_classes.at(s);
        return to_ret;
    }

    //
    // Sequence Logger stuff
    //
    void add_sequence(std::string s) {
        sequence_classes.emplace(s, PerfSequence());
    }
    void log_sequence(std::string s, double x) {
        sequence_classes[s].log(x);
    }
    const std::vector<double>& get_sequence(std::string s) const {
        return sequence_classes.at(s).sequence;
    }
    void print_sequence(std::string s) const {
        std::cout << "=======" << std::endl;
        std::cout << s << std::endl;
        std::cout << "=======" << std::endl;
        
        auto to_print = sequence_classes.at(s).sequence;
        for(auto a : to_print) {
            std::cout << a << std::endl;
        }
    }

private:
    PerfLog(){}
public:
    PerfLog(PerfLog const&) = delete;
    void operator=(PerfLog const&) = delete;
};

#endif
