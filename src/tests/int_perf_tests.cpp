#include <types/int256.hpp>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <numeric>

double Mean(std::vector<long long>& items){
    double val = 0.0;

    for(long long& item : items)
        val += item;
    
    return val / items.size();
}

double stdev(std::vector<long long>& items, double mean) {
    double variance = 0;
    for(long long& item : items)
        variance += pow(item - mean, 2);
    
    return sqrt(variance / items.size());

}

void test_division_avx_float(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
    __m256i v1 = a.get();
    __m256i v2 = b.get();

    std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    auto start = std::chrono::system_clock::now();
    auto stop = start;
    __m256i iresult;

    unsigned int i{0};
    for(i; i < iters; ++i) {
        __m256 fv1 = _mm256_cvtepi32_ps(v1);
        __m256 fv2 = _mm256_cvtepi32_ps(v2);
        __m256 result = _mm256_div_ps(fv1, fv2);
        iresult = _mm256_cvttps_epi32(result);

        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    double mean = Mean(counts);
    
    std::cout << "Result: " << avx::Int256(iresult).str() << '\n';
    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms, stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}

void test_division_avx_seq(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
    __m256i v1 = a.get();
    __m256i v2 = b.get();

    std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    auto start = std::chrono::system_clock::now();
    auto stop = start;

    __m256i iresult = _mm256_setzero_si256();
    unsigned int i{0};
    for(; i < iters; ++i){
        int* av = (int*)&v1, *bv = (int*)&v2;

        iresult = _mm256_set_epi32(
            av[7] / bv[7],
            av[6] / bv[6],
            av[5] / bv[5],
            av[4] / bv[4],
            av[3] / bv[3],
            av[2] / bv[2],
            av[1] / bv[1],
            av[0] / bv[0]
        );

        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    std::cout << "Result: " << avx::Int256(iresult).str() << '\n';

    double mean = Mean(counts);
    
    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms, stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}

void test_division_avx_seq_float(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
    __m256i v1 = a.get();
    __m256i v2 = b.get();

    std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    auto start = std::chrono::system_clock::now();
    auto stop = start;
    __m256i iresult = _mm256_setzero_si256();
    unsigned int i{0};
    for(; i < iters; ++i){
        __m256 fv1 = _mm256_cvtepi32_ps(v1);
        __m256 fv2 = _mm256_cvtepi32_ps(v2);
        float* av = (float*)&fv1, *bv = (float*)&fv2;

        iresult = _mm256_set_epi32(
            av[7] / bv[7],
            av[6] / bv[6],
            av[5] / bv[5],
            av[4] / bv[4],
            av[3] / bv[3],
            av[2] / bv[2],
            av[1] / bv[1],
            av[0] / bv[0]
        );
        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    
    std::cout << "Result: " << avx::Int256(iresult).str() << '\n';

    double mean = Mean(counts);

    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}


void test_mod_avx_float(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
    __m256i v1 = a.get();
    __m256i v2 = b.get();

    std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    auto start = std::chrono::system_clock::now();
    auto stop = start;

    __m256i iresult = _mm256_setzero_si256();

    unsigned int i{0};
    for(i; i < iters; ++i){
        __m256 fv1 = _mm256_cvtepi32_ps(v1);
        __m256 fv2 = _mm256_cvtepi32_ps(v2);
        __m256 result = _mm256_div_ps(fv1, fv2);
        __m256i byprod = _mm256_cvttps_epi32(result);
        iresult = _mm256_sub_epi32(v1, _mm256_mullo_epi32(v2, byprod));
        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    std::cout << "Result: " << avx::Int256(iresult).str() << '\n';

    double mean = Mean(counts);

    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms, stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}

void test_mod_avx_seq(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
    __m256i v1 = a.get();
    __m256i v2 = b.get();

    std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    auto start = std::chrono::system_clock::now();
    auto stop = start;

    __m256i iresult = _mm256_setzero_si256();
    unsigned int i{0};
    for(; i < iters; ++i){
        int* av = (int*)&v1, *bv = (int*)&v2;

        iresult = _mm256_set_epi32(
            av[7] % bv[7],
            av[6] % bv[6],
            av[5] % bv[5],
            av[4] % bv[4],
            av[3] % bv[3],
            av[2] % bv[2],
            av[1] % bv[1],
            av[0] % bv[0]
        );

        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    std::cout << "Result: " << avx::Int256(iresult).str() << '\n';

    double mean = Mean(counts);
    
    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms, stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}

void baseline_avx_add(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
        std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    auto start = std::chrono::system_clock::now();
    auto stop = start;

    avx::Int256 result;
    unsigned int i{0};
    for(; i < iters; ++i){
        result = a + b;
        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    std::cout << "Result: " << result.str() << '\n';

    double mean = Mean(counts);
    
    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms, stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}

void baseline_avx_add_raw(const avx::Int256& a, const avx::Int256& b, unsigned int iters){
    std::cout << "Starting performance test " << __func__ << '\n';

    std::vector<long long> counts;
    __m256i result;
    __m256i av = a.get();
    __m256i bv = b.get();
    auto start = std::chrono::system_clock::now();
    auto stop = start;


    unsigned int i{0};
    for(; i < iters; ++i){
        result = _mm256_add_epi32(av, bv);
        stop = std::chrono::system_clock::now();
        counts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
        start = stop;
    }
    std::cout << "Result: " << avx::Int256(result).str() << '\n';

    double mean = Mean(counts);
    
    printf("Performance test %s finished. Iterations: %d Time total: %.3lf ms, stddev. %.3lf us, per loop %.3lf us\n",
        __func__,
        i,
        std::accumulate(counts.begin(), counts.end(), 0.0) / 1000.f,
        stdev(counts, mean),
        mean
    );
}


int main(int argc, char* argv[]) {
    if (argc < 2)
        return 1;
    unsigned int iters = 0;
    try{
        iters = std::stoul(argv[1]);
    } catch (std::invalid_argument) {
        std::cerr << "Failed to parse argument" << argv[1] << '\n';
        return 1;
    }

    avx::Int256 a{128, 125, 456, 265, 710, 288, 353, 321};
    avx::Int256 b{5, 14, 456, 3, 21, 33, 24, 88};

    std::cout << "Vector a: " << a.str() << '\n';
    std::cout << "Vector b: " << b.str() << '\n';

    test_division_avx_float(a, b, iters);
    puts("--");
    test_division_avx_seq(a, b, iters);
    puts("--");
    test_division_avx_seq_float(a, b, iters);
    puts("--");
    test_mod_avx_float(a, b, iters);
    puts("--");
    test_mod_avx_seq(a, b, iters);
    puts("--");
    baseline_avx_add(a, b, iters);
    puts("--");
    baseline_avx_add_raw(a, b, iters);


    return 0;
}