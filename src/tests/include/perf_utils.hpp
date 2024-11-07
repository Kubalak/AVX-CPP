#pragma once
#ifndef _AVXCPP_PERF_UTILS_HPP
#define _AVXCPP_PERF_UTILS_HPP
#include "test_utils.hpp"
#include <utility>

namespace testing{
    namespace perf{

        void printCPUDetails() {
            std::array<char, 128> buff;
            std::string cpu_info;
            FILE* pipe = _popen("wmic cpu get name,numberofcores,numberoflogicalprocessors /FORMAT:list", "r");
            if(pipe){
                while(!feof(pipe))
                {
                    if(fgets(buff.data(), buff.size(), pipe) != NULL)
                        cpu_info += buff.data();
                }
                _pclose(pipe);

                std::cout << "CPU info: \n\"" << cpu_info << "\"\n";
                #ifndef NDEBUG
                    std::ofstream procinfo("procinfo.txt", std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
                    if(procinfo.good()){
                        procinfo.write(cpu_info.data(), cpu_info.size());
                        procinfo.close();
                    }
                #endif
                
            }
        }

        /**
         * Performs a CPU warmup using dummy AVX2 based load for `ms` number of milliseconds.
         * 
         * Disclaimer: Fully loads only one CPU thread.
         * 
         * @param ms How many milliseconds of load should take.
         * @param silent If set as `true` don't print to stdout.
         */
        void doCPUWarmup(const unsigned int& ms = 10, const bool& silent = false) noexcept {
            auto start = std::chrono::steady_clock::now();
            auto stop = start;

            __m256i a = _mm256_setzero_si256(), b = _mm256_setzero_si256();
            alignas(32) int v[8]; 

            volatile int64_t c;
            while(1) {
                for(unsigned int i{0};i < 8; ++i)
                    v[i] = rand();
                a = _mm256_xor_si256(a, b);
                a = _mm256_load_si256((const __m256i*)v);
                stop = std::chrono::steady_clock::now();
                c = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
                if(c >= ms)
                    break;
            }
            
            if(!silent) {
                stop = std::chrono::steady_clock::now();
                auto [value, unit] = universalDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count());
                std::printf("CPU warmup finished in %.4lf %s\n", value, unit.c_str());
            }
        }

        /**
         * Performs a performance test of + and += operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testAddAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            T d(dLit);
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a + b;
                c += d;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] + bV[pos];
                cV[pos] += dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of + and += operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testAddSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) + *(bV.data() + pos);
                *(cV.data() + pos) += d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of - and -= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testSubAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            T d(dLit);
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a - b;
                c -= d;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] - bV[pos];
                cV[pos] -= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of - and -= operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testSubSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) - *(bV.data() + pos);
                *(cV.data() + pos) -= d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of * and *= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Some `avx` types do not use AVX2 (for now) to do the multiplication.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testMulAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            T d(dLit);
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a * b;
                c *= d;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] * bV[pos];
                cV[pos] *= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of * and *= operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testMulSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) * *(bV.data() + pos);
                *(cV.data() + pos) *= d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of / and /= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Some `avx` types do not use AVX2 for this.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testDivAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            T d(dLit);
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a / b;
                c /= d;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] / bV[pos];
                cV[pos] /= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of - and -= operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testDivSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) / *(bV.data() + pos);
                *(cV.data() + pos) /= d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of % and %= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Some `avx` types do not use AVX2 for this.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testModAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            T d(dLit);
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a % b;
                c %= d;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] % bV[pos];
                cV[pos] %= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of % and %= operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testModSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) % *(bV.data() + pos);
                *(cV.data() + pos) %= d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of & and &= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs & operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. No | ^ implemented because all SIMD bitwise operators have the same latency.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testAndAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            T d(dLit);
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a & b;
                c &= d;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] & bV[pos];
                cV[pos] &= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of & and &= operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testAndSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) & *(bV.data() + pos);
                *(cV.data() + pos) &= d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of << and <<= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs << operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = T::storedType>
        int64_t testLshiftAVX(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV, const bool print = true, const unsigned int size = T::size){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }
            if(aV.size() != cV.size())
                 cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();
            T c;
            S dLit = bV[bV.size() / 2];
            uint64_t pos = 0;

            while(pos + size < aV.size()){
                T a(aV.data() + pos);
                T b(bV.data() + pos);
                c = a << b;
                c <<= dLit;
                c.save(cV.data() + pos);
                pos += size;
            }

            while(pos < aV.size()){
                cV[pos] = aV[pos] & bV[pos];
                cV[pos] <<= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of << and <<= operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename S>
        int64_t testLshiftSeq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size()){
                std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
                return -1;
            }

            if(aV.size() != cV.size())
                cV.resize(aV.size());

            auto start = std::chrono::steady_clock::now();

            S d = bV[bV.size() / 2];

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                *(cV.data() + pos) = *(aV.data() + pos) << *(bV.data() + pos);
                *(cV.data() + pos) <<= d;
            }
            
            auto stop = std::chrono::steady_clock::now();
            if(print)
                print_test_duration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

    }
}

#endif