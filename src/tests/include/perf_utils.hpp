#pragma once
#ifndef _AVXCPP_PERF_UTILS_HPP
#define _AVXCPP_PERF_UTILS_HPP
#include "test_utils.hpp"
#include "cpuinfo.hpp"
#include <utility>

#define _AVX_ADD_RAW    1 // Use to check if verification failed e.g. `(testing::perf::allPerfTest() & _AVX_ADD_RAW) != 0;`
#define _AVX_ADD        2 // Use to check if verification failed for `Test add AVX2`.
#define _AVX_ADD_SEQ    4 // Check if verification failed for sequential addition.

#define _AVX_SUB_RAW    8    // Use to check if verification failed.
#define _AVX_SUB        0x10 // Use to check if verification failed.
#define _AVX_SUB_SEQ    0x20 // Use to check if verification failed.

#define _AVX_MUL_RAW    0x40  // Use to check if verification failed.
#define _AVX_MUL        0x80  // Use to check if verification failed.
#define _AVX_MUL_SEQ    0x100 // Use to check if verification failed.

#define _AVX_DIV_RAW    0x200 // Use to check if verification failed.
#define _AVX_DIV        0x400 // Use to check if verification failed.
#define _AVX_DIV_SEQ    0x800 // Use to check if verification failed.

#define _AVX_MOD_RAW    0x1000 // Use to check if verification failed.
#define _AVX_MOD        0x2000 // Use to check if verification failed.
#define _AVX_MOD_SEQ    0x4000 // Use to check if verification failed.

#define _AVX_LSH_RAW    0x8000  // Use to check if verification failed.
#define _AVX_LSH        0x10000 // Use to check if verification failed.
#define _AVX_LSH_SEQ    0x20000 // Use to check if verification failed.

#define _AVX_IGNORE_LSH 0x07FFF // Use to ignore left shift operator verification error.

namespace testing{
    namespace perf{

        template<typename S>
        struct RawAVXFuncs{
            /**
             * Function to perform addition of two vectors using raw AVX2 code. \n
             * Code the function should perform:
             * 
             * 1. Load data from `aV` to temporary vector (`a`) \n
             * 2. Load data from `bV` to temporary vector (`b`) \n
             * 3. Add `a` and `b` -> `c` \n
             * 4. Add `b[b.size() / 2]` to `c`. \n
             * 5. Save `c` to `cV` according to index.
             * 
             * Or following Python way of doing things `cV = (aV + bV) + bV[bV.size() / 2]`
             * 
             * Params in order from first to last.
             * @param aV First vector.
             * @param bV Second vector.
             * @param cV Results vector.
             * @param print Whether to print info or not. 
             * @returns Time taken by the test in nanoseconds.
             */
            std::function<int64_t(const std::vector<S>&, const std::vector<S>&, std::vector<S>&, const bool)> addRaw;

            /**
             * Function to perform substraction of two vectors using raw AVX2 code. \n
             * Code the function should perform:
             * 
             * 1. Load data from `aV` to temporary vector (`a`) \n
             * 2. Load data from `bV` to temporary vector (`b`) \n
             * 3. Substract `a` and `b` -> `c` \n
             * 4. Substract `b[b.size() / 2]` from `c`. \n
             * 5. Save `c` to `cV` according to index.
             * 
             * Or following Python way of doing things `cV = (aV - bV) - bV[bV.size() / 2]`
             * 
             * Params in order from first to last.
             * @param aV First vector.
             * @param bV Second vector.
             * @param cV Results vector.
             * @param print Whether to print info or not. 
             * @returns Time taken by the test in nanoseconds.
             */
            std::function<int64_t(const std::vector<S>&, const std::vector<S>&, std::vector<S>&, const bool)> subRaw;

            /**
             * Function to perform multiplication of two vectors using raw AVX2 code. \n
             * Code the function should perform:
             * 
             * 1. Load data from aV to temporary vector (`a`) \n
             * 2. Load data from bV to temporary vector (`b`) \n
             * 3. Multiply `a` by `b` -> `c` \n
             * 4. Multiply `c` by `b[b.size() / 2]`. \n
             * 5. Save `c` to `cV` according to index.
             * 
             * Or following Python way of doing things `cV = (aV * bV) * bV[bV.size() / 2]`
             * 
             * Params in order from first to last.
             * @param aV First vector.
             * @param bV Second vector.
             * @param cV Results vector.
             * @param print Whether to print info or not. 
             * @returns Time taken by the test in nanoseconds.
             */
            std::function<int64_t(const std::vector<S>&, const std::vector<S>&, std::vector<S>&, const bool)> mulRaw;

            /**
             * Function to perform division of two vectors using raw AVX2 code. \n
             * Code the function should perform:
             * 
             * 1. Load data from `aV` to temporary vector (`a`) \n
             * 2. Load data from `bV` to temporary vector (`b`) \n
             * 3. Divide `a` by `b` -> `c` \n
             * 4. Divide `c` by `b[b.size() / 2]`. \n
             * 5. Save `c` to `cV` according to index.
             * 
             * Or following Python way of doing things `cV = (aV / bV) / bV[bV.size() / 2]`
             * 
             * Params in order from first to last.
             * @param aV First vector.
             * @param bV Second vector.
             * @param cV Results vector.
             * @param print Whether to print info or not. 
             * @returns Time taken by the test in nanoseconds.
             */
            std::function<int64_t(const std::vector<S>&, const std::vector<S>&, std::vector<S>&, const bool)> divRaw;

            /**
             * Function to perform modulo of two vectors using raw AVX2 code. \n
             * Code the function should perform:
             * 
             * 1. Load data from `aV` to temporary vector (`a`) \n
             * 2. Load data from `bV` to temporary vector (`b`) \n
             * 3. Mod `a` by `b` -> `c` \n
             * 4. Mod `c` by `b[b.size() / 2]`. \n
             * 5. Save `c` to `cV` according to index.
             * 
             * Or following Python way of doing things `cV = (aV % bV) % bV[bV.size() / 2]`
             * 
             * Params in order from first to last.
             * @param aV First vector.
             * @param bV Second vector.
             * @param cV Results vector.
             * @param print Whether to print info or not. 
             * @returns Time taken by the test in nanoseconds.
             */
            std::function<int64_t(const std::vector<S>&, const std::vector<S>&, std::vector<S>&, const bool)> modRaw;

            /**
             * Function to perform left shifting of two vectors using raw AVX2 code. \n
             * Code the function should perform:
             * 
             * 1. Load data from `aV` to temporary vector (`a`) \n
             * 2. Load data from `bV` to temporary vector (`b`) \n
             * 3. Left shift `a` by `b` -> `c` \n
             * 4. Left shift `c` by `b[b.size() / 2]`. \n
             * 5. Save `c` to `cV` according to index.
             * 
             * Or following Python way of doing things `cV = (aV << bV) << bV[bV.size() / 2]`
             * 
             * Params in order from first to last.
             * @param aV First vector.
             * @param bV Second vector.
             * @param cV Results vector.
             * @param print Whether to print info or not. 
             * @returns Time taken by the test in nanoseconds.
             */
            std::function<int64_t(const std::vector<S>&, const std::vector<S>&, std::vector<S>&, const bool)> lshRaw;
        };

        template <typename S>
        struct TestConfig {
            int randomSeed = 42;
            int warmupDuration = 10;
            bool verifyValues = true;
            bool doWarmup = true;
            bool printCPUInfo = true;
            bool printWarmupInfo = true;
            bool printPreparationTime = true;
            bool printTestFailed = false;
            bool printVerificationFailed = false;
            // Stores raw AVX functions for performance comparison
            RawAVXFuncs<S> avxFuncs;
        };

        

        void printCPUDetails() {
            std::cout << "CPU name: " << cpuinfo::getCPUName() << '\n';
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
                std::printf("CPU warmup done in    %.4lf %s\n", value, unit.c_str());
            }
        }

        template<typename S>
        std::string validationToStr(const std::tuple<int64_t, S, S>& retVal){
            if(std::get<0>(retVal) == -2) return "E_INVAL_SIZE";
            if(std::get<0>(retVal) == -1) return "OK";
            return "[" + std::to_string(std::get<0>(retVal)) + "] -> " + std::to_string(std::get<1>(retVal)) + " vs " + std::to_string(std::get<2>(retVal));
        }


        /**
         * Verifies results of adding `aV` and `bV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifyAdd(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) + *(bV.data() + pos);
                temp += d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Verifies results of substracting `bV` from `aV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifySub(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) - *(bV.data() + pos);
                temp -= d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Verifies results of multiplying `aV`and `bV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifyMul(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) * *(bV.data() + pos);
                temp *= d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Verifies results of dividing `aV` by `bV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifyDiv(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) / *(bV.data() + pos);
                temp /= d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Verifies results of modulo between `aV` and `bV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifyMod(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) % *(bV.data() + pos);
                temp %= d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Verifies results of AND between `aV` and `bV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifyAnd(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) & *(bV.data() + pos);
                temp &= d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Verifies results of left shift between `aV` and `bV` comparing it with `cV`.
         *
         * @param aV First vector
         * @param bV Second vector
         * @param cV Results vector
         * @param print Print to cerr in case of failure.
         * @returns `-2` if vector sizes don't match. `-1` if verification finished successfully. Otherwise returns a position where `cV` does not match expected value.
         */
        template<typename S>
        std::tuple<int64_t, S, S> verifyLshift(const std::vector<S>& aV, const std::vector<S>& bV, const std::vector<S>& cV,  const bool  print = true){
            if(aV.size() != bV.size() || aV.size() != cV.size()) {
                std::cerr << "Sizes don't match (" << aV.size() << " vs " << bV.size() << " vs " << cV.size() << ")!\n";
                return std::make_tuple(-2, 0, 0);
            }

            S d = bV[bV.size() / 2];
            S temp;

            for(uint64_t pos{0}; pos < aV.size(); ++pos){
                temp = *(aV.data() + pos) << *(bV.data() + pos);
                temp <<= d;
                if(cV[pos] != temp){

                    if(print)
                        std::cerr << "Validation failed for index [" << pos << "]: expected " << temp << " results vector value " << cV[pos]  <<  '\n';

                    return std::make_tuple(pos, temp, cV[pos]);
                }
            }

            return std::make_tuple(-1, 0, 0);
        }

        /**
         * Performs a performance test of + and += operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of + and += operator sequentially on `aV` and `bV`. Use as a baseline for non-AVX2 enabled calculations.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Compilers like `gcc` or `clang` may produce AVX2 instructions for loop optimization. Use `-fno-tree-vectorize` to prevent this behaviour.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of - and -= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                printTestDuration(__func__, start, stop);
            
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
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of * and *= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Some `avx` types do not use AVX2 (for now) to do the multiplication.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                printTestDuration(__func__, start, stop);
            
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of / and /= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Some `avx` types do not use AVX2 for this.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                printTestDuration(__func__, start, stop);
            
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of % and %= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs + operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. Some `avx` types do not use AVX2 for this.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                printTestDuration(__func__, start, stop);
            
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of & and &= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs & operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results. No | ^ implemented because all SIMD bitwise operators have the same latency.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                printTestDuration(__func__, start, stop);
            
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs a performance test of << and <<= operator on types from `avx` namespace. Loads data from `aV` and `bV` and performs << operation.
         * 
         * Disclaimer - due to small loop size they introduce some overhead on results.
         * @param aV First vector with data.
         * @param bV Second vector with data.
         * @param cV Results vector.
         * @param print If set to `false` function produces no output to stdout. Otherwise prints test duration.
         * @param size Number of elements in classes from `avx` namespace. Don't set unless you know what you're doing!
         * @returns Pair containing the results vector and total time in nanoseconds. You can use `testing::universalDuration` to get human-readable values.
         */
        template<typename T, typename S = typename T::storedType>
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
                cV[pos] = aV[pos] << bV[pos];
                cV[pos] <<= dLit;
                ++pos;
            }
            
            auto stop = std::chrono::steady_clock::now();

            if(print)
                printTestDuration(__func__, start, stop);
            
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
                printTestDuration(__func__, start, stop);
            
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        }

        /**
         * Performs all (template) testing for given type. Returns an `int` with according bits set.
         * @param aV Vector containing first argument when testing operators e.g. `aV[i] / bV[i]`.
         * @param bV Vector containing second argument for operators.
         * @param cV Vector to store results. Also used for verification purposes.
         * @param config Tests configuration struct.
         * @param itemsCount By default set to `-1`. When value is greater than `0 `adjusts `aV`, `bV` and `cV` sizes to passed value.
         * @returns `int` of which corresponding bit is set to 1 if verification test has failed. Otherwise set to `0`. Bits are set according to tests order in logs.
         */
        template<typename T, typename S = typename T::storedType>
        int allPerfTest(std::vector<S> &aV, std::vector<S> &bV, std::vector<S> &cV, const TestConfig<S> &config, int64_t itemsCount = -1) {

            if(itemsCount > 0){
                aV.resize(itemsCount);
                bV.resize(itemsCount);
                cV.resize(itemsCount);
            }
            else if (aV.size() != bV.size() || aV.size() != cV.size()) {
                fprintf(stderr, "Sizes don't match!\naV.size = %10zu\nbV.size = %10zu\ncV.size = %10zu\n", aV.size(), bV.size(), cV.size());
                return -1;
            }

            printf("All performance tests for %s {%s x%d}. \nCompiled using %s %d.%d.%d on %s at %s %s\n", demangle(typeid(T).name()).c_str(), demangle(typeid(S).name()).c_str(), T::size, getCompilerName(), getCompilerMajor(), getCompilerMinor(), getCompilerPatchLevel(), getPlatform(), __DATE__, __TIME__);
            printf("Testing with vector size of %zu (%zu bytes)\n", aV.size(), aV.size() * sizeof(S));

            if(config.printCPUInfo)
                printCPUDetails();

            auto start = std::chrono::steady_clock::now();

            std::srand(config.randomSeed);
            constexpr bool isLong = sizeof(S) > 4;
            for(size_t i = 0; i < aV.size(); ++i){
                if(isLong){
                    aV[i] = std::rand();
                    aV[i] <<= 32;
                    aV[i] |= std::rand();

                    bV[i] = std::rand();
                    bV[i] <<= 32;
                    bV[i] |= std::rand() | 1;
                }
                else {
                    aV[i] = std::rand();
                    bV[i] = std::rand() | 1;
                }
            }

            auto stop = std::chrono::steady_clock::now();

            if(config.printPreparationTime){
                auto[value, unit] = testing::universalDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
                printf("%-20s %8.4lf %s\n",  "Preparation took: ", value, unit.c_str());
            }

            int64_t times[18];
            std::tuple<int64_t, S, S> validations[18];
            for(char i=0;i<18;++i)
                validations[i] = std::make_tuple(-1, 0, 0);

            if(config.doWarmup)
                doCPUWarmup(config.warmupDuration, !config.printWarmupInfo);

            start = std::chrono::steady_clock::now();

            // THIS IS SH*T IK

            if(config.avxFuncs.addRaw){
                times[0] = config.avxFuncs.addRaw(aV, bV, cV, config.printTestFailed);
                if(config.verifyValues)
                    validations[0] = testing::perf::verifyAdd(aV, bV, cV, config.printVerificationFailed);
            }

            times[1] = testing::perf::testAddAVX<T>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[1] = testing::perf::verifyAdd(aV, bV, cV, config.printVerificationFailed);

            times[2] = testing::perf::testAddSeq<S>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[2] = testing::perf::verifyAdd(aV, bV, cV, config.printVerificationFailed);

            if(config.avxFuncs.subRaw){
                times[3] = config.avxFuncs.subRaw(aV, bV, cV, config.printTestFailed);
                if(config.verifyValues)
                    validations[3] = testing::perf::verifySub(aV, bV, cV, config.printVerificationFailed);
            }

            times[4] = testing::perf::testSubAVX<T>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[4] = testing::perf::verifySub(aV, bV, cV, config.printVerificationFailed);

            times[5] = testing::perf::testSubSeq<S>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[5] = testing::perf::verifySub(aV, bV, cV, config.printVerificationFailed);

            if(config.avxFuncs.mulRaw){
                times[6] = config.avxFuncs.mulRaw(aV, bV, cV, config.printTestFailed);
                if(config.verifyValues)
                    validations[6] = testing::perf::verifyMul(aV, bV, cV, config.printVerificationFailed);
            }
            
            times[7] = testing::perf::testMulAVX<T>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[7] = testing::perf::verifyMul(aV, bV, cV, config.printVerificationFailed);

            times[8] = testing::perf::testMulSeq<S>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[8] = testing::perf::verifyMul(aV, bV, cV, config.printVerificationFailed);
            
            if(config.avxFuncs.divRaw){
                times[9] = config.avxFuncs.divRaw(aV, bV, cV, config.printTestFailed);
                if(config.verifyValues)
                    validations[9] = testing::perf::verifyDiv(aV, bV, cV, config.printVerificationFailed);
            }

            times[10] = testing::perf::testDivAVX<T>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[10] = testing::perf::verifyDiv(aV, bV, cV, config.printVerificationFailed);

            times[11] = testing::perf::testDivSeq<S>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[11] = testing::perf::verifyDiv(aV, bV, cV, config.printVerificationFailed);
            
            if(config.avxFuncs.modRaw){
                times[12] = config.avxFuncs.addRaw(aV, bV, cV, config.printTestFailed);
                if(config.verifyValues)
                    validations[12] = testing::perf::verifyMod(aV, bV, cV, config.printVerificationFailed);
            }

            times[13] = testing::perf::testModAVX<T>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[13] = testing::perf::verifyMod(aV, bV, cV, config.printVerificationFailed);

            times[14] = testing::perf::testModSeq<S>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[14] = testing::perf::verifyMod(aV, bV, cV, config.printVerificationFailed);
            
            if(config.avxFuncs.lshRaw){
                times[15] = config.avxFuncs.lshRaw(aV, bV, cV, config.printTestFailed);
                if(config.verifyValues)
                    validations[15] = testing::perf::verifyLshift(aV, bV, cV, config.printVerificationFailed);
            }

            times[16] = testing::perf::testLshiftAVX<T>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[16] = testing::perf::verifyLshift(aV, bV, cV, config.printVerificationFailed);

            times[17] = testing::perf::testLshiftSeq<S>(aV, bV, cV, config.printTestFailed);
            if(config.verifyValues)
                validations[17] = testing::perf::verifyLshift(aV, bV, cV, config.printVerificationFailed);

            std::string validationRes = "";
            std::pair<double, std::string> duration;

            if(config.avxFuncs.addRaw){
                duration = testing::universalDuration(times[0]);
                if(config.verifyValues)
                    validationRes = validationToStr(validations[0]);
                printf("%-20s %8.4lf %-3s%s\n", "Test add AVX2 raw:", duration.first, duration.second.c_str(), validationRes.c_str());
            }
            else 
                printf("Test add AVX2 raw:       skipped...\n");
            
            duration = testing::universalDuration(times[1]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[1]);
            printf("%-20s %8.4lf %-3s%s\n", "Test add AVX2:", duration.first, duration.second.c_str(), validationRes.c_str());

            duration = testing::universalDuration(times[2]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[2]);
            printf("%-20s %8.4lf %-3s%s\n", "Test add seq:", duration.first, duration.second.c_str(), validationRes.c_str());
            
            if(config.avxFuncs.subRaw){
                duration = testing::universalDuration(times[3]);
                if(config.verifyValues)
                    validationRes = validationToStr(validations[3]);
                printf("%-20s %8.4lf %-3s%s\n", "Test sub AVX2 raw:", duration.first, duration.second.c_str(), validationRes.c_str());
            }
            else 
                printf("Test sub AVX2 raw:       skipped...\n");
            
            duration = testing::universalDuration(times[4]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[4]);
            printf("%-20s %8.4lf %-3s%s\n", "Test sub AVX2:", duration.first, duration.second.c_str(), validationRes.c_str());

            duration = testing::universalDuration(times[5]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[5]);
            printf("%-20s %8.4lf %-3s%s\n", "Test sub seq:", duration.first, duration.second.c_str(), validationRes.c_str());

            if(config.avxFuncs.mulRaw){
                duration = testing::universalDuration(times[6]);
                if(config.verifyValues)
                    validationRes = validationToStr(validations[6]);
                printf("%-20s %8.4lf %-3s%s\n", "Test mul AVX2 raw:", duration.first, duration.second.c_str(), validationRes.c_str());
            }
            else
                printf("Test mul AVX2 raw:       skipped...\n");

            duration = testing::universalDuration(times[7]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[7]);
            printf("%-20s %8.4lf %-3s%s\n", "Test mul AVX2:", duration.first, duration.second.c_str(), validationRes.c_str());

            duration = testing::universalDuration(times[8]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[8]);
            printf("%-20s %8.4lf %-3s%s\n", "Test mul seq:", duration.first, duration.second.c_str(), validationRes.c_str());

            if(config.avxFuncs.divRaw){
                duration = testing::universalDuration(times[9]);
                if(config.verifyValues)
                    validationRes = validationToStr(validations[9]);
                printf("%-20s %8.4lf %-3s%s\n", "Test div AVX2 raw:", duration.first, duration.second.c_str(), validationRes.c_str());
            }
            else 
                printf("Test div AVX2 raw:       skipped...\n");

            duration = testing::universalDuration(times[10]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[10]);
            printf("%-20s %8.4lf %-3s%s\n", "Test div AVX2:", duration.first, duration.second.c_str(), validationRes.c_str());

            duration = testing::universalDuration(times[11]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[11]);
            printf("%-20s %8.4lf %-3s%s\n", "Test div seq:", duration.first, duration.second.c_str(), validationRes.c_str());

            if(config.avxFuncs.modRaw){
                duration = testing::universalDuration(times[12]);
                if(config.verifyValues)
                    validationRes = validationToStr(validations[12]);
                printf("%-20s %8.4lf %-3s%s\n", "Test mod AVX2 raw:", duration.first, duration.second.c_str(), validationRes.c_str());
            }
            else
                printf("Test mod AVX2 raw:       skipped...\n");

            duration = testing::universalDuration(times[13]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[13]);
            printf("%-20s %8.4lf %-3s%s\n", "Test mod AVX2:", duration.first, duration.second.c_str(), validationRes.c_str());

            duration = testing::universalDuration(times[14]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[14]);
            printf("%-20s %8.4lf %-3s%s\n", "Test mod seq:", duration.first, duration.second.c_str(), validationRes.c_str());

            if(config.avxFuncs.lshRaw){
                duration = testing::universalDuration(times[15]);
                if(config.verifyValues)
                    validationRes = validationToStr(validations[15]);
                printf("%-20s %8.4lf %-3s%s\n", "Test lshift AVX2 raw:", duration.first, duration.second.c_str(), validationRes.c_str());
            }
            else 
                printf("Test lshift AVX2 raw:    skipped...\n");

            duration = testing::universalDuration(times[16]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[16]);
            printf("%-20s %8.4lf %-3s%s\n", "Test lshift AVX2:", duration.first, duration.second.c_str(), validationRes.c_str());

            duration = testing::universalDuration(times[17]);
            if(config.verifyValues)
                validationRes = validationToStr(validations[17]);
            printf("%-20s %8.4lf %-3s%s\n", "Test lshift seq:", duration.first, duration.second.c_str(), validationRes.c_str());

            int result = 0;
            if(config.verifyValues)
                for(unsigned int i = 0; i < 18; ++i)
                    result |= (std::get<0>(validations[i]) != -1) << i;
            
            stop = std::chrono::steady_clock::now();

            duration = universalDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count());

            printf("%-20s %8.4lf %s\n","Tests finished in:", duration.first, duration.second.c_str());

            return result;
        }

        /**
         * Returns the time of function execution in nanoseconds.
         * @param f Function to be executed.
         * @param args Arguments to be passed to the function.
         */
        template<typename S, typename ...args>
        uint64_t funcTime(S f, args&& ...params) {
            auto start = std::chrono::steady_clock::now();
            f(std::forward<args>(params)...);
            auto stop = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

    }
}

#endif