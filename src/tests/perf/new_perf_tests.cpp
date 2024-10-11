#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <utility>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <types/int256.hpp>

constexpr unsigned int GB = 1024*1024*1024;
constexpr unsigned int MB = 1024*1024;
constexpr unsigned int KB = 1024;

std::pair<double, std::string> universal_duration(int64_t ticks){
    static const std::array<std::string, 5> times{"ns", "us", "ms", "s", "m"};
    unsigned int i = 0;
    while(ticks / pow(1000., i) > 1000) ++i;

    if(i > 4)
        return {ticks / pow(1000., 4), "m"};

    return {ticks / pow(1000., i), times[i]};
}

bool verify_results(const std::vector<int> &results) {
    std::vector<int> buffer(1 * MB / sizeof(int)); // 1MB buffer

    if(!std::filesystem::exists("int_result.bin") || !std::filesystem::is_regular_file("int_result.bin")){
        std::cerr << "File int_result.bin does not exist or is not a regular file!\n";
        return false;
    }

    uintmax_t filesize = std::filesystem::file_size("int_result.bin");
    if(filesize / sizeof(int) != results.size()){
        std::cerr << "File size mismatch with vector (" << filesize << " vs " << results.size() * sizeof(int) << ")\n";
        return false;
    }

    
    std::ifstream resultFile("int_result.bin", std::ios_base::binary);
    if(resultFile.good()){
        unsigned long long index = 0;
        while(index < results.size()) {

            resultFile.read((char*)buffer.data(), buffer.size() * sizeof(int));
            int64_t items_count = resultFile.gcount() / sizeof(int);

            for(int64_t i{0}; i < items_count; ++i) {

                if(buffer[i] != results[index + i]){
                    fprintf(stderr, "[%llu] %d != %d\n", index, buffer[i], results[index + i]);
                    resultFile.close();
                    return false;
                }
            }
            index += items_count;
        }
        resultFile.close();
    } else {
        std::cerr << "Unable to open file int_result.bin\n";
        return false;
    }

    return true;
}

int main(int argc, char* argv[]){
    static_assert(sizeof(int) == 4, "You are using 32-bit compile option! Please change to x64.");
    
    auto start = std::chrono::steady_clock::now();
    std::vector<int> a, b, c;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, 1000000);

    bool is_a = std::filesystem::exists("int_noise_a.bin") && std::filesystem::is_regular_file("int_noise_a.bin");
    bool is_b = std::filesystem::exists("int_noise_b.bin") && std::filesystem::is_regular_file("int_noise_b.bin");

    // File to check results
    bool is_c = std::filesystem::exists("int_result.bin") && std::filesystem::is_regular_file("int_result.bin");

    if(is_a && is_b){
        uintmax_t a_size = std::filesystem::file_size("int_noise_a.bin");
        uintmax_t b_size = std::filesystem::file_size("int_noise_b.bin");

        if(a_size == b_size){
            a.resize(a_size / sizeof(int));
            b.resize(b_size / sizeof(int));
            c.resize(b_size / sizeof(int));

            std::ifstream afile("int_noise_a.bin", std::ios_base::binary);
            if(afile.good()){    
                afile.read((char*)a.data(), a.size()*sizeof(int));
                afile.close();
            } else {
                is_c = false;
                std::cerr << "Unable to open int_noise_a.bin\n";
                
                for(auto& item : a)
                    item = dist(rng);
            }

            std::ifstream bfile("int_noise_b.bin", std::ios_base::binary);
            if(bfile.good()) {
                bfile.read((char*)b.data(), b.size()*sizeof(int));
                bfile.close();
            } else {
                is_c = false;
                std::cerr << "Unable to open int_noise_b.bin\n";

                for(auto& item : b)
                    item = dist(rng);
            }
        } 
    } else {
        a.resize(GB / sizeof(int));
        b.resize(GB / sizeof(int));
        c.resize(GB / sizeof(int));
    }

    if(!is_a || !is_b){
        for(size_t i{0}; i < a.size();++i){
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
    }

    size_t index = 0;
    auto stop = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();

    auto duration_pair = universal_duration(duration);

    std::cout << "Preparation:\t" << duration_pair.first << ' ' << duration_pair.second << " to complete.\n";

    start = std::chrono::steady_clock::now();
    size_t max_index = a.size() - 8;
    while(index < max_index){
        __m256i aV = _mm256_lddqu_si256((const __m256i*)(a.data() + index));
        __m256i bV = _mm256_lddqu_si256((const __m256i*)(b.data() + index));
        __m256i cV = _mm256_add_epi32(aV, bV);
        cV = _mm256_slli_epi32(cV, 2);
        cV = _mm256_mullo_epi32(cV, _mm256_set1_epi32(5));
        cV = _mm256_or_si256(cV, bV);
        cV = _mm256_sub_epi32(cV, aV);
        _mm256_storeu_si256((__m256i*)(c.data() + index), cV);
        index += 8;
    }

    for(; index < a.size(); ++index){
        c[index] = a[index] + b[index];
        c[index] <<= 2;
        c[index] *= 5;
        c[index] |= b[index];
        c[index] -= a[index];
    }
    
    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();

    duration_pair = universal_duration(duration);

    std::cout << "Raw AVX2:\t" << duration_pair.first << ' ' << duration_pair.second << " to complete.\n";

    if(is_c){
        std::string is_ok = (verify_results(c) ? "OK" : "NOT OK");
        std::cout << "Verification:\t" << is_ok << '\n';
    }

    for(auto& val:c)
        val = 0;

    index = 0;
    start = std::chrono::steady_clock::now();

    while(index < max_index){
        avx::Int256 aV(a.data() + index);
        avx::Int256 bV(b.data() + index);
        auto cV = aV + bV;
        cV <<= 2;
        cV *= 5;
        cV |= bV;
        cV -= aV;
        cV.save(c.data() + index);
        index += 8;
    }

    for(; index < a.size(); ++index){
        c[index] = a[index] + b[index];
        c[index] <<= 2;
        c[index] *= 5;
        c[index] |= b[index];
        c[index] -= a[index];
    }


    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();

    duration_pair = universal_duration(duration);

    std::cout << "Using Int256:\t" << duration_pair.first << ' ' << duration_pair.second << " to complete.\n";

    if(is_c){
        std::string is_ok = (verify_results(c) ? "OK" : "NOT OK");
        std::cout << "Verification:\t" << is_ok << '\n';
    }

    for(auto& val:c)
        val = 0;

    start = std::chrono::steady_clock::now();

    auto *aP = a.data(), *bP = b.data(), *cP = c.data();
    for(size_t i{0}; i < a.size(); ++i){
        cP[i] = aP[i] + bP[i];
        cP[i] <<= 2;
        cP[i] *= 5;
        c[i] |= b[i];
        c[i] -= a[i];
    }

    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();

    duration_pair = universal_duration(duration);

    std::cout << "Not Using AVX2:\t" << duration_pair.first << ' ' << duration_pair.second << " to complete.\n";

    if(is_c){
        std::string is_ok = (verify_results(c) ? "OK" : "NOT OK");
        std::cout << "Verification:\t" << is_ok << '\n';
    }

    if(!is_a){
        std::ofstream aOut("int_noise_a.bin", std::ios_base::binary);
        if(aOut.good()){
            aOut.write((char*)a.data(), a.size() * sizeof(int));
            if(aOut.fail())
                std::cerr << "Failed to write to int_noise_a.bin\n";
            aOut.close();
        }
    }

    if(!is_b){
        std::ofstream bOut("int_noise_b.bin", std::ios_base::binary);
        if(bOut.good()){
            bOut.write((char*)b.data(), b.size() * sizeof(int));
            if(bOut.fail())
                std::cerr << "Failed to write to int_noise_b.bin\n";
            bOut.close();
        }
    }

    if(!is_c){
        std::ofstream cOut("int_result.bin", std::ios_base::binary);
        if(cOut.good()){
            cOut.write((char*)c.data(), c.size() * sizeof(int));
            if(cOut.fail())
                std::cerr << "Failed to write to int_noise_c.bin\n";
            cOut.close();
        }
    }


    return 0;
}