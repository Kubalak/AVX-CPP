#include <iostream>
#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>
#include <cstdio>
#include <ops/avxmath.hpp>
#include <filesystem>

#ifdef _MSC_VER
    #define Sleef_cosd4_u35avx2 _mm256_cos_pd
    #define Sleef_asind4_u35avx2 _mm256_asin_pd
#else
    #include <sleef.h>
#endif

void computeHaversineSeq(const std::vector<double> &latitudes_1, const std::vector<double> &longitudes_1, const std::vector<double> &latitudes_2, const std::vector<double> &longitudes_2, std::vector<double> &distances) {
    if(latitudes_1.size() != longitudes_1.size() || latitudes_1.size() != latitudes_2.size() || latitudes_1.size() != longitudes_2.size()){
        std::cerr << "Vector sizes don't match!\n";
        return;
    }
    if(latitudes_1.size() != distances.size())
        distances.resize(latitudes_1.size());
    double r = 6371.0f;
    double p = std::numbers::pi / 180.0f;
    size_t size = latitudes_1.size();

    for(size_t i{0}; i < size; ++i) {
        double a = .5f - std::cos((latitudes_2[i] - latitudes_1[i]) * p) / 2 + std::cos(latitudes_1[i] * p) * std::cos(latitudes_2[i] * p) * (1 - std::cos((longitudes_2[i] - longitudes_1[i]) * p)) / 2;
        distances[i] = 2 * r * std::asin(std::sqrt(a));
    }
}

void computeHaversineSleef(const std::vector<double> &latitudes_1, const std::vector<double> &longitudes_1, const std::vector<double> &latitudes_2, const std::vector<double> &longitudes_2, std::vector<double> &distances) {
    if(latitudes_1.size() != longitudes_1.size() || latitudes_1.size() != latitudes_2.size() || latitudes_1.size() != longitudes_2.size()){
        std::cerr << "Vector sizes don't match!\n";
        return;
    }
    
    if(latitudes_1.size() != distances.size())
        distances.resize(latitudes_1.size());

    __m256d r = _mm256_set1_pd(6371.0);
    __m256d p = _mm256_set1_pd(std::numbers::pi / 180.0);
    double pS = std::numbers::pi / 180.0;
    __m256d a;
    __m256d one = _mm256_set1_pd(1);
    __m256d two = _mm256_set1_pd(2);
    __m256d twor = _mm256_mul_pd(two, r);
    size_t size = latitudes_1.size();
    size_t index = 0;

    
    for(; index + 4 < size; index += 4) {
        __m256d dphi = Sleef_cosd4_u35avx2(
            _mm256_mul_pd(
                _mm256_sub_pd(_mm256_loadu_pd(latitudes_2.data() + index), _mm256_loadu_pd(latitudes_1.data() + index)),
                p
            )
            
        );

        __m256d cphi1 = Sleef_cosd4_u35avx2(
            _mm256_mul_pd(
                _mm256_loadu_pd(latitudes_1.data() + index),
                p
            )
        );

        __m256d cphi2 = Sleef_cosd4_u35avx2(
            _mm256_mul_pd(
                _mm256_loadu_pd(latitudes_2.data() + index),
                p
            )
        );

        __m256d last = _mm256_sub_pd(
            one,
            Sleef_cosd4_u35avx2(
                _mm256_mul_pd(
                    _mm256_sub_pd(
                        _mm256_loadu_pd(longitudes_2.data() + index),
                        _mm256_loadu_pd(longitudes_1.data() + index)
                    ),
                    p
                )
            )
        );

        a = _mm256_add_pd(
                _mm256_sub_pd(
                    one,                    
                    dphi
                ),
            _mm256_mul_pd(cphi1,_mm256_mul_pd(cphi2, last))
        );



        _mm256_storeu_pd(
            distances.data() + index, 
            _mm256_mul_pd(twor, Sleef_asind4_u35avx2(_mm256_sqrt_pd(_mm256_div_pd(a, two))))
        );
    }

    while(index < size){
        double a = .5 - std::cos((latitudes_2[index] - latitudes_1[index]) * pS) / 2 + std::cos(latitudes_1[index] * pS) * std::cos(latitudes_2[index] * pS) * (1 - std::cos((longitudes_2[index] - longitudes_1[index]) * pS)) / 2;
        distances[index++] = 2 * 6371.0 * std::asin(std::sqrt(a));
    }    
}

void computeHaversineAVX(const std::vector<double> &latitudes_1, const std::vector<double> &longitudes_1, const std::vector<double> &latitudes_2, const std::vector<double> &longitudes_2, std::vector<double> &distances) {
    if(latitudes_1.size() != longitudes_1.size() || latitudes_1.size() != latitudes_2.size() || latitudes_1.size() != longitudes_2.size()){
        std::cerr << "Vector sizes don't match!\n";
        return;
    }
    
    if(latitudes_1.size() != distances.size())
        distances.resize(latitudes_1.size());

    avx::Double256 r(6371.0);
    avx::Double256 p = std::numbers::pi / 180.0;
    double pS = std::numbers::pi / 180.0;
    avx::Double256 a;
    avx::Double256 one(1);
    avx::Double256 two(2);
    avx::Double256 twor = two * r;
    size_t size = latitudes_1.size();
    size_t index = 0;
    
    for(; index + 4 < size; index += 4) {
        a = 1.0 - avx::cos((avx::Double256(latitudes_2.data() + index) - avx::Double256(latitudes_1.data() + index)) * p) + avx::cos(avx::Double256(latitudes_1.data()+ index) * p) * avx::cos(avx::Double256(latitudes_2.data() + index) * p) * (avx::Double256(1.0) - avx::cos((avx::Double256(longitudes_2.data() + index) - avx::Double256(longitudes_1.data() + index)) * p));
        (twor * avx::asin(avx::sqrt(a / two))).save(distances.data() + index);
    }

    while(index < size){
        double a = .5 - std::cos((latitudes_2[index] - latitudes_1[index]) * pS) / 2 + std::cos(latitudes_1[index] * pS) * std::cos(latitudes_2[index] * pS) * (1 - std::cos((longitudes_2[index] - longitudes_1[index]) * pS)) / 2;
        distances[index++] = 2 * 6371.0 * std::asin(std::sqrt(a));
    }    
}

struct points {
    double lat1;
    double long1;
    double lat2;
    double long2;
};

// double hav(const double lat1, const double long1, const double lat2, const double long2){
//     double r = 6371.0;
//     double p = std::numbers::pi / 180.0;
//     double a = 0.5 - std::cos((lat2 - lat1) * p) / 2+ std::cos(lat1 * p) * std::cos(lat2 * p) * (1 - std::cos((long2 - long1) * p)) / 2;
//     return 2 * r * std::asin(std::sqrt(a));
// }



int main(int argc, char* argv[]) {

    std::cout << std::filesystem::current_path() << '\n';
    FILE *file = fopen("values_full_tuples.bin", "rb");
    std::vector<points>buf(262144);
    std::vector<double> lats1, longs1, lats2, longs2;
    if(file) {
        size_t bytesRead;
        while((bytesRead = fread(reinterpret_cast<char*>(buf.data()), sizeof(points), 262144, file)) == 262144){
            lats1.reserve(lats1.size() + 262144);
            longs1.reserve(longs1.size() + 262144);
            lats2.reserve(lats2.size() + 262144);
            longs2.reserve(longs2.size() + 262144);

            for(auto& val : buf) {
                lats1.push_back(val.lat1);
                longs1.push_back(val.long1);
                lats2.push_back(val.lat2);
                longs2.push_back(val.long2);
            }
            
        }

        for(size_t i{0}; i < bytesRead; ++i){
            lats1.push_back(buf[i].lat1);
            longs1.push_back(buf[i].long1);
            lats2.push_back(buf[i].lat2);
            longs2.push_back(buf[i].long2);
        }
        
        fclose(file);

        std::cout << "Loaded " << lats1.size() << " values\n";
        printf("   %-10s %-10s %-10s %-10s\n", "Lat1", "Long1", "Lat2", "Long2");
        for(int i{0}; i < 10; ++i)
            printf("%2d %.5f %.5f %.5f %.5f\n", i, lats1[i], longs1[i], lats2[i], longs2[i]);

        std::vector<double> distances(lats1.size());
        
        auto start = std::chrono::steady_clock::now();
        
        // size_t size = lats1.size();
        // for(size_t i{0}; i < size; ++i) {
        //     distances[i] = hav(*(lats1.data() + i), *(longs1.data() + i), *(lats2.data() + i), *(longs2.data() + i));
        // }
        
        computeHaversineSeq(lats1, longs1, lats2, longs2, distances);

        auto stop = std::chrono::steady_clock::now();
        printf("\nCalculations finished in %.6lf ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() / 1000.0);
        puts("   Distances");

        for(int i{0}; i < 10; ++i)
            printf("%2d %lf\n", i, distances[i]);
        
        for(auto& distance : distances)
            distance = 0.0;

        start = std::chrono::steady_clock::now();
        
        computeHaversineSleef(lats1, longs1, lats2, longs2, distances);

        stop = std::chrono::steady_clock::now();

        printf("\nCalculations using SIMD Sleef finished in %.6lf ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() / 1000.0);
        puts("   Distances");
        
        for(int i{0}; i < 10; ++i)
            printf("%2d %lf\n", i, distances[i]);
        
        for(auto& distance : distances)
            distance = 0.0;

        start = std::chrono::steady_clock::now();
        
        computeHaversineAVX(lats1, longs1, lats2, longs2, distances);

        stop = std::chrono::steady_clock::now();

        printf("\nCalculations using SIMD lib finished in %.6lf ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() / 1000.0);
        puts("   Distances");
        
        for(int i{0}; i < 10; ++i)
            printf("%2d %lf\n", i, distances[i]);

    }
    else {
        fprintf(stderr, "%s\n", strerror(errno));
    }

    return 0;
}

/*

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>

void computeHaversineSeq(const double *latitudes_1, const double *longitudes_1, const double *latitudes_2, const double *longitudes_2, double *distances, unsigned int size) {
    double p = 3.141592653589793 / 180.0;
    double a;
    for(size_t i = 0; i < size; ++i) {
        a = .5f - cos((latitudes_2[i] - latitudes_1[i]) * p) / 2 + cos(latitudes_1[i] * p) * cos(latitudes_2[i] * p) * (1 - cos((longitudes_2[i] - longitudes_1[i]) * p)) / 2;
        distances[i] = 2 * 6371.0 * asin(sqrt(a));
    }
}


struct points {
    double lat1;
    double long1;
    double lat2;
    double long2;
};


int main(int argc, char* argv[]) {

    FILE *file = fopen("values_full_tuples.bin", "rb");
    struct points buf[1024];
    double *lats1 = malloc(1024 * sizeof(double)), *longs1 = malloc(1024 * sizeof(double)), *lats2 = malloc(1024 * sizeof(double)), *longs2 = malloc(1024 * sizeof(double));
    if(file) {
        size_t itemsRead;
        size_t totalItems = 0;
        while((itemsRead = fread((char*)(buf), sizeof(struct points), 1024, file)) == 1024){
            
            for(size_t i = 0; i < itemsRead; ++i) {
                lats1[totalItems + i] = buf[i].lat1;
                longs1[totalItems + i] = buf[i].long1;
                lats2[totalItems + i] = buf[i].lat2;
                longs2[totalItems + i] = buf[i].long2;
            }
            
            totalItems += itemsRead;

            lats1 = realloc(lats1, (totalItems + 1024) * sizeof(double));
            longs1 = realloc(longs1,  (totalItems + 1024) * sizeof(double));
            lats2 = realloc(lats2, (totalItems + 1024) * sizeof(double));
            longs2 = realloc(longs2, (totalItems + 1024) * sizeof(double));

            
        }

        for(size_t i = 0; i < itemsRead; ++i){
            lats1[itemsRead + i] = buf[i].lat1;
            longs1[itemsRead + i] = buf[i].long1;
            lats2[itemsRead + i] = buf[i].lat2;
            longs2[itemsRead + i] = buf[i].long2;
        }

        totalItems += itemsRead;
        
        fclose(file);

        printf("Loaded %ld values\n" , totalItems);
        printf("   %-10s %-10s %-10s %-10s\n", "Lat1", "Long1", "Lat2", "Long2");
        for(int i = 0; i < 10; ++i)
            printf("%2d %.5f %.5f %.5f %.5f\n", i, lats1[i], longs1[i], lats2[i], longs2[i]);

        clock_t stop, start;
        start = clock();
        double *distances = malloc(totalItems * sizeof(double));
        // size_t size = lats1.size();
        // for(size_t i{0}; i < size; ++i) {
        //     distances[i] = hav(*(lats1.data() + i), *(longs1.data() + i), *(lats2.data() + i), *(longs2.data() + i));
        // }
        
        computeHaversineSeq(lats1, longs1, lats2, longs2, distances, totalItems);

        stop = clock();
        printf("\nCalculations finished in %.6lf ms\n\n", ((double)(stop-start)*1000.0/CLOCKS_PER_SEC));
        puts("   Distances");

        for(int i = 0; i < 10; ++i)
            printf("%2d %lf\n", i, distances[i]);
    }
    else {
        fprintf(stderr, "%s\n", strerror(errno));
    }

    return 0;
}*/