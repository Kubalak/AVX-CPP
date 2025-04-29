#include <types/float256.hpp>
#include <types/double256.hpp>
#include <test_utils.hpp>

int testCmpFloat(){
    avx::Float256 a(1.0f), b({1.0f, 2.0f, 1.5f}), c(-0.0f), d(0.0f);
    int result = 0;
    if(!(a == a)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Float256",
            "avx::Float256",
            "True",
            "False"
        );
    }

    if(!(a == 1.0f)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Float256",
            "avx::Float256",
            "True",
            "False"
        );
    }

    if(!(b != 0)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Float256",
            "avx::Float256",
            "True",
            "False"
        );
    }

    if(!(a != b)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "!=",
            "avx::Float256",
            "avx::Float256",
            "True",
            "False"
        );
    }

    if(!(c == d)) {
            result = 1;
            testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Float256",
            "avx::Float256",
            "True",
            "False"
        );
    }

    if(!(c == c)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Float256",
            "avx::Float256",
            "True",
            "False"
        );
    }

    return result;
}



int main(int argc, char *argv[]) {
    float a = 0.0f;
    float b = -0.0f;
    int result = testCmpFloat();
    std::cout << "0.0f == -0.0f -> " << (a == b) << '\n';
    return result;
}