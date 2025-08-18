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

int testCmpDouble() {

    avx::Double256 a(1.0), b({1.0, 2.0, 1.5}), c(-0.0), d(0.0);
    int result = 0;
    if(!(a == a)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Double256",
            "avx::Double256",
            "True",
            "False"
        );
    }

    if(!(a == 1.0)){
        result = 1;
        testing::printTestFailed(
            __FILE__,
            __LINE__,
            __func__,
            "==",
            "avx::Double256",
            "avx::Double256",
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
            "avx::Double256",
            "avx::Double256",
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
            "avx::Double256",
            "avx::Double256",
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
            "avx::Double256",
            "avx::Double256",
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
            "avx::Double256",
            "avx::Double256",
            "True",
            "False"
        );
    }

    return result;

}



int main(int argc, char *argv[]) {
    float a = 0.0f;
    float b = -0.0f;
    double c = 0.0;
    double d = -0.0;
    int result = testCmpFloat();
    result |= testCmpDouble();
    std::cout << "0.0f == -0.0f -> " << (a == b) << '\n';
    std::cout << "0.0  == -0.0  -> " << (c == d) << '\n';

    std::cout << avx::Float256{1, 1.2f, 3, 5, 0, 7.68f}.str() << '\n';
    std::cout << avx::Double256{1, 1.2f, 0, 7.68f}.str() << '\n';
    return result;
}