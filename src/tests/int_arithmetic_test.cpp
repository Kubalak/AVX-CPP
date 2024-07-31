#include "types/int256.hpp"
#include <iostream>

int check_create(void) {

    std::cout << "Starting " << __func__ << '\n';

    avx::Int256 a(std::array<int, 8>{0, 1, 2, 3, 4, 5, 6, 7});

    char res = 0;
    
    if(a[0] != 7){
        printf("[0] Values don't match %d != %d\n", a[0], 7);
        res |= 1;
    }
    
    if(a[1] != 6){
        printf("[1] Values don't match %d != %d\n", a[1], 6);
        res |= 1;
    }

    if(a[2] != 5){
        printf("[2] Values don't match %d != %d\n", a[2], 5);
        res |= 1;
    }
    
    if(a[3] != 4){
        printf("[3] Values don't match %d != %d\n", a[3], 4);
        res |= 1;
    }
    
    if(a[4] != 3){
        printf("[4] Values don't match %d != %d\n", a[4], 3);
        res |= 1;
    }
    
    if(a[5] != 2){
        printf("[5] Values don't match %d != %d\n", a[5], 2);
        res |= 1;
    }
    
    if(a[6] != 1){
        printf("[6] Values don't match %d != %d\n", a[6], 1);
        res |= 1;
    }
    
    if(a[7] != 0){
        printf("[7] Values don't match %d != %d\n", a[7], 0);
        res |= 1;
    }

    return res;
}

int check_add(void){
    std::cout << "Starting " << __func__ << '\n';
    avx::Int256 a({1, 3, 5, 2, 8, 9, 7, 10});
    avx::Int256 c(std::array<int,8>{6, 8, 10, 7, 13, 14, 12, 15});
    std::cout << "a: " << a.str() << '\n';
    avx::Int256 b = a + 5;
    std::cout << "b: " << b.str() << '\n';

    if(b != c){
        std::cout << "Add failed - expected: " << c.str() << "real: " << b.str() <<'\n';
        return 1;
    }


    return 0;
}

int check_sub(void) {
    std::cout << "Starting " << __func__ << '\n';
    avx::Int256 a({1, 3, 5, 2, 8, 9, 7, 10});
    avx::Int256 c(std::array<int,8>{-4, -2, 0, -3, 3, 4, 2, 5});
    std::cout << "a: " << a.str() << '\n';
    avx::Int256 b = a - 5;
    std::cout << "b: " << b.str() << '\n';

    if(b != c){
        std::cout << "Sub failed - expected: " << c.str() << "real: " << b.str() <<'\n';
        return 1;
    }

    return 0;
}

int check_mul(void) {
    std::cout << "Starting " << __func__ << '\n';
    avx::Int256 a({1, 3, 5, 2, 8, 9, 7, 10});

    std::cout << "a: " << a.str() << '\n';
    avx::Int256 b = a * 5;
    std::cout << "b: " << b.str() << '\n';

    if(b == avx::Int256({5, 15, 25, 10, 40, 45, 35, 50}))
        return 0;
    
    return 1;
}

int check_div(void) {
    std::cout << "Starting " << __func__ << '\n';
    avx::Int256 a({2, 4, 8, 16, 32, 64, 128, 256});

    avx::Int256 b = a / 3;

    avx::Int256 c({0, 1, 2, 5, 10, 21, 42, 85});

    std::cout << "a: " << a.str() << '\n';
    std::cout << "b: " << b.str() << '\n';
    std::cout << "c: " << c.str() << '\n';

    if(c != b)
        return 1;

    return 0;
}


int main(int argc, char* argv[]){
    int res = 0;
    res |= check_create();
    res |= check_add();
    res |= check_sub();
    res |= check_mul();
    res |= check_div();
    
    return res;
}