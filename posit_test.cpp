#include <iostream>
#include "posit.h"

#define N 32
#define ES 2

typedef Posit<N, ES> posit32_t;

int main() {
    float a = 1.5f;
    float b = 2.25f;

    // Construct posit numbers
    posit32_t p1(a);
    posit32_t p2(b);

    // Convert back to float for display
    std::cout << "p1 = " << float(p1) << std::endl;
    std::cout << "p2 = " << float(p2) << std::endl;

    // Arithmetic
    posit32_t sum = p1 + p2;
    posit32_t diff = p1 - p2;
    posit32_t prod = p1 * p2;
    posit32_t quot = p1 / p2;

    std::cout << "Sum (p1 + p2)      = " << float(sum) << std::endl;
    std::cout << "Difference (p1 - p2) = " << float(diff) << std::endl;
    std::cout << "Product (p1 * p2)   = " << float(prod) << std::endl;
    std::cout << "Quotient (p1 / p2)  = " << float(quot) << std::endl;

    return 0;
}
