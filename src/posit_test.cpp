#include <iostream>
#include <iomanip>
#include <cmath>
#include "posit8.h"

// Helper to print Posit8 details
void printPosit(const std::string& label, const Posit8& p) {
    std::cout << label << " Posit8: 0x" 
              << std::hex << std::setw(2) << std::setfill('0') << int(p.bits) 
              << " (float approx: " << std::dec << p.toFloat() << ")\n";
}

// Test multiplication and print results
void testMultiply(const Posit8& a, const Posit8& b) {
    printPosit("Input A", a);
    printPosit("Input B", b);
    Posit8 res = a * b;
    printPosit("Result A * B", res);
    float expected = a.toFloat() * b.toFloat();
    std::cout << "Expected float multiplication: " << expected << "\n\n";
}

int main() {
    std::cout << "=== Posit8 Multiplication Testbench ===\n\n";

    // Test 1: Regular values
    Posit8 a(0x30); // ~0.75
    Posit8 b(0x48); // ~2.5
    std::cout << "Test 1: Regular values\n";
    testMultiply(a, b);

    // Test 2: Zero multiplication
    Posit8 zero(0x00);
    std::cout << "Test 2: Zero multiplication\n";
    testMultiply(zero, b);

    // Test 3: NaR multiplication
    Posit8 nar(0x80);
    std::cout << "Test 3: NaR multiplication\n";
    testMultiply(nar, b);

    // Test 4: Both zero
    std::cout << "Test 4: Both zero\n";
    testMultiply(zero, zero);

    // Test 5: Large positive and negative values
    Posit8 largePos(0x7F); // max posit positive
    Posit8 largeNeg(0x81); // max posit negative (two's complement of 0x7F)
    std::cout << "Test 5: Large positive and negative values\n";
    testMultiply(largePos, largeNeg);

    // Test 6: Very small positive value (subnormal)
    Posit8 smallPos(0x01);
    std::cout << "Test 6: Very small positive value\n";
    testMultiply(smallPos, a);

    // Test 7: Multiplying by one (identity)
    Posit8 one(0x40); // posit representation of 1.0
    std::cout << "Test 7: Multiplying by one\n";
    testMultiply(a, one);

    return 0;
}
