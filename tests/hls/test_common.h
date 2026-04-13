#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

// Helper function to check for errors
bool check_error(const std::string &op_name, double expected, double actual, double &max_error, const std::string &a_str, const std::string &b_str)
{
    double error = 0.0;

    // Handle special cases
    if (std::isinf(expected) && std::isinf(actual) && (expected > 0) == (actual > 0))
    {
        // Matching infinities with same sign
        error = 0.0;
    }
    else if (std::isnan(expected) && std::isnan(actual))
    {
        // Both are NaN
        error = 0.0;
    }
    else if (expected != 0)
    {
        // Relative error for non-zero expected
        error = std::fabs(expected - actual) / std::fabs(expected);
    }
    else
    {
        // Absolute error for zero expected
        error = std::fabs(expected - actual);
    }

    if (error > max_error)
    {
        max_error = error;
    }

    // Use a reasonable threshold for floating point comparisons
    const double threshold = 1e-4; // Relative error threshold

    if (error > threshold)
    {
        std::cerr << "ERROR: " << op_name << " mismatch!" << std::endl;
        std::cerr << "  a = " << a_str << ", b = " << b_str << std::endl;
        std::cerr << "  Expected: " << std::setprecision(15) << expected << std::endl;
        std::cerr << "  Actual:   " << std::setprecision(15) << actual << std::endl;
        std::cerr << "  Relative Error: " << error << std::endl;
        return true; // Error found
    }

    return false; // No error
}

// Test vectors for all operations
const double TEST_VECTORS[][2] = {
    {3.14159, 2.71828},     // Pi and e
    {100.0, 0.5},           // Large and small numbers
    {-50.25, 25.125},       // Negative and positive
    {1.0e6, 1.0e-6},        // Very large and very small
    {0.0, 123.456},         // Zero and non-zero
    {123.456, 0.0},         // Division by zero test case
    {1.0, 1.0},             // Identity cases
    {0.0, 0.0},             // Zero cases
    {-1.0, -1.0},           // Negative cases
    {1.2345e-10, 9.8765e10} // Very small and very large
};

const int NUM_TEST_CASES = sizeof(TEST_VECTORS) / sizeof(TEST_VECTORS[0]);

// Function to format double for output
std::string format_double(double val)
{
    std::ostringstream oss;
    if (std::abs(val) > 1e4 || (std::abs(val) < 1e-4 && val != 0.0))
    {
        oss << std::scientific << std::setprecision(6) << val;
    }
    else
    {
        oss << std::fixed << std::setprecision(6) << val;
    }
    return oss.str();
}

std::string format_float(float val)
{
    std::ostringstream oss;
    if (std::abs(val) > 1e4 || (std::abs(val) < 1e-4 && val != 0.0))
    {
        oss << std::scientific << std::setprecision(6) << val;
    }
    else
    {
        oss << std::fixed << std::setprecision(6) << val;
    }
    return oss.str();
}

#endif // TEST_FLOATX_COMMON_H
