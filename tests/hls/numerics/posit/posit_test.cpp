#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

// Declare the top-level function to be tested
void posit_top(double in_a, double in_b, double &out_add, double &out_sub, double &out_mul, double &out_div);

// Helper function to check for errors
bool check_error(const std::string &op_name, double expected, double actual, double &max_error)
{
    double error = 0.0;
    if (std::isinf(expected) && std::isinf(actual) && (expected > 0 == actual > 0))
    {
        // Matching infinities are okay
        error = 0.0;
    }
    else if (std::isnan(expected) && std::isnan(actual))
    {
        // Matching NaNs are okay
        error = 0.0;
    }
    else if (expected != 0)
    {
        error = std::fabs(expected - actual) / std::fabs(expected);
    }
    else
    {
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
        std::cerr << "ERROR: " << op_name << " mismatch!\n"
                  << "  Expected: " << std::setprecision(15) << expected << "\n"
                  << "  Actual:   " << std::setprecision(15) << actual << "\n"
                  << "  Relative Error: " << error << std::endl;
        return true; // Error found
    }
    return false; // No error
}

int main(void)
{
    std::cout << "--- Starting FloatX HLS C-Simulation Testbench ---" << std::endl;

    // Test data
    double test_vectors[][2] = {
        {1.0e6, 1.0e-6},
        {3.14159, 2.71828},
        {100.0, 0.5},
        {-50.25, 25.125},
        {0.0, 123.456},
        {123.456, 0.0} // Test division by zero
    };

    int error_count = 0;
    double max_add_err = 0.0, max_sub_err = 0.0, max_mul_err = 0.0, max_div_err = 0.0;

    for (const auto &vec : test_vectors)
    {
        double a = vec[0];
        double b = vec[1];

        std::cout << "\nTesting with a = " << a << ", b = " << b << std::endl;

        // Software (golden) results
        double sw_add = a + b;
        double sw_sub = a - b;
        double sw_mul = a * b;
        double sw_div = (b != 0.0) ? (a / b) : std::numeric_limits<double>::infinity();

        // Hardware (HLS) results
        double hw_add, hw_sub, hw_mul, hw_div;
        posit_top(a, b, hw_add, hw_sub, hw_mul, hw_div);

        // Compare results
        if (check_error("Addition", sw_add, hw_add, max_add_err))
            error_count++;
        if (check_error("Subtraction", sw_sub, hw_sub, max_sub_err))
            error_count++;
        if (check_error("Multiplication", sw_mul, hw_mul, max_mul_err))
            error_count++;
        if (check_error("Division", sw_div, hw_div, max_div_err))
            error_count++;
    }

    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "Max relative error (Add): " << max_add_err << std::endl;
    std::cout << "Max relative error (Sub): " << max_sub_err << std::endl;
    std::cout << "Max relative error (Mul): " << max_mul_err << std::endl;
    std::cout << "Max relative error (Div): " << max_div_err << std::endl;

    if (error_count == 0)
    {
        std::cout << "\nSUCCESS: All tests passed!" << std::endl;
        return 0; // Success
    }
    else
    {
        std::cout << "\nFAILURE: " << error_count << " mismatches found." << std::endl;
        return 1; // Failure
    }
}
