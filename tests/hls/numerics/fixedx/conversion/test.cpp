#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

#include "../../test_common.h"

extern "C" void top(double in_a, double &out);

int main()
{
    std::cout << "--- Starting FloatX double Test ---" << std::endl;

    int error_count = 0;
    double max_error = 0.0;
    double thresh_error = 5.15e-08;

    for (int i = 0; i < NUM_TEST_CASES; i++)
    {
        double a = TEST_VECTORS[i][0];

        std::string a_str = format_double(a);

        std::cout << "\nTest case " << (i + 1) << "/" << NUM_TEST_CASES
                  << ": " << a_str << std::endl;

        // Software (golden) result
        double sw_result = a;

        // Hardware (HLS) result
        double hw_result;
        top(a, hw_result);

        double error = std::fabs(sw_result - hw_result);
        if (sw_result != 0.0)
            error = error / std::fabs(sw_result);

        std::cout << "Error: " << error << std::endl;

        if (error > thresh_error)
        {
            std::cout << "Error too high" << std::endl;
            error_count++;
        }
        /*
        // Compare results
        if (check_error("Conversion", sw_result, hw_result, max_error, a_str, a_str))
        {
            error_count++;
        }
        */
    }

    std::cout << "\n--- Conversion Test Summary ---" << std::endl;
    std::cout << "Total test cases: " << NUM_TEST_CASES << std::endl;
    std::cout << "Failed test cases: " << error_count << std::endl;
    std::cout << "Maximum relative error: " << max_error << std::endl;

    // if (max_error > thresh_error)
    //     return 1;

    return (error_count > 0) ? 1 : 0;
}
