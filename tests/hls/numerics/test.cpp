#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

#include "../test_common.h"
#include "common.h"

extern "C" void top(double in_a, double in_b, double out[20]);

int main()
{
    std::cout << "--- Starting Numerics Test ---" << std::endl;

    int error_count = 0;
    double max_error = 0.0;
    double thresh_error = 1.0e-01;

    for (int i = 0; i < NUM_TEST_CASES; i++)
    {
        double a = TEST_VECTORS[i][0];
        double b = TEST_VECTORS[i][1];

        std::string a_str = format_double(a);
        std::string b_str = format_double(b);

        std::cout << "\nTest case " << (i + 1) << "/" << NUM_TEST_CASES
                  << ": " << a_str << "  " << b_str << std::endl;

        // Software (golden) result
        double sw_result[64];
        test<double>(a, b, sw_result);

        // Hardware (HLS) result
        double hw_result[64];
        top(a, b, hw_result);

        for (int i = 0; i < 64; i++)
        {
            double error = std::fabs(sw_result[i] - hw_result[i]);
            if(sw_result[i] != 0)
                error /= sw_result[i];
                
            std::cout << "Error: " << error << std::endl;

            if (error > thresh_error)
            {
                std::cout << "Error too high: " << sw_result[i] << " vs " << hw_result[i] << std::endl;
                error_count++;
            }
            /*
            // Compare results
            if (check_error("Addition", sw_result, hw_result, max_error, a_str, b_str))
            {
                error_count++;
            }
            */
        }
    }

    std::cout << "\n--- Addition Test Summary ---" << std::endl;
    std::cout << "Total test cases: " << NUM_TEST_CASES << std::endl;
    std::cout << "Failed test cases: " << error_count << std::endl;
    std::cout << "Maximum relative error: " << max_error << std::endl;

    // if (max_error > thresh_error)
    //     return 1;

    return (error_count > 0) ? 1 : 0;
}
