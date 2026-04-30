#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include "linalg/linalg.h"

// Declare the top-level function to be tested
extern "C" void top(double in_0,
                    double in_1,
                    double in_2,
                    double in_3,
                    double in_4,
                    double in_5,
                    double in_6,
                    double in_7,
                    double in_8,
                    double &out);

static bool check_rel_error(const std::string &name, double expected, double actual, double &max_err, double thresh = 1e-6)
{
    double err;
    if (expected != 0.0)
        err = std::fabs(expected - actual) / std::fabs(expected);
    else
        err = std::fabs(expected - actual);
    if (err > max_err)
        max_err = err;
    if (err > thresh)
    {
        std::cerr << "ERROR: " << name << " mismatch: exp=" << std::setprecision(15) << expected
                  << " act=" << std::setprecision(15) << actual << " relerr=" << err << std::endl;
        return true;
    }
    return false;
}

int main()
{
    // Print which numeric format this build is testing
#if defined(FORMAT_POSIT)
    std::cout << "--- Starting Posit Determinant Test ---" << std::endl;
#elif defined(FORMAT_FLOAT)
    std::cout << "--- Starting native float Determinant Test ---" << std::endl;
#else
    std::cout << "--- Starting FloatX Determinant Test ---" << std::endl;
#endif

    int errors = 0;

    // 0: determinant
    {
        double out;
        // Matrix (row-major)
        double m[9] = {
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0};

        top(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], out);
        // Golden determinant
        double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        double maxe = 0.0;
        errors += check_rel_error("det", det, out, maxe);
    }

    if (errors == 0)
    {
        std::cout << "\nSUCCESS: All linalg HLS tests passed!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nFAILURE: " << errors << " mismatches found." << std::endl;
        return 1;
    }
}
