// blas_lapack_bench.cpp
//
// Quick benchmark to see if BLAS/LAPACK performance (and threading) is your bottleneck.
// Measures:
//  1) Dense SPD Cholesky solve: dposv (factor+solve)
//  2) Split factor/solve: dpotrf + repeated dpotrs
//  3) A pure BLAS-3 kernel: dgemm (often the real speed indicator)
//
// Compile (OpenBLAS example):
//   g++ -O3 -march=native -DNDEBUG blas_lapack_bench.cpp -o bench \
//       -llapacke -llapack -lopenblas
//
// Run (control threads for fair comparison):
//   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bench
//   OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 ./bench
//
// If you use MKL, link with MKL and set MKL_NUM_THREADS.

#include <lapacke.h>

#ifdef USE_CBLAS
#include <cblas.h>
#else
// We can still call Fortran dgemm symbol if you don’t want cblas, but cblas is easiest.
// If you don't have cblas, compile with -DUSE_CBLAS and link to the BLAS that provides it.
// Many OpenBLAS installs provide cblas by default.
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

static inline double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void make_spd(std::vector<double>& A, int n, double lambda, uint32_t seed)
{
    // Build SPD matrix A = R^T R + lambda I
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<double> R((size_t)n * (size_t)n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            R[(size_t)i + (size_t)j * (size_t)n] = nd(rng); // column-major

    // A = R^T R
    std::fill(A.begin(), A.end(), 0.0);
#ifdef USE_CBLAS
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, n, n, 1.0,
                R.data(), n,
                R.data(), n,
                0.0,
                A.data(), n);
#else
    // Fallback: naive (slower) but keeps the benchmark self-contained.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += R[(size_t)k + (size_t)i * (size_t)n] * R[(size_t)k + (size_t)j * (size_t)n];
            A[(size_t)i + (size_t)j * (size_t)n] = sum;
        }
#endif

    // Add lambda I
    for (int i = 0; i < n; ++i)
        A[(size_t)i + (size_t)i * (size_t)n] += lambda;
}

static double bench_dposv(int n, int iters)
{
    std::vector<double> A((size_t)n * (size_t)n);
    std::vector<double> b((size_t)n);

    // Warmup data
    make_spd(A, n, 1e-2, 123);
    for (int i = 0; i < n; ++i) b[i] = 1.0 / (1.0 + i);

    // Warmup
    {
        auto A0 = A;
        auto x0 = b;
        int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, 1, A0.data(), n, x0.data(), n);
        if (info != 0) { std::cerr << "dposv warmup failed info=" << info << "\n"; std::exit(1); }
    }

    double t0 = now_ms();
    for (int it = 0; it < iters; ++it)
    {
        auto A0 = A;     // dposv overwrites
        auto x0 = b;
        int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, 1, A0.data(), n, x0.data(), n);
        if (info != 0) { std::cerr << "dposv failed info=" << info << "\n"; std::exit(1); }
        // stop optimizer
        volatile double sink = x0[0];
        (void)sink;
    }
    double t1 = now_ms();
    return (t1 - t0) / iters;
}

static double bench_dpotrf_dpotrs(int n, int nrhs, int iters)
{
    std::vector<double> A((size_t)n * (size_t)n);
    std::vector<double> B((size_t)n * (size_t)nrhs);

    make_spd(A, n, 1e-2, 456);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            B[(size_t)i + (size_t)j * (size_t)n] = std::sin(0.01 * (i + 1) * (j + 1));

    // Warmup
    {
        auto A0 = A;
        int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, A0.data(), n);
        if (info != 0) { std::cerr << "dpotrf warmup failed info=" << info << "\n"; std::exit(1); }
        auto X0 = B;
        info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', n, nrhs, A0.data(), n, X0.data(), n);
        if (info != 0) { std::cerr << "dpotrs warmup failed info=" << info << "\n"; std::exit(1); }
    }

    double t0 = now_ms();
    for (int it = 0; it < iters; ++it)
    {
        auto A0 = A;
        int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, A0.data(), n);
        if (info != 0) { std::cerr << "dpotrf failed info=" << info << "\n"; std::exit(1); }

        auto X0 = B;
        info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', n, nrhs, A0.data(), n, X0.data(), n);
        if (info != 0) { std::cerr << "dpotrs failed info=" << info << "\n"; std::exit(1); }

        volatile double sink = X0[0];
        (void)sink;
    }
    double t1 = now_ms();
    return (t1 - t0) / iters;
}

static double bench_dgemm(int n, int iters)
{
#ifndef USE_CBLAS
    (void)n; (void)iters;
    std::cerr << "DGEMM benchmark disabled (compile with -DUSE_CBLAS and link CBLAS/OpenBLAS/MKL)\n";
    return -1.0;
#else
    std::vector<double> A((size_t)n * (size_t)n);
    std::vector<double> B((size_t)n * (size_t)n);
    std::vector<double> C((size_t)n * (size_t)n);

    std::mt19937 rng(789);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (auto& v : A) v = ud(rng);
    for (auto& v : B) v = ud(rng);
    std::fill(C.begin(), C.end(), 0.0);

    // Warmup
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), n);

    double t0 = now_ms();
    for (int it = 0; it < iters; ++it)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), n);
        volatile double sink = C[0];
        (void)sink;
    }
    double t1 = now_ms();
    return (t1 - t0) / iters;
#endif
}

int main(int argc, char** argv)
{
    int n = 1042;
    int iters = 10;
    int nrhs = 19; // similar to your Schur multi-RHS (gv + pose cols)

    if (argc >= 2) n = std::atoi(argv[1]);
    if (argc >= 3) iters = std::atoi(argv[2]);
    if (argc >= 4) nrhs = std::atoi(argv[3]);

    std::cout << "n=" << n << " iters=" << iters << " nrhs=" << nrhs << "\n";
    std::cout << "Tip: set OPENBLAS_NUM_THREADS / MKL_NUM_THREADS / OMP_NUM_THREADS\n\n";

    double t_dposv = bench_dposv(n, iters);
    std::cout << "dposv (factor+solve) avg: " << t_dposv << " ms\n";

    double t_split = bench_dpotrf_dpotrs(n, nrhs, iters);
    std::cout << "dpotrf+dpotrs (nrhs=" << nrhs << ") avg: " << t_split << " ms\n";

    double t_gemm = bench_dgemm(n, iters);
    if (t_gemm > 0.0)
        std::cout << "dgemm (n x n) avg: " << t_gemm << " ms\n";

    return 0;
}