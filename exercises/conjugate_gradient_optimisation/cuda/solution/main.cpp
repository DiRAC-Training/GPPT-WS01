#include <chrono>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

#include "idx.hpp"
#include "precision.hpp"
#include "solver.hpp"
#include "test.hpp"

#include "util.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

static std::mt19937 rng;

void calc_b(real *b, const real *A, const real *x, const int n);
void fill_rand_vec(real *x, int size, real min, real max);
void generate_positive_definite(real *A, int n, bool poorly_conditioned_matrix);
bool all_tests_pass();

int main(int argc, char *argv[]) {
  // Parameters
  const int seed_in = get_argval<int>(argv, argv + argc, "--seed", -1);
  const uint n = get_argval<uint>(argv, argv + argc, "-n", 8192);
  const uint cg_max_iter = get_argval<uint>(argv, argv + argc, "--cg_max_iter", 32);
  const bool well_conditioned = get_arg(argv, argv + argc, "--well_conditioned");
  const bool disable_unit_tests =
      get_arg(argv, argv + argc, "--disable_unit_tests");
  const bool only_unit_tests = get_arg(argv, argv + argc, "--only_unit_tests");

  // Check tests
  bool tests_passed = true;
  if (!disable_unit_tests) {
    tests_passed = all_tests_pass();
    if (!tests_passed)
      return -1;

    if (tests_passed && only_unit_tests) {
      std::cout << "All tests passed!\n";
      return 0;
    }
  }

  std::cout << "\n";
  std::cout << "Matrix size: " << n << " x " << n << "\n";
  std::cout << "Max iterations: " << cg_max_iter << "\n";

  std::mt19937::result_type seed = 0;
  if(seed_in == -1) {
    std::cout << "Generating random seed\n";
    seed = static_cast<std::mt19937::result_type>(
        std::chrono::system_clock::now().time_since_epoch().count());
  } else {
    seed = seed_in;
  }
  std::cout << "Using RNG seed: " << seed << "\n";
  rng.seed(seed);

  real *A = nullptr;
  cudaMallocManaged(&A, n * n * sizeof(real));
  real *b = nullptr;
  cudaMallocManaged(&b, n * sizeof(real));
  real *x = nullptr;
  cudaMallocManaged(&x, n * sizeof(real));
  real *x_soln = new real[n];

  std::cout << "\n";

  // Initial conditions
  if(well_conditioned) {
    std::cout << "Matrix is POORLY conditioned\n";
  } else {
    std::cout << "Matrix is WELL conditioned\n";
  }
  generate_positive_definite(A, n, !well_conditioned); // Generate positive def matrix
  std::cout << "Generating random solution\n";
  fill_rand_vec(x_soln, n, -1.0, 1.0); // Generate a random solution vector
  std::cout << "Generating right hand side\n";
  calc_b(b, A, x_soln,
         n); // Multiple matrix with solution to get RHS of equation, b
  std::fill(x, x + n, 0.0);

  std::cout << "\n";

  // Solve
  std::cout << "Starting solver\n";
  auto start = high_resolution_clock::now();
  int iters;
  iters = cg_solve(x, A, b, n, cg_max_iter);
  auto stop = high_resolution_clock::now();
  auto duration =
      duration_cast<std::chrono::microseconds>(stop - start).count();

  std::cout << "Performed " << iters << " iterations" << std::endl;
  std::cout << "Solve time: " << duration << " us" << std::endl;
  std::cout << "Time per iteration: " << duration / iters << " us" << std::endl;

  real av_error2 = 0.0;
  for (int i = 0; i < n; ++i) {
    av_error2 += std::fabs(x_soln[i] - x[i]);
  }
  av_error2 /= n;

  std::cout << "Average error = " << std::sqrt(av_error2) << "\n";

  cudaFree(A);
  cudaFree(b);
  cudaFree(x);

  delete[] x_soln;

  return 0;
}

// ============================================================
// YOU DO NOT NEED TO READ BELOW THIS LINE
// ============================================================

/// Generate b from Ax = b
void calc_b(real *b, const real *A, const real *x, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    b[i] = 0.0;

    for (int j = 0; j < n; ++j) {
      b[i] += A[idx(i, j, n)] * x[j];
    }
  }
}

/// Fill x with random values in [min, max)
void fill_rand_vec(real *x, int size, real min, real max) {
  std::uniform_real_distribution<real> dist(min, max);
  for (int i = 0; i < size; ++i) {
    x[i] = dist(rng);
  }
}

/// Create a random, positive-definite matrix
void generate_positive_definite(real *A, int n, bool poorly_conditioned_matrix) {
  std::cout << "Generating matrix\n";
  // Create a totally random matrix
  auto B = std::vector<real>(n * n);
  fill_rand_vec(B.data(), n * n, 0.0, 1.0);

// Make a symmetric matrix
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[idx(i, j, n)] = (B[idx(i, j, n)] + B[idx(j, i, n)]) / 2.0;
    }
  }

  real diag_shift;
  if(poorly_conditioned_matrix) {
    // Change DIAG_SCALE to tweak how well-conditioned the matrix is (changes number of iterations to convergence)
    // 0.42 is stable and results in a fairly poorly-conditioned matrix (i.e. lots of iterations!)
    // Other values are untested
    const real DIAG_SCALE = 0.42;
    diag_shift = DIAG_SCALE * std::sqrt(real(n));
  } else {
    // make the matrix extremely well conditioned (should converge very quickly)
    diag_shift = fmax(real(n) / 32, 1.0);
  }

  for (int i = 0; i < n; ++i)
    A[idx(i, i, n)] += diag_shift;
}

bool all_tests_pass() {
  bool all_passed = true;
  if (!test_matvec_identity()) {
    all_passed = false;
    std::cout << "matvec identity failed!\n";
  }

  if (!test_matvec_simple()) {
    all_passed = false;
    std::cout << "matvec simple failed!\n";
  }

  if (!test_dot()) {
    all_passed = false;
    std::cout << "dot failed!\n";
  }

  return all_passed;
}
