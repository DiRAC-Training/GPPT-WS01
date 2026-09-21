#include <cstdio>
#include <cstring>
#include <limits>

#include "idx.hpp"
#include "precision.hpp"
#include "solver.hpp"

// Dense matrix-vector product: y = A * x.
void matvec(real *y, const real *A, const real *x, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    real sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[idx(i, j, n)] * x[j];
    y[i] = sum;
  }
}

// AXPBY operation: y = alpha * x + beta * y.
void axpby(real *y, const real *x, const real alpha, const real beta,
           const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i] + beta * y[i];
}

// Dot product: result = sum(a[i] * b[i]).
real dot(const real *a, const real *b, const int n) {
  real sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n; i++)
    sum += a[i] * b[i];
  return sum;
}

// Solve A*x = b using the conjugate gradient method.
int cg_solve(real *x, const real *A, const real *b, const int n,
             const int max_iter) {
  real *r = new real[n];
  real *p = new real[n];
  real *A_times_p = new real[n];

  // Step 1: r_0 = b - A*x_0
  matvec(r, A, x, n);
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    r[i] = b[i] - r[i];
  }

  // Step 2: p_0 = r_0
  std::memcpy(p, r, n * sizeof(real));

  real residual_sq_old = dot(r, r, n);
  int n_iter;

  for (n_iter = 0; n_iter < max_iter; n_iter++) {
    // Step 3a: alpha_k = (r_k . r_k) / (p_k . A*p_k)
    matvec(A_times_p, A, p, n);
    real alpha = residual_sq_old / dot(p, A_times_p, n);

    // Step 3b: x_{k+1} = x_k + alpha_k * p_k
    axpby(x, p, alpha, 1.0, n);

    // Step 3c: r_{k+1} = r_k - alpha_k * A*p_k
    axpby(r, A_times_p, -alpha, 1.0, n);

    // Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    real residual_sq_new = dot(r, r, n);
    // This method is so good it crashes if the residual gets too small!
    if (residual_sq_new < std::numeric_limits<real>().epsilon())
      break;

    real beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);
    residual_sq_old = residual_sq_new;

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new / n);
  }

  delete[] r;
  delete[] p;
  delete[] A_times_p;

  return n_iter;
}
