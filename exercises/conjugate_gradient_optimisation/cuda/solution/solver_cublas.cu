#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "errors.hpp"
#include "solver.hpp"

// Wrapper struct providing a cuBLAS handle with automatic setup and teardown.
struct CublasHandle {
  cublasHandle_t handle;
  CublasHandle() {
    cublasCreate(&handle);
    cublasSetStream(handle, 0); // share the default stream with our kernels
  }
  ~CublasHandle() { cublasDestroy(handle); }
};
static CublasHandle cublas;

// Dense matrix-vector product: y = A * x. A is row-major, so cuBLAS sees A^T
// => CUBLAS_OP_T recovers A*x.
void matvec(real *y, const real *A, const real *x, const int n) {
  const real one = 1.0f, zero = 0.0f;
#ifdef SINGLE_PRECISION
  cublasSgemv(cublas.handle, CUBLAS_OP_T, n, n, &one, A, n, x, 1, &zero, y, 1);
#else
  cublasDgemv(cublas.handle, CUBLAS_OP_T, n, n, &one, A, n, x, 1, &zero, y, 1);
#endif
  CHECK_LAST_CUDA_ERROR();
}

// AXPBY operation: y = alpha * x + beta * y.
void axpby(real *y, const real *x, const real alpha, const real beta,
           const int n) {
  if (beta != 1.0f) {
#ifdef SINGLE_PRECISION
    cublasSscal(cublas.handle, n, &beta, y, 1);
#else
    cublasDscal(cublas.handle, n, &beta, y, 1);
#endif
    CHECK_LAST_CUDA_ERROR();
  }
#ifdef SINGLE_PRECISION
  cublasSaxpy(cublas.handle, n, &alpha, x, 1, y, 1);
#else
  cublasDaxpy(cublas.handle, n, &alpha, x, 1, y, 1);
#endif
  CHECK_LAST_CUDA_ERROR();
}

// Dot product: result = sum(a[i] * b[i]).
real dot(const real *a, const real *b, const int n) {
  real result = 0.0f;
#ifdef SINGLE_PRECISION
  cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
#else
  cublasDdot(cublas.handle, n, a, 1, b, 1, &result);
#endif
  CHECK_LAST_CUDA_ERROR();
  return result;
}

// Solve A*x = b using the conjugate gradient method.
int cg_solve(real *x, const real *A, const real *b, const int n,
             const int max_iter) {
  real *r, *p, *A_times_p;
  CHECK_CUDA_ERROR(cudaMalloc(&r, n * sizeof(real)));
  CHECK_CUDA_ERROR(cudaMalloc(&p, n * sizeof(real)));
  CHECK_CUDA_ERROR(cudaMalloc(&A_times_p, n * sizeof(real)));

  // Step 1: r_0 = b - A*x_0
  matvec(r, A, x, n);
  axpby(r, b, 1.0, -1.0, n);

  // Step 2: p_0 = r_0
  CHECK_CUDA_ERROR(
      cudaMemcpy(p, r, n * sizeof(real), cudaMemcpyDeviceToDevice));

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

    real residual_sq_new = dot(r, r, n);

    // This method is so good it crashes if the residual gets too small!
    if (residual_sq_new < std::numeric_limits<real>().epsilon())
      break;

    // Step 3e: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    real beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new / n);

    residual_sq_old = residual_sq_new;
  }

  CHECK_CUDA_ERROR(cudaFree(r));
  CHECK_CUDA_ERROR(cudaFree(p));
  CHECK_CUDA_ERROR(cudaFree(A_times_p));

  return n_iter;
}
