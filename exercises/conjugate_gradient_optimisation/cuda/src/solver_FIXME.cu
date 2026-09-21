#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "errors.hpp"
#include "idx.hpp"
#include "solver.hpp"

#ifdef BLOCK_SIZE
const int block_size = BLOCK_SIZE;
#else
const int block_size = 32;
#endif

// Wrapper struct providing a cuBLAS handle with automatic setup and teardown.
struct CublasHandle {
  cublasHandle_t handle;
  CublasHandle() { cublasCreate(&handle); }
  ~CublasHandle() { cublasDestroy(handle); }
};
static CublasHandle cublas;

// CUDA kernels

__global__ void matvec_kernel(real *y, const real *matrix_data, const real *x,
                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TODO EX2:
    //     compute y[i] = sum over j of vals[i * n + j] * x[j]
  }
}

__global__ void axpby_kernel(real *y, const real *x, real alpha, real beta,
                             int n) {
  // TODO EX3:
  //     compute y[i] = alpha * x[i] + beta * y[i]
}

/// Copied from solver.cpp
void matvec_cpu(real *y, const real *A, const real *x, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    real sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[idx(i, j, n)] * x[j];
    y[i] = sum;
  }
}

// Once matvec_kernel has been implemented above, this launcher
// will run on the GPU with no further changes needed.
void matvec(real *y, const real *A, const real *x, const int n) {
  // TODO EX2 comment the line below out to disable the CPU version
  matvec_cpu(y, A, x, n);

  // TODO EX2 uncomment the lines below to enable the GPU version
  // int n_blocks = (n + block_size - 1) / block_size;
  // matvec_kernel<<<n_blocks, block_size>>>(y, A, x, n);
  // CHECK_LAST_CUDA_ERROR();
}

// TODO EX3: replace the CPU fallback below with a launch of
// axpby_kernel, following the same pattern as matvec() above. The fallback
// is only here so that intermediate builds (after porting matvec in step 2
// but before porting axpby) still produce correct answers; it relies on x
// and y being in managed memory, which you arranged in step 1.
void axpby(real *y, const real *x, const real alpha, const real beta,
           const int n) {
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i] + beta * y[i];
}

// Provided: dot product via cuBLAS. Do not modify.
real dot(const real *a, const real *b, const int n) {
  real result = 0.0f;
  cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
  CHECK_LAST_CUDA_ERROR();
  return result;
}

// Solve A*x = b using the conjugate gradient method.
int cg_solve(real *x, const real *A, const real *b, const int n,
             const int max_iter) {
  // TODO EX1: convert these allocations to cudaMallocManaged
  real *r = new real[n];
  real *p = new real[n];
  real *A_times_p = new real[n];

  // Step 1: r_0 = b - A*x_0
  matvec(r, A, x, n);
  for (int i = 0; i < n; i++)
    r[i] = b[i] - r[i];

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

    // Step 3d: convergence check -- stop if ||r_{k+1}|| < tol
    real residual_sq_new = dot(r, r, n);

    // This method is so good it crashes if the residual gets too small!
    if (residual_sq_new < std::numeric_limits<real>().epsilon())
      break;

    // Step 3e: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
    //          p_{k+1} = r_{k+1} + beta_k * p_k
    real beta = residual_sq_new / residual_sq_old;
    axpby(p, r, 1.0, beta, n);

    std::printf("%d: r = %.6e\n", n_iter, residual_sq_new);

    residual_sq_old = residual_sq_new;
  }

  // TODO EX1: convert these deallocations to cudaFree
  delete[] r;
  delete[] p;
  delete[] A_times_p;

  return n_iter;
}
