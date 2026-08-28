#include <thrust/device_vector.h>

const int THREADS_PER_BLOCK = 256;
const int N = 50;                 // per-array size -> the register-pressure lever
const long WORK = 50000;          // held constant so arithmetic volume is fixed

__global__ void register_kernel(double *data, int n, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  double a[N], b[N];
  for (int k = 0; k < N; ++k) { a[k] = data[idx] + k; b[k] = data[idx] - k; }

  // Cross-coupled and damped: every element stays live (register pressure) and
  // nothing folds. `iters` is a runtime argument so the recurrence can't close.
  for (int i = 0; i < iters; ++i) {
    for (int k = 0; k < N - 1; ++k) {
      a[k+1] = a[k+1]*0.99 + (a[k] + b[k])*0.005;
      b[k+1] = b[k+1]*0.99 + (a[k] + b[k])*0.005;
    }
  }

  data[idx] = a[N-1] + b[N-1];    // one write so the loops aren't dead code
}

int main() {
  int n = 1 << 22;
  int iters = WORK / (N - 1);
  thrust::device_vector<double> data(n, 1.0);
  register_kernel<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
      data.data().get(), n, iters);
  cudaDeviceSynchronize();
  return 0;
}
