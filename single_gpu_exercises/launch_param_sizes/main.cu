#include <thrust/device_vector.h>
#include <cstdlib>

const int N = 1 << 26;

// One thread per element, so the work is fixed by N regardless of block size.
__global__ void axpy_kernel(float *x, const float *y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  x[idx] = 2.0f * x[idx] + y[idx];
}

int main(int argc, char **argv) {
  int block_size = atoi(argv[1]);
  int n_blocks = (N + block_size - 1) / block_size;

  thrust::device_vector<float> x(N, 1.0f);
  thrust::device_vector<float> y(N, 2.0f);

  axpy_kernel<<<n_blocks, block_size>>>(x.data().get(), y.data().get(), N);
  cudaDeviceSynchronize();

  return 0;
}
