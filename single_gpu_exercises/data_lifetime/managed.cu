#include <math.h>
#include <cufft.h>

const int N = 1 << 22;  // 4,194,304 samples = 64 MB of complex doubles
const int THREADS_PER_BLOCK = 256;

// Normalise on the host: the spectrum migrates off the GPU and back.
void run_naive(cufftHandle plan, cufftDoubleComplex *data) {
  cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  for (int i = 0; i < N; i++) {
    data[i].x /= N;
    data[i].y /= N;
  }
  cufftExecZ2Z(plan, data, data, CUFFT_INVERSE);
  cudaDeviceSynchronize();
}

__global__ void normalise(cufftDoubleComplex *data, double scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  data[idx].x *= scale;
  data[idx].y *= scale;
}

// Normalise on the GPU: data stays resident between the two FFTs.
void run_resident(cufftHandle plan, cufftDoubleComplex *data) {
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
  normalise<<<blocks, THREADS_PER_BLOCK>>>(data, 1.0 / N, N);
  cufftExecZ2Z(plan, data, data, CUFFT_INVERSE);
  cudaDeviceSynchronize();
}

int main() {
  cufftDoubleComplex *data;
  cudaMallocManaged(&data, sizeof(cufftDoubleComplex) * N);
  for (int i = 0; i < N; i++) {
    data[i].x = sin((double)i / N * 4.0 * M_PI);
    data[i].y = 0.0;
  }

  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

  run_naive(plan, data);
  run_resident(plan, data);

  cufftDestroy(plan);
  cudaFree(data);
  return 0;
}
