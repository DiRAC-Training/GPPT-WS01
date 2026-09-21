#include <math.h>
#include <cufft.h>

const int N = 1 << 22;  // 4,194,304 samples = 64 MB of complex doubles
const int THREADS_PER_BLOCK = 256;

// Normalise on the host: the spectrum is copied off the GPU, scaled, and sent back.
void run_naive(cufftHandle plan, cufftDoubleComplex *data_d,
               cufftDoubleComplex *host, cufftDoubleComplex *freq) {
  size_t bytes = sizeof(cufftDoubleComplex) * N;
  cudaMemcpy(data_d, host, bytes, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, data_d, data_d, CUFFT_FORWARD);
  cudaMemcpy(freq, data_d, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    freq[i].x /= N;
    freq[i].y /= N;
  }
  cudaMemcpy(data_d, freq, bytes, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, data_d, data_d, CUFFT_INVERSE);
  cudaMemcpy(host, data_d, bytes, cudaMemcpyDeviceToHost);
}

__global__ void normalise(cufftDoubleComplex *data, double scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  data[idx].x *= scale;
  data[idx].y *= scale;
}

// Normalise on the GPU: data stays resident between the two FFTs.
void run_resident(cufftHandle plan, cufftDoubleComplex *data_d,
                  cufftDoubleComplex *host) {
  size_t bytes = sizeof(cufftDoubleComplex) * N;
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cudaMemcpy(data_d, host, bytes, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, data_d, data_d, CUFFT_FORWARD);
  normalise<<<blocks, THREADS_PER_BLOCK>>>(data_d, 1.0 / N, N);
  cufftExecZ2Z(plan, data_d, data_d, CUFFT_INVERSE);
  cudaMemcpy(host, data_d, bytes, cudaMemcpyDeviceToHost);
}

int main() {
  size_t bytes = sizeof(cufftDoubleComplex) * N;

  cufftDoubleComplex *host, *freq;
  cudaMallocHost(&host, bytes);
  cudaMallocHost(&freq, bytes);
  for (int i = 0; i < N; i++) {
    host[i].x = sin((double)i / N * 4.0 * M_PI);
    host[i].y = 0.0;
  }

  cufftDoubleComplex *data_d;
  cudaMalloc(&data_d, bytes);

  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

  run_naive(plan, data_d, host, freq);
  run_resident(plan, data_d, host);

  cufftDestroy(plan);
  cudaFree(data_d);
  cudaFreeHost(host);
  cudaFreeHost(freq);
  return 0;
}
