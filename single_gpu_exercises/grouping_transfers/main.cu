#include <cuda_runtime.h>

const int N = 16'000'000;              // doubles -> 128 MB, moved H->D each pass
const int chunk_counts[] = {1, 10, 100, 1000};   // divides N evenly

int main() {
  size_t bytes = sizeof(double) * N;

  double *host, *dev;
  cudaMallocHost(&host, bytes);        // pinned host allocation
  cudaMalloc(&dev, bytes);
  for (int i = 0; i < N; ++i) host[i] = 1.0;

  // Same 128 MB every pass, only the number of memcpy calls changes.
  for (int chunks : chunk_counts) {
    size_t chunk_elems = N / chunks;
    size_t chunk_bytes = chunk_elems * sizeof(double);
    for (int c = 0; c < chunks; ++c) {
      size_t offset = (size_t)c * chunk_elems;
      cudaMemcpy(dev + offset, host + offset, chunk_bytes, cudaMemcpyHostToDevice);
    }
  }

  cudaFree(dev);
  cudaFreeHost(host);
  return 0;
}
