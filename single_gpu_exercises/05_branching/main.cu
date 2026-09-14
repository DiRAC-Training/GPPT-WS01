#include <thrust/device_vector.h>
#include <cstdlib>

const int N = 1 << 22;
const int WORK = 1000;   // total iterations, split unevenly between the two arms

// group = 1 -> warps diverge; group = 32 -> warps coherent.
__global__ void branching_kernel(float *data, int n, int group) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // Different counts (that sum to WORK) -> the arms can't be fused into a
  // branchless single loop through inlining, so the branching is real.
  int reps_add = WORK/2 - WORK/10;   // 400
  int reps_sub = WORK/2 + WORK/10;   // 600

  float value = data[idx];
  if ((idx / group) % 2 == 0) {
    for (int i = 0; i < reps_add; ++i) value = value * 1.001f + 1.0f;
  } else {
    for (int i = 0; i < reps_sub; ++i) value = value * 0.999f - 1.0f;
  }
  data[idx] = value;
}

int main(int argc, char **argv) {
  int group = atoi(argv[1]);          // 1 = divergent, 32 = coherent
  int block = 256;
  thrust::device_vector<float> data(N, 1.0f);
  branching_kernel<<<(N + block - 1) / block, block>>>(data.data().get(), N, group);
  cudaDeviceSynchronize();
  return 0;
}
