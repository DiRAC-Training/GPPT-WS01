#include <thrust/device_vector.h>

const int X = 256;        // columns per row (loads per thread)
const int Y = 1 << 20;    // rows -> threads; grid = Y/256 = 4096 blocks

// Thread `row` walks its columns with stride X: adjacent threads are X floats
// apart, so each warp step scatters across 32 cache lines -> uncoalesced.
__global__ void row_sum_strided(float *data, float *out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int col = 0; col < X; ++col) sum += data[row * X + col];
  out[row] = sum;
}

// Same sum, but adjacent threads are 1 float apart at every step, so a warp
// reads one contiguous 128-byte segment per step -> coalesced.
__global__ void row_sum_coalesced(float *data, float *out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int col = 0; col < X; ++col) sum += data[col * Y + row];
  out[row] = sum;
}

int main() {
  int block = 256;
  int grid = Y / block;
  thrust::device_vector<float> data(X * Y, 1.0f);
  thrust::device_vector<float> out(Y);

  row_sum_strided<<<grid, block>>>(data.data().get(), out.data().get());
  row_sum_coalesced<<<grid, block>>>(data.data().get(), out.data().get());

  cudaDeviceSynchronize();
  return 0;
}
