#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Size of problem
const int N = 1 << 25;

// Kernel launch parameters
const int BLOCK_SIZE = 128;

// Size of shared memory in multiples of block size
const int SHARED_STRIDES = 96;

// Sums the shared memory values from adjacent threads
__global__ void shared_memory_kernel(float* data, long data_size) {

  extern __shared__ float shared_x[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_size) return;
  
  int threadId = threadIdx.x;
  
  for (int i = 0; i < SHARED_STRIDES; i++){
    shared_x[i * blockDim.x + threadId] = i + 1 * threadId * data[idx];
  }
  
  __syncthreads();

  for (int i = 0; i < SHARED_STRIDES; i++){
    if (threadId == 0) break;
    data[idx] += shared_x[i * blockDim.x + threadId - 1];
  }
}


int main() {
  
  // Calculate the number of blocks based on our chosen inputs
  int n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Determine the amount of shared memory we want to request for each block
  int shared_memory_size = SHARED_STRIDES * BLOCK_SIZE;
  
  thrust::host_vector<float> data_h(N);
  srand(42);
  for (int i = 0; i < N; i++){
    data_h[i] = (float)rand() / RAND_MAX;
  }
  thrust::device_vector<float> data_d = data_h;

  printf("Shared mem size %zu", shared_memory_size * sizeof(float));
  
  shared_memory_kernel<<<n_blocks, BLOCK_SIZE, shared_memory_size * sizeof(float)>>>(data_d.data().get(),N);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    return 1;
  }

  return 0;
}
