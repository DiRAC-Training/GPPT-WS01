#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <mpi.h>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            fprintf(stderr,                                                             \
                    "ERROR: MPI call \"%s\" in line %d of file %s failed "              \
                    "with %s (%d).\n",                                                  \
                    #call, __LINE__, __FILE__, mpi_error_string, mpi_status);         \
        }                                                                             \
    }

#include <cuda_runtime.h>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with %s (%d).\n",                                                       \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

constexpr int block_size = 256;

// Fill the send buffer with a unique value per (rank, message, element) so
// that correctness can be verified after communication.
__global__ void init_kernel(real* data, int rank, int num_messages, int msg_size,
                            int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_messages * stride;
    if (idx < total) {
        int m = idx / stride;
        int i = idx % stride;
        if (i < msg_size)
            data[idx] = static_cast<real>(rank * 1000000 + m * 10000 + i);
        else
            data[idx] = static_cast<real>(-1.0);
    }
}

// Pack strided messages into a contiguous buffer.
__global__ void pack_kernel(real* dst, const real* src, int num_messages,
                            int msg_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_messages * msg_size;
    if (idx < total) {
        int m = idx / msg_size;
        int i = idx % msg_size;
        dst[m * msg_size + i] = src[m * stride + i];
    }
}

// Unpack a contiguous buffer back into strided messages.
__global__ void unpack_kernel(real* dst, const real* src, int num_messages,
                              int msg_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_messages * msg_size;
    if (idx < total) {
        int m = idx / msg_size;
        int i = idx % msg_size;
        dst[m * stride + i] = src[m * msg_size + i];
    }
}

// Check that the first msg_size elements of each strided message match the
// expected values from source_rank.
int verify(const real* h_data, int source_rank, int num_messages, int msg_size,
           int stride) {
    for (int m = 0; m < num_messages; m++)
        for (int i = 0; i < msg_size; i++)
            if (h_data[m * stride + i] !=
                static_cast<real>(source_rank * 1000000 + m * 10000 + i))
                return 0;
    return 1;
}

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    const int num_messages = get_argval<int>(argv, argv + argc, "-num_messages", 128);
    const int msg_size     = get_argval<int>(argv, argv + argc, "-msg_size", 1024);
    const int stride       = get_argval<int>(argv, argv + argc, "-stride", msg_size + 128);
    const int num_iters    = get_argval<int>(argv, argv + argc, "-num_iters", 100);

    // GPU binding via local rank
    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                     MPI_INFO_NULL, &local_comm));
        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_free(&local_comm));
    }
    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    CUDA_RT_CALL(cudaSetDevice(local_rank % num_devices));
    CUDA_RT_CALL(cudaFree(0));

    // Ring topology: send right, receive left
    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;

    const int buf_size    = num_messages * stride;
    const int packed_size = num_messages * msg_size;

    real *d_send, *d_recv, *d_send_buf, *d_recv_buf;
    CUDA_RT_CALL(cudaMalloc(&d_send, buf_size * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&d_recv, buf_size * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&d_send_buf, packed_size * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&d_recv_buf, packed_size * sizeof(real)));

    real* h_recv;
    CUDA_RT_CALL(cudaMallocHost(&h_recv, buf_size * sizeof(real)));

    // ================================================================
    // Approach 1: Blocking MPI_Sendrecv — one call per message (baseline)
    // ================================================================
    init_kernel<<<(buf_size + block_size - 1) / block_size, block_size>>>(
        d_send, rank, num_messages, msg_size, stride);
    CUDA_RT_CALL(cudaMemset(d_recv, 0, buf_size * sizeof(real)));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    double t0 = MPI_Wtime();
    for (int iter = 0; iter < num_iters; iter++) {
        for (int m = 0; m < num_messages; m++) {
            MPI_CALL(MPI_Sendrecv(d_send + m * stride, msg_size, MPI_REAL_TYPE, next, 0,
                                  d_recv + m * stride, msg_size, MPI_REAL_TYPE, prev, 0,
                                  MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    double t1 = MPI_Wtime();

    CUDA_RT_CALL(cudaMemcpy(h_recv, d_recv, buf_size * sizeof(real),
                            cudaMemcpyDeviceToHost));
    int correct_blocking = verify(h_recv, prev, num_messages, msg_size, stride);
    MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &correct_blocking, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));

    // ================================================================
    // Approach 2: Packing kernel + non-blocking MPI
    // ================================================================
    init_kernel<<<(buf_size + block_size - 1) / block_size, block_size>>>(
        d_send, rank, num_messages, msg_size, stride);
    CUDA_RT_CALL(cudaMemset(d_recv, 0, buf_size * sizeof(real)));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    double t2 = MPI_Wtime();
    for (int iter = 0; iter < num_iters; iter++) {
        pack_kernel<<<(packed_size + block_size - 1) / block_size, block_size>>>(
            d_send_buf, d_send, num_messages, msg_size, stride);
        CUDA_RT_CALL(cudaDeviceSynchronize());

        MPI_Request reqs[2];
        MPI_CALL(MPI_Isend(d_send_buf, packed_size, MPI_REAL_TYPE, next, 0,
                           MPI_COMM_WORLD, &reqs[0]));
        MPI_CALL(MPI_Irecv(d_recv_buf, packed_size, MPI_REAL_TYPE, prev, 0,
                           MPI_COMM_WORLD, &reqs[1]));
        MPI_CALL(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE));

        unpack_kernel<<<(packed_size + block_size - 1) / block_size, block_size>>>(
            d_recv, d_recv_buf, num_messages, msg_size, stride);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    double t3 = MPI_Wtime();

    CUDA_RT_CALL(cudaMemcpy(h_recv, d_recv, buf_size * sizeof(real),
                            cudaMemcpyDeviceToHost));
    int correct_packing = verify(h_recv, prev, num_messages, msg_size, stride);
    MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &correct_packing, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));

    // ================================================================
    // Results
    // ================================================================
    if (rank == 0) {
        double time_blocking = t1 - t0;
        double time_packing  = t3 - t2;
        printf("Packing kernels: %d messages x %d elements (stride %d), "
               "%d iterations, %d GPUs\n",
               num_messages, msg_size, stride, num_iters, size);
        printf("Blocking:  %8.4f s  (correct: %s)\n", time_blocking,
               correct_blocking ? "yes" : "no");
        printf("Packing:   %8.4f s  (correct: %s)\n", time_packing,
               correct_packing ? "yes" : "no");
        if (correct_packing && time_packing > 0)
            printf("Speedup:   %8.2f\n", time_blocking / time_packing);
    }

    CUDA_RT_CALL(cudaFreeHost(h_recv));
    CUDA_RT_CALL(cudaFree(d_recv_buf));
    CUDA_RT_CALL(cudaFree(d_send_buf));
    CUDA_RT_CALL(cudaFree(d_recv));
    CUDA_RT_CALL(cudaFree(d_send));

    MPI_CALL(MPI_Finalize());
    return 0;
}
