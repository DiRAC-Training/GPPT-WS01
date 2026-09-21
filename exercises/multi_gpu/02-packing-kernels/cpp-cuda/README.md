<!--
Copyright (c) 2026 Miren Radia
-->

# Exercise 02: Packing Kernels and Non-Blocking Communication

## Task: Replace Multiple Blocking Messages with a Packing Kernel

### Description

The starting point `packing.cpp` performs a ring-shift communication between
GPUs: each rank sends `num_messages` small messages (each `msg_size` elements)
to its right neighbour and receives from its left neighbour, repeating for
`num_iters` iterations.  Because the messages are non-contiguous in GPU memory
(they are spaced `stride` elements apart, with the gap representing local data
that is not communicated), each individual MPI call can only send `msg_size`
elements at a time.

The code contains **two timed approaches** so that the benefit of packing can
be measured directly:

1. **Blocking baseline** (already implemented): a loop of `num_messages`
   blocking `MPI_Sendrecv` calls per iteration, one per message.  This incurs
   `num_messages` separate MPI latencies per iteration.

2. **Packing + non-blocking** (TODOs): a CUDA kernel packs the `num_messages`
   strided messages into a single contiguous buffer, one `MPI_Isend` /
   `MPI_Irecv` pair communicates the packed buffer, `MPI_Waitall` completes
   the transfer, and another CUDA kernel unpacks the received buffer back into
   the strided layout.

Once the TODOs are implemented, both approaches should report `correct: yes`
and the printed speedup shows the benefit of packing.

Once you are familiar with the code, please work on the `TODO`s in
`packing.cpp`:

1. Implement the `pack_kernel` that gathers `num_messages` strided messages
   into a contiguous buffer.
   * Each thread copies one element:
     `dst[m * msg_size + i] = src[m * stride + i]`.
   * The total number of threads needed is `num_messages * msg_size`.
1. Implement the `unpack_kernel` that scatters a contiguous buffer back into
   the strided layout.
   * Each thread copies one element:
     `dst[m * stride + i] = src[m * msg_size + i]`.
1. In the packing loop (Approach 2), replace the TODO comments with:
   * Launch `pack_kernel` to gather `d_send` into `d_send_buf`.
   * Synchronize the GPU (`cudaDeviceSynchronize`) so the packing is complete
     before the MPI call.
   * Post `MPI_Isend` (`d_send_buf` to `next`) and `MPI_Irecv` (`d_recv_buf`
     from `prev`) into an array of two `MPI_Request` objects.
     * The signatures are
       ```cpp
       int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
                     int dest, int tag, MPI_Comm comm, MPI_Request *request);
       int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
                     int source, int tag, MPI_Comm comm, MPI_Request *request);
       ```
   * Wait for both requests with `MPI_Waitall`.
   * Launch `unpack_kernel` to scatter `d_recv_buf` into `d_recv`.
   * Use GPU-aware MPI, so the send and receive buffers are located in GPU
     memory.  Use the defined `MPI_REAL_TYPE`.

### Build and run commands

* First make sure you have loaded appropriate modules.  For the Nvidia A100s in
  CSD3 (`ampere` partition), the modules are in
  [`csd3-modules-cpp-cuda.sh`](../../csd3-modules-cpp-cuda.sh) and can be loaded
  with
  ```bash
  source ../../csd3-modules-cpp-cuda.sh
  ```
* Build with
  ```bash
  make
  ```
* You can run a quick test interactively (assuming you are on a node with
  Nvidia GPUs) to check it runs without crashing:
  ```bash.
  make test
  ```
* To run on the compute nodes:
  ```bash
  make run
  ```

## Extension task

### Description

Investigate how the speedup changes as the number of separate messages and the
size of the messages changes. You can adjust these parameters by passing the
following flags to the application:
* `-num_messages <number of messages>` (default: 128)
* `-msg_size <size>` (default: 1024)