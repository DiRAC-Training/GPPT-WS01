<!--
Copyright (c) 2026 Miren Radia
-->

# Exercise 02: Packing Kernels and Non-Blocking Communication

## Task: Replace Multiple Blocking Messages with a Packing Kernel

### Description

The starting point `packing.F90` performs a ring-shift communication between
GPUs: each rank sends `num_messages` small messages (each `msg_size` elements)
to its right neighbour and receives from its left neighbour, repeating for
`num_iters` iterations.  Because the messages are non-contiguous in GPU memory
(they are stored as the first `msg_size` rows of each column in a
`stride`-row array, with the gap representing local data that is not
communicated), each individual MPI call can only send `msg_size` elements at a
time.

The data arrays are laid out as `send(stride, num_messages)`, so a message
`m` occupies the contiguous section `send(1:msg_size, m)` — a column slice.
The packed buffer `send_buf(msg_size, num_messages)` is fully contiguous in
memory (column-major), so a single MPI call can transfer all messages at once.

The code contains **two timed approaches** so that the benefit of packing can
be measured directly:

1. **Blocking baseline** (already implemented): a loop of `num_messages`
   blocking `MPI_Sendrecv` calls per iteration, one per message.  This incurs
   `num_messages` separate MPI latencies per iteration.

2. **Packing + non-blocking** (TODOs): an OpenMP target offload kernel packs
   the `num_messages` strided messages into a single contiguous buffer, one
   `MPI_Isend` / `MPI_Irecv` pair communicates the packed buffer, `MPI_Waitall`
   completes the transfer, and another OpenMP kernel unpacks the received
   buffer back into the strided layout.

Once the TODOs are implemented, both approaches should report `correct: yes`
and the printed speedup shows the benefit of packing.

Once you are familiar with the code, please work on the `TODO`s in
`packing.F90`:

1. Implement the `pack_kernel` that gathers `num_messages` strided messages
   into a contiguous buffer.
   * Each thread copies one element: `dst(i, m) = src(i, m)`.
   * Use a `!$omp target teams distribute parallel do collapse(2)`
     construct.
1. Implement the `unpack_kernel` that scatters a contiguous buffer back into
   the strided layout.
   * Each thread copies one element: `dst(i, m) = src(i, m)`.
   * Use a `!$omp target teams distribute parallel do collapse(2)`
     construct.
1. In the packing loop (Approach 2), replace the TODO comments with:
   * Call `pack_kernel` to gather `send` into `send_buf`.
   * Post `MPI_Isend` (`send_buf` to `next`) and `MPI_Irecv` (`recv_buf`
     from `prev`) inside a `!$omp target data use_device_addr(send_buf, recv_buf)`
     region so that device pointers are passed to GPU-aware MPI.
     * The signatures are
       ```fortran
       MPI_Isend(buf, count, datatype, dest, tag, comm, request, ierror)
       MPI_Irecv(buf, count, datatype, source, tag, comm, request, ierror)
       ```
   * Call `MPI_Waitall` to complete both requests before ending the
     `use_device_addr` region.
   * Call `unpack_kernel` to scatter `recv_buf` into `recv`.
   * Use GPU-aware MPI, so the send and receive buffers are located in GPU
     memory.  Use the defined `MPI_REAL_TYPE`.

### Build and run commands

* First make sure you have loaded appropriate modules.  For the Nvidia A100s in
  CSD3 (`ampere` partition), the modules are in
  [`csd3-modules-fortran-openmp.sh`](../../csd3-modules-fortran-openmp.sh) and
  can be loaded with
  ```bash
  source ../../csd3-modules-fortran-openmp.sh
  ```
* Build with
  ```bash
  make
  ```
  This uses the `mpifort` MPI compiler wrapper, which is set up to use the
  `nvfortran` compiler (`OMPI_FC=nvfortran`).
* You can run a quick test interactively (assuming you are on a node with
  Nvidia GPUs) to check it runs without crashing:
  ```bash
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
