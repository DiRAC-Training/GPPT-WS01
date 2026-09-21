<!--
Original work Copyright (c) 2021 FZJ-JSC
Modified work Copyright (c) 2026 Miren Radia
-->

# Exercise 03: Overlap Communication and Computation with MPI

## Task: Hide the halo exchange behind the bulk computation using CUDA streams

### Description
The purpose of this task is to overlap the MPI halo exchange with the Jacobi
computation using multiple CUDA streams. The starting point of this task is a
skeleton `jacobi.cpp` in which the multi-GPU parallelization is working: the
mesh is decomposed such that each rank computes either `(ny - 2) / size` or
`(ny - 2) / size + 1` rows and the halo exchange between neighbouring ranks is
implemented with GPU-aware non-blocking MPI calls (`MPI_Irecv`, `MPI_Isend`,
completed by one `MPI_Waitall` per iteration). However, the code is not
efficient yet: the whole Jacobi iteration (a single kernel covering all local
rows) has to finish before any communication is started, because the host
synchronizes on the whole compute stream before posting the exchanges. Take
some time to get familiar with the code. There is also a single-GPU version
with which the performance and numerical results are compared. As this
skeleton already produces correct results, success in this exercise will need to
be measured by profiling the code, not by the verification. Bear in mind that on
4 GPUs, the performance difference will be negligible and will only become more
apparent in the scaling as the number of GPUs increases.  Once you are familiar
with the code, please work on the `TODOs` in `jacobi.cpp`:

1. Query the CUDA stream priority range (least/greatest) with
   `cudaDeviceGetStreamPriorityRange` and declare the additional top and
   bottom CUDA streams (`push_top_stream`, `push_bottom_stream`) and their
   corresponding events (`push_top_done`, `push_bottom_done`).
1. Create the additional streams with the greatest available priority and
   change the creation of the compute stream to use the least available
   priority, using `cudaStreamCreateWithPriority`.
1. Split the launch of the Jacobi kernel into three launches:
   * The bulk of the local domain (rows `iy_start+1` to `iy_end-1`) on the
     compute stream.
   * The top boundary row (`iy_start`) on `push_top_stream` and the bottom
     boundary row (`iy_end-1`) on `push_bottom_stream`.
   * Make the boundary streams wait on the `reset_l2norm_done` event with
     `cudaStreamWaitEvent` before launching, as all three kernels atomically
     accumulate into `l2_norm_d`, which needs to be zeroed first.
   * Record the `push_top_done` / `push_bottom_done` events after each
     boundary launch.
1. Make the compute stream wait on both boundary events before copying the
   norm to the host, as the boundary kernels also accumulate into `l2_norm_d`.
1. Replace the `cudaEventSynchronize(compute_done)` before the top exchange
   (receive bottom halo, send top row) with a synchronisation on the top
   stream only, so the exchange can start once the one-row top boundary
   kernel is done while the bulk kernel still runs.
1. Add a synchronisation on the bottom stream before the bottom exchange
   (receive top halo, send bottom row).
1. Destroy the additional streams and events before the program ends.

Do not change the non-blocking MPI calls themselves: they are already posted
correctly for an overlap — the overlap is achieved purely by changing what the
host (and the GPU streams) synchronise on. While the host waits in
`MPI_Waitall`, the bulk kernel keeps running on the GPU.

### Build and run commands

* First make sure you have loaded appropriate modules. For the Nvidia A100s in
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
* You can run a very quick test (10 iterations) interactively (assuming you are
  on a node with Nvidia GPUs) to check it runs without crashing:
  ```bash
  make test
  ```
* To run a more reasonable number of iterations (1000) on the compute nodes:
  ```bash
  make run
  ```
  For `make run` (and the other submission targets) the environment variable
  `NP` can be set to change the number of MPI processes.
* The effect of the overlap is best seen in a profile. Generate a report to
  view in Nsight Systems with
  ```bash
  make profile
  ```
  first for the unmodified skeleton and then for your solution, and compare
  the timelines (navigate to the `it_*` NVTX ranges): in the solution, the
  MPI communication should happen while the bulk Jacobi kernel is still
  running.
