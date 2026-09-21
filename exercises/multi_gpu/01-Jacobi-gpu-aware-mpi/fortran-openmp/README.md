<!--
Original work Copyright (c) 2021 FZJ-JSC
Modified work Copyright (c) 2026 Miren Radia
-->

# Exercise 01: Multi-GPU Parallelization with GPU-aware MPI

## Task: Parallelize Jacobi Solver for Multiple GPUs using GPU-aware MPI

### Description
The purpose of this task is to use GPU-aware MPI to parallelize a Jacobi solver.
The starting point of this task is a skeleton `jacobi.F90`, in which the OpenMP
target offload kernels are already defined and also some basic setup routines
are present. There is also a single-GPU version with which the performance and
numerical results are compared. Take some time to get familiar with the code.
The mesh arrays are laid out as `a(nx, ny)`, so a row `a(:, iy)` — the data
sent to/received from a neighbour — is contiguous in memory.
Once you are familiar with the code, please work on the `TODOs` in `jacobi.F90`:

1. Get the available GPU devices and use it and the local rank to set the active
   GPU for each process. With OpenMP this is done with the
   `omp_get_num_devices()` and `omp_set_default_device()` routines from the
   `omp_lib` module.
1. Compute the top and bottom neighbours. We are using periodic boundaries on
   top and bottom, so rank 0's top neighbour is (size-1) and rank (size-1)'s
   bottom neighbour is rank 0.
1. Use `MPI_Sendrecv` to exchange data between the neighbors
   * The signature of `MPI_Sendrecv` is
     ```fortran
     MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount,
                  recvtype, source, recvtag, comm, status, ierror)
         TYPE(*), DIMENSION(..), INTENT(IN) :: sendbuf
         INTEGER, INTENT(IN) :: sendcount, dest, sendtag, recvcount, source, recvtag
         TYPE(MPI_Datatype), INTENT(IN) :: sendtype, recvtype
         TYPE(*), DIMENSION(..) :: recvbuf
         TYPE(MPI_Comm), INTENT(IN) :: comm
         TYPE(MPI_Status) :: status
         INTEGER, OPTIONAL, INTENT(OUT) :: ierror
     ```
   * Use GPU-aware MPI, so the send and receive buffers are located in
     GPU memory. With OpenMP target offload, wrap the MPI calls in an
     `!$omp target data use_device_addr(a_new)` region so that device
     pointers are passed to MPI.
   * The first newly calculated row (`iy_start`) is sent to the top neighbour
     and the bottom boundary row (`iy_end`) is received from the bottom
     process.
   * The last calculated row (`iy_end-1`) is sent to the bottom process and the
     top boundary (`1`) is received from the top.
   * The `!$omp target teams distribute parallel do` constructs in
     `jacobi_kernel` are synchronous, so the computation on the GPU has
     completed before the data transfer can start.
   * Use the defined `MPI_REAL_TYPE`. This allows an easy switch between single
     and double precision (compile with `make USE_DOUBLE=1`).

### Build and run commands

* First make sure you have loaded appropriate modules. For the Nvidia A100s in
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
* You can run a very quick test (10 iterations) interactively (assuming you are
  on a node with Nvidia GPUs) to check it runs without crashing:
  ```bash
  make test
  ```
* To run a more reasonable number of iterations (1000) on the compute nodes:
  ```bash
  make run
  ```

## Advanced Task: Optimize Load Balancing

### Description
* The work distribution of the first task is not ideal, because it can lead to
the process with the last rank having to calculate significantly more than all
the others. Therefore, the load distribution is to be optimized in this task.
* Compute the `chunk_size` such that each rank gets either `(ny - 2) / size` or
  `(ny - 2) / size + 1` rows.
* Compute how many processes get `(ny - 2) / size` and how many get
  `(ny - 2) / size + 1` rows.
* Adapt the computation of `iy_start_global`.
