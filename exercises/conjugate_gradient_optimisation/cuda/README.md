# Porting exercise: porting a conjugate gradient algorithm

In this exercise you will be guided through the porting of a conjugate gradient algorithm from an existing CPU implementation to a GPU implementation. You do not need to understand the details of the algorithm but the components you will be porting include a matrix-vector multiplication, a dot product and a type of vector addition.

Try to struggle a little before revealing the hints or checking solutions. This has been shown to improve the learning experience.

## The problem

The conjugate gradient (CG) finds the solution $x$ to the matrix equation $A\vec{x} = \vec{b}$ for a given matrix $A$ and vector $b$[^pos_def]. In our code we will be randomly generating both $A$ and $x$, calculating $b$, then using our algorithm to come up with the (known) solution $x$.

[^pos_def]: For the CG algorithm to work it must be *positive-definite* and *diagonally dominant*. In this code, the matrix is generated so that it has both these properties.

The **conjugate gradient (CG)** method is found in `solver.cpp`. This is the part of the code you will port. The CG loop itself runs on the host; each iteration calls three linear algebra routines, and these are what we will move onto the GPU:

- `matvec` — dense matrix–vector product $\mathbf{y} = A \mathbf{x}$
- `axpby`  — vector update $\mathbf{y} = \alpha \mathbf{x} + \beta \mathbf{y}$
- `dot`    — inner product $\mathbf{a} \cdot \mathbf{b}$

## Project layout

```bash
Makefile
include          // You will probably not touch these
├── idx.hpp
├── solver.hpp
├── precision.hpp
└── test.hpp
src
├── main.cpp
├── solver_FIXME.cu    // START HERE
└── test.cpp
```

Also in `solution` is a number of solutions. You can build and run these with:

```bash
make cuda && ./build/cuda # Unoptimised CUDA solution. Your solution will likely look very similar!
make cuda_optimised && ./build/cuda_optimised # Optimised CUDA matvec using fully coalesced memory accesses and a warp-reduce
make cublas && ./build/cublas # Uses a cuBLAS ibrary call for every operation
make cublas_sym && ./build/cublas_sym # Same as above but using an explicitly symmetric matvec (sometimes faster)
```

## Running the baseline

Build and run the CPU version first so you have a reference for correctness and timing:

```bash
make cpu && build/cpu
```

You should see output like:

```text
mkdir -p build/solution
mkdir -p build
nvc++ -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"/opt/cuda/include" "-DBLOCK_SIZE=128" -c src/main.cpp -o build/main.o
nvc++ -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"/opt/cuda/include" "-DBLOCK_SIZE=128" -c src/test.cpp -o build/test.o
nvc++ -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"/opt/cuda/include" "-DBLOCK_SIZE=128" -c src/solver.cpp -o build/solver.o
nvc++ -fopenmp build/main.o build/test.o build/solver.o -o build/cpu

Unit tests passed!

Matrix size: 8192 x 8192
Max iterations: 32
Generating random seed
Using RNG seed: 1784293229753343619

Matrix is WELL conditioned
Generating matrix
Generating random solution
Generating right hand side

Starting solver
0: r = 6.484026e+04
1: r = 9.832198e+01
2: r = 5.190696e-01
3: r = 2.853553e-03
4: r = 1.069918e-02
5: r = 1.242727e-05
6: r = 6.537093e-08
7: r = 3.496577e-10
8: r = 1.134825e-09
Performed 9 iterations
Solve time: 148252 us
Time per iteration: 16472 us
Average error = 0.000832548
```

Take note of your **time per iteration** for this CPU version reported here in microseconds. This is ultimately what you are trying to improve in the GPU port.

## Task 0: Get used to the code

Before diving into changing the code, **get used to the key parts in `main.cpp` and `solver.cpp`**:

- `main()`: runs tests, sets up the matrix problem, runs the algorithm and checks its output. Note some command line options:
    - `--seed`: Sets the random number generator seed (or set to `-1` to generate a random seed).
    - `--well_conditioned`: If true, this will force the matrix to be poorly conditioned, i.e. it will take more iterations to converge. **More iterations usually means more accurate timing measurements so do use this when recording runtimes.**
    - `-n`: side-length of the matrix (which is square, so total size is `N * N`). **Use this to change the problem size and see how it affects relative performance.**
    - `--cg_max_iter`: *Maximum* number of iterations before the CG algorithm stops. **Use this to manage the total number of iterations when using `well_conditioned` above.**
- `#define SINGLE_PRECISION` in `include/precision.hpp`. This allows the entire code to be switched to run in single or double precision.
    - **Use this to check the code's performance in double or single precision.**
    - Note: This is why we use `real` instead of explicitly `float` or `double` for declaring floating point numbers throughout the code. This is a particularly simple but coarse-grained and potentially error-prone way of supporting different precisions. More sophisticated approaches will offer different trade-offs.

You should notice at the start of `main` that some unit tests are run to check the result of `matvec` and `dot` against known inputs and outputs. This should give you some confidence as you edit the code that you're maintaining correctness. The reported residual also indicates correctness. Generally, the residual `r` should get smaller until it reaches the stopping value (set to around `1e-12`). **Once converged, the reported average error for a matrix size of 8192 x 8192 should be around $0.001$.**

---

To get an idea of the intended performance, try profiling the GPU solution with:

```bash
make cuda && ./build/cuda
nsys profile -o build/report1 ./build/cuda
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum  build/report1.nsys-rep
```

**Do not look at the solution code yet.**

## Task 1: The porting task

You will rewrite two operations as CUDA kernels and manage memory through the CUDA runtime. You shouldn't expect any speedup until every kernel is ported. In fact, due to increased data movement, the code will likely be slower!

You should start with `solver_FIXME.cu` which contains some ported code and some code for you to complete. Your tasks are indicated with TODO comments.

In the starter file is:

- `cg_solve` with the full CG iteration, using `new[]` allocations (you convert these in Step 1).
- A `matvec_kernel` with an empty body (See Step 2) and a complete launcher that calls it.
- An `axpby_kernel` with an empty body (See Step 3) and a temporary host-fallback launcher marked for replacement in Step 3.
- A provided `dot` function using cuBLAS, including handle setup — do not modify.

At each step you will be asked to re-run the code.

### Step 1: Switch allocations to managed memory

**Convert all *appropriate* allocations to use managed memory so the returned memory can be used from either the host or device.**

The allocations you should convert are near comments marked with `// TODO EX1` in `main` and `cg_solve`. 

**Follow where these pointers are used to work out if the pointer is used in a CUDA kernel. If it is, it will need to be converted to managed memory.**

<details>
<summary>Hint</summary>

Regular pointers allocated and deallocated through `new` and `delete` are converted as:

```cpp
// Before:
float *v = new float[n];
// ...
delete[] v;

// After:
float *v;
cudaMallocManaged(&v, n * sizeof(float));
// ...
cudaFree(v);
```

</details>

**Try to identify yourself which allocations you think need converting before checking the table in the hint below**.

<details>
<summary>Hint</summary>

| File         | Allocation(s) |
|--------------|---------------|
| `main.cpp`   | `A`, `x`, `b` |
| `solver_FIXME.cu`  | `r`, `p`, `A_times_p` (inside `cg_solve`) |

`x_soln` does not need converted because it is never used by a kernel.

</details>


Currently in `solver_FIXME.cu`, both `matvec` and `axpby` will run on the CPU, but `dot` has already been converted to a cuBlas call. This means if you have implemented the correct managed allocations and the memory can be accessed from both the host and device, the code should compile and run.

**Use the following command to check if your allocations are correct:**

```bash
make fixme && ./build/fixme
```

This should compile and produce a very similar (but probably not identical) output to the purely CPU version you ran in the previous section.

Note: This will build and run completely fine if you have incorrectly freed memory or if you have allocated *more* managed memory than required.

### Step 1.1: Error handling (Optional)

You might have noticed the use of the macro `CHECK_LAST_CUDA_ERROR()`. This is used in this exercise to check the status of the last called CUDA function and exit the entire program if an error has been reported.

It's good practice to use a similar macro to this to check that each CUDA API call has succeeded. Failing to do so can result in errors being reported long after the call that actually caused the error, making debugging difficult.

As an extension exercise, **wrap each of your calls to `cudaMallocManaged` and `cudaFree` with the other macro found in `errors.hpp`, `CHECK_CUDA_ERROR(val)`**.

What happens when you request more memory than available on the GPU? What happens when you attempt to free the same pointer twice? Does the error make sense?

Try unwrapping the call and performing the same experiment. Does the code fail in the same way?

### Step 2: Port `matvec`

**Open `solver_FIXME.cu` and fill in the body of `matvec_kernel`.**

The kernel should perform the same operation as the CPU version in `solver.cpp`.

**IMPORTANT**: Currently, the `matvec` function in `solver_FIXME.cu` calls the CPU version. You will have to comment or uncomment the indicated lines to make sure the function runs your GPU kernel.

Try to come up with your own parallelisation of the operation before checking the hint below or peeking at the solution.

<details>
<summary>Hint</summary>

The natural parallelisation is one thread per row of the matrix. Each thread should compute its row's dot product against the vector `x` and write the result into `y`.

</details>

<details>
<summary>Hint</summary>

Each thread should deal with the i-th row, and perform a sum like:

```cuda
real sum = 0.0;
for (int j = 0; j < n; j++)
  sum += A[i * n + j] * x[j];
```

</details>

<details>
<summary>Hint</summary>

Feel free to peek at the solution in `solution/solver_cuda.cu`!

</details>


Once you have finished your GPU implementation of `matvec`, you have a choice of how to test it. You can either rely on the end-to-end test of the entire algorithm run through `main` or you can adapt the existing CPU unit tests to work with your new GPU code. Since we have provided solutions in `solution/test.cpp`, you can also choose to copy the provided unit test solutions and nobody can stop you.

**Test your implementation by recompiling and rerunning the program.**

### Step 3: Port `axpby`

There are two sub-tasks in `solver_FIXME.cu`:

1. Fill in the body of `axpby_kernel`.
2. Replace the CPU fallback inside the `axpby` launcher with a kernel launch, following the same pattern as `matvec`.

<details>
<summary>Hint</summary>

`axpby` is purely element-wise: one thread per element, compute `y[i] = alpha * x[i] + beta * y[i]`.

</details>

<details>
<summary>Hint</summary>

The kernel launch in `axpby` should look something like:

```cuda
int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
axpby_kernel<<<grid, BLOCK_SIZE>>>(y, x, alpha, beta, n);
```

</details>


There are no unit tests for this function so to test it, **you should compile and run the entire program again.**

With all operations now running on the GPU, you should now see a meaningful speedup over the CPU baseline.

## Extension tasks

- Change the matrix size with `-n` to understand how the performance changes with problem size. Try this with both the CPU and GPU versions.
- The block size can be changed at compile time with `make fixme BLOCK_SIZE=128`. Run with a variety of block sizes to find the optimal value.
- Profile the GPU version  (e.g. `nsys stats <exe>`) and identify which kernel dominates the runtime. Is it what you expected?
- Compare the performance of the various solutions. Which is optimal for the default problem size? Does the optimal solution change with different problem sizes?
- Compare against the cuBLAS-based solutions: `make cublas && ./build/cublas` and `make cublas_sym && ./build/cublas_sym`.
