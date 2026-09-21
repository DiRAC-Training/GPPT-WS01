# Recap exercise: porting a conjugate gradient algorithm to OpenMP target offload

In this exercise you will take a CPU conjugate gradient (CG) solver, already parallelised with plain OpenMP (`!$omp parallel do`), and port it to run on a GPU using OpenMP's **target offload** directives. You do not need to understand the details of the algorithm, but the components you will be porting include a matrix-vector multiplication, a dot product and a vector update.

Try to struggle a little before revealing the hints or checking the solution. This has been shown to improve the learning experience.

The CG loop itself lives in `solver.f90`. Each iteration calls three linear algebra routines, and these -- plus a couple of small loops in the driving routine itself -- are what you will move onto the GPU:

- `matvec` -- dense matrix-vector product $\mathbf{y} = A\mathbf{x}$
- `dot` -- inner product $\mathbf{a} \cdot \mathbf{b}$
- `axpby` -- vector update $\mathbf{y} = \alpha\mathbf{x} + \beta\mathbf{y}$

## Project layout

```bash
Makefile
precision.f90    // Sets the working floating point precision
main.f90         // Sets up the problem, runs the solver, checks the answer
test.f90         // Unit tests for matvec and dot, run automatically at startup
solver.f90       // The CPU OpenMP baseline
solver_FIXME.f90 // START HERE
solver_gpu.f90   // The finished GPU port -- the answer key for this exercise
```

You can build and run the solution in `solution_gpu.f90` at any point to see where you're headed:

```bash
make gpu && ./build/gpu
```

## Running the baseline

**Build and run the CPU version first so you have a reference for correctness and timing:**

```bash
make cpu && ./build/cpu
```

You should see output along these lines:

```text
gfortran -O3 -march=native precision.f90 solver.f90 test.f90 main.f90 -Jbuild -o build/cpu
 Generating matrix
 Generating random solution
 Generating right hand side
0: r =    0.190212E+04
1: r =    0.996115E+02
...
 Performed            9  iterations
 Solve time:                706974  us
 Time per iteration:                 78552  us
 Average error =    9.58454446E-04
```

Take note of your **time per iteration**, reported in microseconds. This is what you are trying to beat with the GPU port.

## Task 0: Get used to the code

Before changing anything, get used to the key parts of `main.f90` and `solver.f90`:

- `n` in `main.f90` sets the problem size (an `n`-by-`n` matrix). **Use this to see how performance scales with problem size once you have a GPU version working.**
- `DIAG_SCALE` controls how well-conditioned the matrix is. Left at its default (`-1.0`), the matrix is well-conditioned and the solve converges in a handful of iterations. A commented-out block near the top of `main.f90` switches to a larger, more ill-conditioned problem (`n = 32000`, more iterations) -- **more iterations generally means more reliable timing measurements.**
- `precision_mod` (in `precision.f90`) defines the working kind `wp` as `real32`. **Flip this to `real64` to compare single- and double-precision performance** once your port is working.
- At the start of `main`, unit tests for `matvec` and `dot` run automatically (from `test.f90`) and the program stops if any fail. This is your first correctness signal as you edit `solver.f90` -- if you break `matvec` or `dot`, you'll know immediately, before the solver even starts.
- The reported residual `r` (printed every iteration) should shrink steadily towards the convergence threshold. The final `Average error`, for the default 8192x8192 problem, should be order $10^{-3}$ or smaller.

## Task 1: The porting task

You'll be turning each `!$omp parallel do` in `solver.f90` into an equivalent `!$omp target teams distribute parallel do`, plus adding one data-management region to avoid needless copying between host and device. You shouldn't expect a speedup until every step is done -- some intermediate stages may even be *slower* than the CPU baseline, because of extra data movement between host and device.

Start from `solver_FIXME.f90`, which is a copy of the CPU version `solver.f90` with your tasks marked by `TODO` comments. At each step you will rebuild and re-run the code, then profile it with `nsys` to see the effect of your change.

### Step 1: Port `matvec`

**Open `solver_FIXME.f90` and look at the two `TODO (Step 1)` comments.**

`matvec`'s loop calls the `idx` function to compute a flat array index. Any function called from inside an offloaded loop needs to be compiled for the device as well as the host.

<details>
<summary>Hint</summary>

A function or subroutine is made available on the device with a `!$omp declare target` directive placed right after its signature:

```fortran
pure function idx(i, j, n) result(k)
  !$omp declare target
  ...
```

</details>

**Try to work out the matching change to `matvec`'s loop directive yourself before checking the hint below.**

<details>
<summary>Hint</summary>

Replace `!$omp parallel do private(j, s)` with the target-offload equivalent, keeping the same clause:

```fortran
!$omp target teams distribute parallel do private(j, s)
```

</details>

<details>
<summary>Hint</summary>

Feel free to peek at `solver_gpu.f90` if you want to check your work directly against the finished solution.

</details>

**Rebuild and run:**

```bash
make fixme && ./build/fixme
```

This should still pass the unit tests and produce a correct (if not yet fast) answer.

**Now profile it:**

```bash
nsys profile -o build/report1 ./build/fixme
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum  build/report1.nsys-rep
```

Look at the "CUDA GPU Kernel Summary" table -- you should see exactly one kernel (`matvec`) running on the device, and a "CUDA Memory Operation Summary" showing host-device copies happening around it.

### Step 2: Port `dot`

**Find the `TODO (Step 2)` comment in `dot` and replace its directive, the same way as Step 1.**

<details>
<summary>Hint</summary>

`dot` uses a `reduction` clause. This clause carries over unchanged onto the target directive -- OpenMP handles reducing a value across the whole device the same way it does across host threads.

</details>

<details>
<summary>Hint</summary>

```fortran
!$omp target teams distribute parallel do reduction(+:sum_val)
```

</details>

**Rebuild, run, and profile again:**

```bash
make fixme && ./build/fixme
nsys profile -o build/report2 ./build/fixme
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum  build/report2.nsys-rep
```

You should now see two kernels in the summary.

### Step 3: Port `axpby`

**Find the `TODO (Step 3)` comment and offload this loop the same way.**

<details>
<summary>Hint</summary>

`axpby` has no extra clauses to preserve -- it's a straight swap:

```fortran
!$omp target teams distribute parallel do
```

</details>

**Rebuild, run, and profile:**

```bash
make fixme && ./build/fixme
nsys profile -o build/report3 ./build/fixme
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum  build/report3.nsys-rep
```

All three routines now have a kernel in the summary. Look at the "CUDA Memory Operation Summary" -- you should already be seeing a lot of repeated host-device copying for the solver's own temporary arrays (`r`, `p`, `A_times_p`), since they aren't resident on the device yet and get remapped on every single call.

### Step 4: Port the remaining loops in `cg_solve`

There are two more loops directly inside `cg_solve` itself, marked with `TODO (Step 4)`:

1. `r(i) = b(i) - r(i)`, computing the initial residual.
2. `p = r`, copying the residual into the search direction.

The first is a simple loop-directive swap, same as the previous steps. The second is different: `p = r` is a whole-array Fortran assignment, not a loop, so it currently runs on the host -- which would force `r` to be copied back from the device just to perform this copy.

<details>
<summary>Hint</summary>

Rewrite the array assignment as an explicit loop, then offload it like the others:

```fortran
!$omp target teams distribute parallel do
do i = 1, n
  p(i) = r(i)
end do
```

</details>

**Rebuild, run, and profile:**

```bash
make fixme && ./build/fixme
nsys profile -o build/report4 ./build/fixme
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum  build/report4.nsys-rep
```

Every loop in the solver now runs on the GPU. Compare the `Time per iteration` you're seeing against your CPU baseline from earlier -- **you may well find it's no faster, or even slower**. Look again at the "CUDA Memory Operation Summary": with no data region around the solve, `r`, `p` and `A_times_p` are still being copied back and forth between host and device on every single kernel launch, every iteration. That transfer overhead is now the bottleneck, not the compute.

### Step 5: Keep the solver's temporaries resident on the device

**Find the two `TODO (Step 5)` comments in `cg_solve`** -- one near the top (after the `allocate` calls) and one near the bottom (after the main iteration loop).

`r`, `p` and `A_times_p` are read and written on every single iteration but never touched by the host in between. Right now, without an enclosing data region, OpenMP has no way to know that -- so it copies them to and from the device at every kernel launch. A `target data` region tells it to allocate them on the device once and leave them there for the whole solve.

<details>
<summary>Hint</summary>

Open the region with `map(alloc: ...)` rather than `map(to:...)` or `map(tofrom:...)`, since these arrays are computed fresh on the device and the host never needs to see or supply their contents directly:

```fortran
!$omp target data map(alloc: r(1:n), p(1:n), A_times_p(1:n))
```

Remember to close it with `!$omp end target data` before the `deallocate` calls.

</details>

<details>
<summary>Hint</summary>

Remember `A`, `b` and `x` are already mapped by `main.f90` -- this region only needs to cover the solver's own temporaries.

</details>

**Rebuild, run, and profile one more time:**

```bash
make fixme && ./build/fixme
nsys profile -o build/report5 ./build/fixme
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum  build/report5.nsys-rep
```

The memory-operation time should collapse compared to Step 4 -- the run is now dominated by actual compute kernels rather than copies. Compare your final `Time per iteration` against the CPU baseline you recorded earlier: you should now see a real speedup.

At this point, your `solver_FIXME.f90` should behave the same as `solver_gpu.f90`. Diff the two files to check.

### Step 6: Decide whether to map the `A`, `b` and `x`

**Find the `target data` region in `main.f90`.**

Do you have to do anything further to transfer `A`, `b` and `x`?


<details>
<summary>Solution</summary>

The `target data` region surrounding the call to `cg_solve` means `A`, `b` and `x` are already handled for you at the top level -- you won't need to map them yourself anywhere inside `solver.f90`.

</details>

## Extension tasks

- Change the matrix size (`n` in `main.f90`) and see how the CPU-vs-GPU speedup changes with problem size.
- Flip `precision_mod`'s `wp` from `real32` to `real64` in `precision.f90` and compare performance and accuracy.
- Use `nsys stats` on your finished version to see which kernel dominates the runtime. Is it what you expected?
- Compare against the cuBLAS-based solutions: `make cublas && ./build/cublas` and `make cublas_sym && ./build/cublas_sym`. How do they compare to your hand-written OpenMP-target version?
