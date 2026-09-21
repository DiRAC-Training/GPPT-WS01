# Intro to CUDA: N-Body Simulation

You are given a working CPU version of a direct-summation n-body simulation
found in a single source file, `nbody.cpp`. In this exercise, you will port this code to GPU using CUDA. There is a full solution given in `nbody.cu` if you would rather peek but we recommend attempting each exercise and using the hints as needed.

You will use the existing unit tests to confidently and carefully port the two major functions one at a time. Then, you will port the entire main loop. There are also some extension exercises.

## Step 0: Building & running unit tests

**Create your copy:**

```bash
cp nbody.cpp nbody_port.cu
```

**Update makefile, copying the recipe for `nbody_gpu`**

<details>
<summary>Solution</summary>

```makefile
nbody_port: nbody_port.cu util.hpp
	nvcc -O3 nbody_port.cu -o nbody_port
```

</details>

**Test build:**

```bash
make nbody_port && ./nbody_port
```

**Run only the unit tests:**

```bash
make nbody_port && ./nbody_port --only_unit_tests
```

## Step 1: Port `advance_pos`

We'll start by porting the unit test `test_advance_pos` and in the process we'll port `advance_pos`.

You should be building and running unit tests with:

```bash
make nbody_port && ./nbody_port --only_unit_tests
```

---

**Copy `advance_pos` to a new function `advance_pos_k`.**

Since we're using the unit test to port this, we need to keep the original `advance_pos` so other parts of the code still compile.

**Update the function signature of `advance_pos_k` to make it a kernel.**

Add the keyword that makes this function a kernel. Update the arguments to take in pointers to `Vec2` (e.g. `Vec2*`). You will also need to add a new argument with the list length `N`.

<details>
<summary>Hint</summary>

The keyword to make a function a kernel is `__global__`.

</details>

<details>
<summary>Solution</summary>

Your function signature should look something like:

```cpp
__global__ void advance_pos_k(Vec2 *pos_prev, const Vec2 *pos, const Vec2 *acc, uint N, real dt) {
```

Notice we've kept `const` where possible and used an unsigned int for `N`.

</details>

---

**Change the loop body to standard SIMT kernel style.**

<details>
<summary>Hint</summary>

You will need to calculate the thread index:

```cpp
const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
```

</details>

<details>
<summary>Hint</summary>

You should make sure any data accesses are within bounds:

```cpp
if (gtid >= N) return;
```

</details>

<details>
<summary>Hint</summary>

Data access is now via `gtid` instead of the original loop index `i`:

```cpp
pos_prev[gtid].x =
  2.0 * pos[gtid].x - pos_prev[gtid].x + acc[gtid].x * dt * dt;
```

</details>

<details>
<summary>Solution</summary>

See `advance_pos_k` in `nbody.cu`.

```cpp
__global__ void advance_pos_k(Vec2 *pos_prev, const Vec2 *pos, const Vec2 *acc,
                            uint N, real dt) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= N)
    return;
  pos_prev[gtid].x =
      2.0 * pos[gtid].x - pos_prev[gtid].x + acc[gtid].x * dt * dt;
  pos_prev[gtid].y =
      2.0 * pos[gtid].y - pos_prev[gtid].y + acc[gtid].y * dt * dt;
}
```

</details>

---

**In `test_advance_pos`, change all `vector<Vec2>` to `thrust::device_vector<Vec2>`**

Woops! We forgot the includes.

**Put these with the other includes:**

```cpp
#include <thrust/copy.h>
#include <thrust/device_vector.h>
```

**Update the call to `advance_pos` in `test_advance_pos` to use the new kernel.**

You will need to access the pointer to GPU memory held within `thrust::device_vector`. Use `pos_prev.data().get()`.

Because we're only processing a few particles, we can manually set the launch parameters to something small like 1 block and 32 threads per block.

<details>
<summary>Hint</summary>

Add kernel launch parameters to `advance_pos`:

```cpp
advance_pos_k<<<1, 32>>>(...)
```

</details>

<details>
<summary>Hint</summary>

Update arguments, converting the `device_vector`s to their internal pointers:

```cpp
advance_pos_k<<<1, 32>>>(pos_prev.data().get(), pos.data().get(), acc.data().get(), pos.size(), dt);
```

</details>

<details>
<summary>Hint</summary>

Remember to synchronise after!

```cpp
advance_pos_k<<<1, 32>>>(pos_prev.data().get(), pos.data().get(), acc.data().get(), pos.size(), dt);
cudaDeviceSynchronize();
```

</details>

In this test, it's actually not necessary to explicitly synchronise because the later access of `pos[0]` will automatically invoke a data transfer. This transfer will wait for the kernel to finish.

## Step 2: Port `calc_acc`

Now we'll do the same for `calc_acc`. This step is similar enough to the previous one that you may wish to skip some tasks and steal sections from the solution `nbody.cu`. If you are still relatively new to writing kernels, we recommend not skipping anything.

You should still be building and running unit tests with:

```bash
make nbody_port && ./nbody_port --only_unit_tests
```

Let's just focus on one unit test: `test_calc_acc_x`.

**Comment out the body of `test_calc_acc_y`.**

See the comments marked TODO 2. This will allow us to focus on porting without updating two very similar unit tests.

---

**Once again, copy `calc_acc` to a new function called `calc_acc_k`.**

**Update the function signature of `calc_acc` to make it a kernel.**

<details>
<summary>Solution</summary>

```cpp
__global__ void calc_acc_k(Vec2 *acc, const Vec2 *pos, const real *mass, uint N,
                         real eps = 0.0) {
```

</details>

**Update `test_calc_acc_x`:**

- [ ] change all `vector`s to `thrust::device_vector`
- [ ] update the call to `calc_acc`
- [ ] add a synchronisation

This is very similar to the previous step so we won't give you any more hints!

---

**Transform the loop body of `calc_acc` into a SIMT-style kernel.**

Assume each thread is assigned one particle and calculates its acceleration due to all other particles. I.e. each thread runs a loop over all other particles and sums the total acceleration.

<details>
<summary>Hint</summary>

Once again, we need to calculate the thread index and make sure we're in bounds:

```cpp
const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
if (gtid >= N)
  return;
```

</details>

<details>
<summary>Hint</summary>

Each thread calculates the total acceleration for one particle. So let's remove the outer loop in `i` and, instead, load `pos[gtid]` and save to `acc[gtid]`:

```cpp
const Vec2 pi = pos[gtid];
...
acc[gtid] = accl;
```

</details>

<details>
<summary>Solution</summary>

See `calc_acc_k` in `nbody.cu`.

</details>

---

Building and running this should result in a compiler error:

```
nbody_port.cu(71): error: calling a __host__ function("calc_acc_pair(    ::Vec2,     ::Vec2, float, float)") from a __global__ function("calc_acc_k") is not allowed
```
This is reminding us that we haven't yet updated some of the called functions to `__device__` functions.

**Mark `norm2`, `sub`, and `calc_acc_pair` as device functions.**

<details>
<summary>Solution</summary>

```cpp
__device__ real norm2(const Vec2 &v) { ... }
__device__ Vec2 sub(const Vec2 &v1, const Vec2 &v2) { ... }
__device__ Vec2 calc_acc_pair(Vec2 pi, Vec2 pj, real mj, real eps = 0.0) { ... }
```

</details>

---

Compiling at this stage reveals an error:

```
nbody_port.cu(52): error: calling a __device__ function("...") from a __host__ function("calc_acc") is not allowed
```

Since we're also compiling the regular, non-kernel `calc_acc`, we need to *also* add `__host__` to these functions:

```cpp
__device__ __host__ real norm2(const Vec2 &v) { ... }
__device__ __host__ Vec2 sub(const Vec2 &v1, const Vec2 &v2) { ... }
__device__ __host__ Vec2 calc_acc_pair(Vec2 pi, Vec2 pj, real mj, real eps = 0.0) { ... }
```

This is a portable way to maintain functions that can be used both from host and device code.

---

**Re-enable `test_calc_acc_y`.**

You can choose to manually perform the same steps as for `test_calc_acc_x` or you can simply copy the solution from `nbody.cu`.

## Step 3: Review and reflect

Take a moment to consider the strategy we've used to port these individual functions into kernels:

1. Identify a function to be ported
2. Identify a unit test that can help ensure correctness as we port
3. Add data transfers to the unit test to ensure the appropriate data is on the GPU
4. Update any calls and synchronisations that need to happen in the test
5. Turn the function into a SIMT kernel
6. Test and iterate until correct

This is an approach centred around unit tests and porting well-structured computational steps one at a time.

Consider the following questions:

**What other tests could be used instead of unit tests?**

**What are the limitations of this approach?**

**What are some alternative approaches you've considered or previously encountered?**

## Step 4: Port the main loop

**Add device copies of all main variables.**

See TODO 4a in the code. We recommend calling the variables the same name with an added `_d` for "device". For example:

```cpp
thrust::device_vector<Vec2> pos_d(N_PARTICLES);
```

**Copy the host variables onto the device *after* the initial conditions have been calculated.**

See TODO 4a. You should use `thrust::copy` which uses a typical C++ style to copy between iterators:

```cpp
thrust::copy(pos.begin(), pos.end(), pos_d.begin());
```

<details>
<summary>Solution</summary>

You should end up with a block of declarations and copies like:

```cpp
thrust::device_vector<Vec2> pos_d(N_PARTICLES);
thrust::device_vector<Vec2> acc_d(N_PARTICLES);
thrust::device_vector<Vec2> pos_prev_d(N_PARTICLES);
thrust::device_vector<real> mass_d(N_PARTICLES);

thrust::copy(pos.begin(), pos.end(), pos_d.begin());
thrust::copy(mass.begin(), mass.end(), mass_d.begin());
thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());
thrust::copy(acc.begin(), acc.end(), acc_d.begin());
```

</details>

---

**Update the calls to `calc_acc` and `advance_pos` to their respective kernels in the main loop.**

You will need to specify the block size and number of blocks for the two kernels. Since they are each processing one particle per thread, you just need to ensure there are more total threads than particles.

<details>
<summary>Hint</summary>

Let's assume an initial block size of 32. We can calculate the number of blocks required to give at least `N_PARTICLES` total threads as:

```cpp
const int block_size = 32;
const int n_blocks = (N_PARTICLES + block_size - 1) / block_size;
```

Put this somewhere before the main loop.

</details>

<details>
<summary>Solution</summary>

Your kernel calls should look like:

```cpp
calc_acc_k<<<n_blocks, block_size>>>(acc_d.data().get(), pos_d.data().get(),
                                   mass_d.data().get(), N_PARTICLES,
                                   epsilon);
advance_pos_k<<<n_blocks, block_size>>>(pos_prev_d.data().get(),
                                      pos_d.data().get(),
                                      acc_d.data().get(), N_PARTICLES, dt);
```

</details>

**Update the pointer swap to work on the device position:**

```cpp
pos_d.swap(pos_prev_d);
```

**(Optional) Add a `cudaDeviceSynchronize` before the `timer.lap` line for more accurate timing**

**Add copies back to the host before each dump.**

Find all occurrences of `dump_to_file` and ensure  the position variable is copied to the host before it is called:

```cpp
thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
```

Note the order of variables.

## Step 5: Testing the entire simulation

You should be able to compile and run the full simulation with:

```bash
make nbody_port && ./nbody_port
```

**Compare final outputs between CPU and GPU versions:**

```bash
make nbody_cpu && ./nbody_cpu && mv final.csv og.csv && make nbody_port && ./nbody_port && diff final.csv og.csv
```

If the final diff prints nothing, that's an anticlimactic success! Well done, you've ported an entire simulation to GPU.

## Step 6: Timing and scaling

With a working version of the GPU code, you should test its performance against the CPU version. Try scaling the number of particles:

```bash
for n in 10 100 1000 10000; do ./nbody_port -n $n --quiet; done
```

Then compare against results from the GPU solution and CPU version.

## Extension exercises

**Better timing with events**

We're timing the main loop from the host which requires synchronisation for a decent measurement. Instead, look into using CUDA events for timing kernels. You should be able to emit an event at the start and end of the main loop and extract timing information from it.

---

**Removing `calc_acc`**

The way we've chosen to set up the initial conditions requires maintaining the CPU version of `calc_acc`. There are real drawbacks to maintaining two copies of the same operation:

- Both should be independently unit tested.
- Both should be updated at the same time, otherwise tricky bugs will appear.

**Try refactoring the initial condition setup to use the GPU version of `calc_acc_k`.**

You will need to rearrange the thrust copies to ensure the required data is on the device at the right point. You will also need to copy the data back to the host to calculate the initial value of `pos_prev`. Alternatively you could port this operation, however it's only called once in the entire code. In a real project, you would have to profile the setup stage to understand whether it's worth porting.

You can find a solution to this in `nbody.cu`.
