# CUDA N-Body: Optimisation Exercise

You are given a working CUDA version of a direct-summation n-body simulation
found in a single source file, `nbody.cu`. It is correct but unoptimised and
includes some deliberate performance issues that you will have to find and fix.

Your job is to find bottlenecks using a profiler and implement related optimisations. You will
make all of your changes in `nbody.cu`. You can compare to the solution in `nbody_tiled.cu` but we recommend you attempt tasks before checking the solution. Similarly, we recommend you struggle a little with tasks before using the hints.

There are two main tasks. Task 1 is about measurement, tuning and data transfer optimisation. Task 2 involves implementing a tiling optimisation for better data reuse within a kernel.

## What you'll learn

By the end of this exercise you should be able to:

- Identify data transfers and large kernels with Nsight Systems
- Use Nsight Compute to explore how resource limits affect kernel performance
- Tune a kernel's block size using profiling data
- Understand a shared-memory tiling optimisation

## Before you start

### Build and run

```sh
make && ./nbody
```

The unit tests run automatically at startup and the program aborts if any fail.
Leave them on — they are a cheap first check on every change. However, keep in
mind that they only run on two particles, so they will not be able to catch
everything you can break in Task 2.

To run only the unit tests:

```sh
make && ./nbody --only_unit_tests
```

In order to check correctness, you will be comparing any new output to the original output `final.csv`.

**Rename this to save it:**

```bash
make && ./nbody && mv final.csv final_og.csv
```

## **Task 1a** - Identify a bottleneck with `nsys`

**Try to identify potential bottlenecks in the main loop in `nbody.cu` before running the profiler.** Just from reading the code, can you get a sense of which kernels may dominate? Can you identify data transfers and synchronisation points that may need investigating?

**Profile the code with `nsys`**:

```sh
nsys profile -o report ./nbody
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum report.nsys-rep
```

**Inspect the output to help answer the questions:**

1. Which kernel takes the most time? Take note of this for later tasks.
2. Where in the main loop is there an obvious bottleneck?

The bottleneck is most clearly seen in the timeline view of the Nsight Systems UI. See the guidance document for more information on using this UI with CSD3. You can still identify it with just the text output from `nsys stats`.

<details>
<summary>Hint</summary>

Look closer at the memory transfers. In the timeline view you should see a large block of async memory transfer happening. In `nsys stats` note how long the memory transfers take compared to the kernel runtime.

</details>

<details>
<summary>Hint</summary>

Read the code in the main loop. You should be able to identify lines that are transferring data.

</details>

<details>
<summary>Hint</summary>

The variables being transferred are `pos` and `pos_prev` (and their GPU partners `pos_d` and `pos_prev_d`. Within the main loop, where are these variables being used? Why are they being transferred?

</details>

<details>
<summary>Solution</summary>

In the main loop, there are the following data transfers:

```cpp
thrust::copy(pos.begin(), pos.end(), pos_d.begin());
thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());

...

thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
thrust::copy(pos_prev_d.begin(), pos_prev_d.end(), pos_prev.begin());
```

These are the only transfers occurring within the main loop and are the main bottleneck visible with Nsight Systems.

Consider: **Why might these lines be accidentally left in during porting?**

</details>

## Task 1b - Fix the bottleneck

**Remove or move appropriate data transfers to speed up the code without introducing a bug.**

Remember to build and profile regularly. You can use `nsys stats` to quickly check relevant metrics.

**Check correctness:**

Since we are only changing data transfers in this task, the output should be *identical*, so we can check overall correctness with:

```bash
diff final.csv final_og.csv
```

If the diff shows any different, you may find it useful to look at the plot of the data:

```sh
../visualisation/plot.py final.csv
```

<details>
<summary>Hint</summary>

`pos` must be transferred before being dumped in:

```cpp
if (t >= next_dump) {
  dump_to_file(format_fname(dump_counter), pos);
  dump_counter += 1;
  next_dump += t_between_dump;
}
```

But in the current version of the code this happens every single timestep. Make sure the transfer happens if and when it's actually needed.

</details>

<details>
<summary>Solution</summary>

You should have now removed all data transfers from the main loop with the exception of the device to host transfer that should now happen inside the dump check:

```cpp
if (t >= next_dump) {
  thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
  dump_to_file(format_fname(dump_counter), pos);
  dump_counter += 1;
  next_dump += t_between_dump;
}
```

Your profile should now reveal that the amount of time spent on data transfers is small compared to the actual runtime of the kernels.

</details>

## Task 1c - Tuning parameters with Nsight Compute

In this task you'll be mainly exploring how some Nsight Compute metrics change in response to block size. You'll use *Duration* to pick the most optimal block size for the hardware.

You should have already found the dominant kernel: `calc_acc` so let's focus on that.

**Look at the code for `calc_acc`. Before profiling, do you think it will be compute-bound or memory-bound? Why?**

**Now profile `calc_acc` with `ncu`.**

To measure a kernel directly with Nsight Compute, use:

```sh
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section LaunchStats --launch-skip 1 --launch-count 1 -k <kernel_name> \
    ./nbody --quiet --disable_unit_tests
```

- `--launch-skip 1 --launch-count 1` profiles a single launch of the selected
  kernel, skipping the first one. A kernel's first launch pays one-off costs such
  as module loading and cold caches, so it is not representative of the rest.
  Without this you profile every launch, which may take a very long time.
- Do **not** use `--set full`. It will take too long.
- `-k <kernel_name>` selects the kernel by name. Change it to profile individual
  kernels as needed.
- `--quiet` to limit the simulation output.
- **IMPORTANT**: `--disable_unit_tests` to ensure the actual kernel launch is profiled, not the one called in the unit tests.

The number to note is *Duration* in the Speed Of Light section. One run should be
enough to get an accurate enough measurement but feel free to check this by changing `--launch-count`.

---

The output of `ncu` can be overwhelming. Focus on the Speed Of Light section for `calc_acc`. Compare
Compute (SM) Throughput with DRAM Throughput.

**Is this kernel memory bound or compute bound?**

---

Find `block_size` in `main()`.

**From your understanding of GPU architecture so far; do you think this value is optimal? Why?**

---

**Vary `block_size` without the profiler.**

We have exposed the block size with the flag `--bs` for easy access. Try running a sweep across some block sizes *without the profiler*:

```bash
for bs in 32 64 128 256 512 1024; do
  ./nbody --quiet --bs $bs --disable_unit_tests -n 100000
done
```

Note that we have increased the number of particles so that we *saturate* the GPU, i.e. we have many more particles than active threads. This is related to the concept of *occupancy*.

What do you notice about the results?

---

**Vary `block_size` with the profiler.**

For example:

```sh
for bs in 32 64 128 256 512 1024; do
    ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section LaunchStats --launch-skip 1 --launch-count 1 -k calc_acc \
    ./nbody --quiet --bs $bs --disable_unit_tests -n 100000 >> profile_bs_$bs.txt
done
```

The outputted `profile_bs_*.txt` will contain the profile outputs.

**Compare metrics across the different runs, focusing particularly on the smallest, 32, largest, 1024, and your previously identified optimum.**

What changes in the occupancy section as the block size varies? Why?

How do you see some of the recommendations changing with block size?

## Task 1.5: Reflection

At this point you've used `nsys` to find the dominant kernel, `calc_acc`, and diagnose a performance bottleneck. You've tuned block size for overall time and explored how the block size can change various metrics in `ncu`.

Profiling is one of the key pillars of GPU programming. If you get used to using these tools often, you will quickly optimise your codes.

Either note down the answers to the following questions or discuss with someone nearby:

**How do you think you will use these tools in your own work?**

**Do you already use these tools in a different way?**

## Task 2: Tiling optimisation

Recall that shared memory is the GPU's programmable cache and, by default, data loaded from DRAM isn't reused. In `calc_acc` we can identify the potential for data reuse in the following lines:

```cpp
  for (int j = 0; j < N; ++j) {
    const Vec2 accll = calc_acc_pair(pi, pos[j], mass[j], eps);
    accl.x += accll.x;
    accl.y += accll.y;
  }
```

The idea of tiling: in `calc_acc`, every thread requests the entire `pos` and `mass`
arrays. Threads in a block request the same data over and over again. Instead,
we can have the block cooperatively load a **tile** of particle data into
shared memory, have every thread in the block consume that tile, then move to
the next one. This makes the reuse explicit.

In this task you will implement a tiled version of this kernel, `calc_acc_tiled`, using shared memory, then find the best tile size by
measurement.

After completing this you should be able to explain how tile size affects the
kernel, whether its best value matches the best block size you found in Task 1,
and how the comparison changes as the problem grows beyond one "wave per SM".

---

**Consider the overall algorithm:**

```
for each tile:
    cooperatively load this tile's positions and masses into shared memory
    <barrier>
    for each particle in the tile:
        accumulate acceleration from it
    <barrier>
```

Each thread still owns one particle `i` and accumulates into one `Vec2`. What
changes is the `j` loop, which becomes two nested loops:

The accumulator lives across the whole outer loop, so it has to be declared
before it and written back after it.

One key aspect to this algorithm is that it requires many threads to synchronise. In simpler algorithms it's easy to ignore the fact that the kernel runs in parallel. But synchronisation requires us to start thinking in terms of blocks or warps of threads executing at the same time.

**Keep in mind that each kernel really runs many copies of itself and some of those copies can communicate through synchronisation.**

---

In `nbody.cu` there is a version of `calc_acc_tiled` with some of the code replaced with `???`. Comment this back in by removing the `/*` and `*/` before starting.

There are related TODO comments near each `???`. They will walk you through:

- Creating shared memory
- Calculating correct indices into global and shared arrays
- Copying data from global to shared memory
- Processing data a tile at a time
- Choosing appropriate sync points

---

**TODO 1: Declaring a shared array**

Shared memory is (usually) declared *statically* so we have to specify a compile-time size. 

Here, we choose to make the block size and the tile size equal and use  `block_size` as the tile size.

```cpp
__shared__ Vec2 shPosition[block_size];
```

Oh no! Our block_size is defined at runtime!

**Remove the runtime variable `block_size` from `main` and instead declare this at the top as a `const int block_size = ...`.**

**Fill in the second shared array that should hold masses.**

Note: making the block and tile the same size makes some of our indexing easier later but couples the two sizes together, which means we can't independently tune the values.

---

**(Optional) Look at the makefile to see how `BLOCK_SIZE` can be passed from `make` to the code during compilation. Implement an `#ifdef` around your definition of `block_size` to allow setting `block_size` to the value of `BLOCK_SIZE` passed to the compiler.**

**TODO 2: Calculating the thread index**

It's the same as every time before. You got this.

<details>
<summary>Solution</summary>

```cpp
const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
```

</details>

**TODO 3 & 4: Global and local indices**

Now we're inside the loop over tiles and we need to populate the shared arrays with data from the global arrays. `idx` is a global index into the `pos` and `mass` arrays while the index in the brackets of `shPosition[???]` is a local index into the shared memory.

**What should the value of both these indices be?**

<details>
<summary>Hint</summary>

It's useful to consider this from the perspective of one block of threads all running in parallel. Each thread inside the block has a unique index assigned to it in `threadIdx.x`. Since we chose to make the tile size and block size equal, we can also use this value to index into the shared memory space. So the local index should be:

```cpp
shMass[threadIdx.x] = (idx < N) ? mass[idx] : real(0.0);
```

</details>

<details>
<summary>Hint</summary>

The point of this algorithm is to process the data one tile at a time, in chunks of size `block_size`. So the first time through the outer tile loop, `tile_start == 0`, and we copy all values from 0 to `block_size` into the shared memory:

```
|0|1|2|...|31| SHARED
 ^ ^ ^        
|0|1|2|...|31| GLOBAL
```

The second time through the loop, `tile_start == 1 * block_size`, and we copy from `1*block_size` to `2*block_size`:

```
|0 |1 |2 |...|31| SHARED
 ^ ^ ^ ^        
|32|33|34|...|63| GLOBAL
```

So the global index into `pos` and `mass` must combine the offset into the global array given by `tile_start` with the index of the thread, `threadIdx.x`:

```cpp
const int idx = tile_start + threadIdx.x;
```

</details>

**TODO 5: What variables do we access?**

**Fill in the appropriate arguments in the call to `calc_acc_pair`.**

<details>
<summary>Hint</summary>

The first argument is actually the same as before. Each instance of the kernel still owns the calculation for the i-th particle, so the call should look like:

```cpp
calc_acc_pair(pi, TODO, TODO, eps)
```

</details>

<details>
<summary>Hint</summary>

Previously the `calc_acc_pair` call looked like this:

```cpp
calc_acc_pair(pi, pos[j], mass[j], eps)
```

But now, we've saved a tile of `pos[j]` and `mass[j]` to shared memory. Access the equivalent values in shared memory.

</details>

<details>
<summary>Solution</summary>

```cpp
calc_acc_pair(pi, shPosition[j], shMass[j], eps);
```

</details>

**TODO 6: When to sync?**

Consider this algorithm without synchronisation and remember this kernel is run by a block of threads in parallel. If some of the threads finish saving their pieces of the tile and move on to processing the tile, they may access parts of the tile that haven't been populated by slower threads. So we need a method of synchronising at particular points. We do this with the builtin function `__syncthreads();`, but where should we synchronise?

**Consider the three suggested synchronisation points. Which should be implemented?**

<details>
<summary>Hint</summary>

We've already described a situation where faster threads could access unpopulated data, so we need a sync after the copy into shared memory:

```cpp
const int idx = tile_start + threadIdx.x;
shPosition[threadIdx.x] = (idx < N) ? pos[idx] : Vec2{0.0, 0.0};
shMass[threadIdx.x] = (idx < N) ? mass[idx] : real(0.0);
// Sync before we use the data
__syncthreads();
```

</details>

<details>
<summary>Hint</summary>

We also need a synchronisation point before we start copying a new tile. **Why?**

</details>

<details>
<summary>Solution</summary>

Again, some threads will process faster than others, so if a fast thread were finished with processing, it would continue on into the next iteration of the tile loop and start copying the next tile's data into shared memory, *overwriting the existing data* and potentially affecting any slow threads that are still processing that tile. So we need a sync point *either* before the tile copy operation, or after processing, as in the solution:

```cpp
for (int j = 0; j < block_size; j++) {
  // Add the acceleration from jth particle to this
  const Vec2 accll = calc_acc_pair(pi, shPosition[j], shMass[j], eps);
  accl.x += accll.x;
  accl.y += accll.y;
}
__syncthreads();
```

</details>

**TODO 7: Where to store the result?**

**Where in the global array should this thread's `accl` be stored?**

<details>
<summary>Solution</summary>

The tiling part of this algorithm only affects the way in which the `j` particles are processed. We've kept the convention of assigning one `i` particle per thread, so just as we load the position as before, `pos[gtid]` we can store `accl` in the same place as before:

```cpp
if (gtid < N)
  acc[gtid] = accl;
```

</details>

**TODO 8: What's with the `?:` operators?**

You'll have noticed the use of operators like `(idx < N) ? pos[idx] : Vec2{0.0, 0.0}`. **Can you think why we've done this?**

<details>
<summary>Solution</summary>

We guard with ternary as threads trying to access `i >= N` cannot return early - they must still reach every `__syncthreads()` and help load tiles for the rest of the block. Alternatively, using padding avoids this branching.

</details>

---

Phew! Let's now test this version with the unit tests:

**Update the calls to `calc_acc` in `test_calc_acc_x` and `test_calc_acc_y` to use the new tiled version.**

Run with:

```bash
make nbody && ./nbody --only_unit_tests
```

**Update the call to `calc_acc` in the main loop to fully transition to the tiled version.**

You should check the output of `final.csv` with an output from the untiled version to ensure this optimisation hasn't changed the numerical values. If it shows no difference then hooray! You've correctly implemented tiling! Or it isn't actually running the correct kernel...

Let's actually check shared memory is being used.

**Profile with `ncu` and inspect the results.**

In the Launch Statistics section, Static Shared Memory Per Block should be
non-zero and consistent with your tile size × (`sizeof(Vec2)`
+ `sizeof(real)`). If it is zero, you are not running the kernel you think you
are.

---

Before you compare this version to the non-tiled version, take a moment to consider what differences you expect to see in the profile. The aim of the tiling optimisation is to remove global memory accesses; how that might affect various metrics? Can you quantify the speedup you might expect?

**Compare to a non-tiled profile. In which metrics can we see the effect of the tiling optimisation?**

---

Now let's tune the block and tile size. Since we made the block size a compile time constant, we can either manually edit it and recompile or set the value from the make command, as already implemented in `makefile` and the solution `nbody_tiled.cu`. You may have implemented this yourself in an earlier task, but feel free to check the solution.

We should be able to run a similar sweep as before:

```bash
for bs in 32 64 128 256 512 1024; do
    make clean && make nbody BLOCK_SIZE=$bs
    ./nbody -n 100000 --quiet --disable_unit_tests
done
```

**Is the optimal value the same as in the untiled version? Why do you think this is?**

Using `ncu` will give more accurate timing measurements and other metrics.

**Run the tuned version through `ncu` and compare the metrics to a profile of the optimal untiled. What do you notice?**

**Choose a larger number of particles and re-run the experiment. How does each version scale with problem size?**

Note that the problem size scales as `N_PARTICLES^2` so a useful value  is `runtime / N_PARTICLES^2`.

## Task 2.5: Reflection

In the previous task you implemented a tiling optimisation that reuses data to avoid extra memory loads. This is a complex optimisation so well done for getting it done!

Since we are using 1D arrays, the global and local indexing and loop structures are about as simple as they can get, but it's still a complex optimisation. In higher dimensional problems, there are more choices to be made about the shape of the tile, and the indexing becomes significantly more complex, so this isn't an optimisation that can be easily thrown at a problem, although it can be extremely effective.

Tiling is common but not the only use of shared memory. This is just one way of using a programmable cache and you should consider other algorithms that could benefit from some explicit caching.

## Extension tasks

- **Pack position and mass together.** `Vec2` + `real` is currently two separate
  loads from two arrays. A single 16-byte structure would be one wider load.
  What does that do to the shared memory traffic and the instruction count?
- **Thread coarsening.** Give each thread 2 or 4 particles instead of one. The
  tile is then loaded once and reused several times, and the loads amortise
  further. Watch what it does to Registers Per Thread, and whether that starts
  limiting occupancy.
- **Pad the data before tiling.** In order to access within the bounds of the data, the tiled kernel branches: `(gtid < N) ? pos[gtid] : Vec2{0.0, 0.0}`. You can avoid these kinds of branches by padding the input data with null data to contain *exactly* a multiple of the block size. This generally depends on the algorithm but we can achieve this here by padding the position and mass arrays with zeros.
