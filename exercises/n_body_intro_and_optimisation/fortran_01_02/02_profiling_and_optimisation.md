# Fortran N-Body: Tiling for GPU Shared-Memory Reuse

In this task you will:

- profile the n-body code
- implement a shared memory optimisation to increase data reuse

By the end of this exercise you should be able to:

- understand the basic commands for profiling with `nsys`
- identify long-running kernels
- identify data transfers
- describe a shared memory tiling operation

## Task 1: Profiling the solution

We have already provided a simple script to profile any of the binaries built in `build/` with `nsys`. In this task you can profile your own solutions or the solutions generated with `make solutions`.

### Task 1a: `nsys`

**Look at `profile.sh` to familiarise yourself with the script, the profiling steps, and their flags. Use `nsys -h` to learn about any unfamiliar flags.**

**Run `profile.sh build/main_gpu` to profile your version or `profile.sh build/solution_gpu` to profile the solution.**

Note: this script does not rebuild the binaries. You must call `make` yourself after making changes.

**Inspect the output.** Compare the different sections, the balance of data transfers to time spent in kernels, the various API calls. What do you notice?

### Task 1b: `ncu`

In the previous task, you will have seen the `CUDA GPU Kernel Summary` section:

```
** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)                         Name                        
 --------  ---------------  ---------  ------------  ------------  ----------  -----------  -----------  ---------------------------------------------------
    100.0    1,097,964,621         11  99,814,965.5  98,438,409.0  97,569,790  107,583,746  3,424,131.8  nvkernel_nbody_simulation_calc_acc__F1L122_2_
      0.0          170,497         10      17,049.7      17,056.0      16,384       18,016        439.9  nvkernel_nbody_simulation_advance_pos__F1L224_15_
```

**Note the name of the generated kernel corresponding to `calc_acc`.  The F1L* name might change in your solution as it indicates a line number**

---

Now let's **profile this particular kernel with `ncu`**:

```bash
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
--section LaunchStats --launch-skip 1 --launch-count 1 -k <KERNEL NAME> \
<BINARY> -n 10000
```

**Inspect the output.** The various sections provide information and recommendations. Note in particular the first section's recommendation. It should be complaining about there being too few blocks. You might have noticed above that we've used only 10k particles, a problem size that seems a little too small for this GPU.

**Try profiling again with more particles: 100k, 1M, higher?** How does the profile change? 

Profiling with 100k particles adequately stresses the RTX 3060 that this exercise was tested on. Other GPUs may need even larger values to properly profile the kernels.

**(Optional task) Edit the name of the kernel in the `ncu` command to an invalid kernel name and rerun the command.** What do you notice?

Helpfully, `ncu` can report the names of valid kernels. So you don't need to run `nsys` every time you need the name of a generated kernel. Finding the names of generated kernels can generally be tricky. You may have to inspect the compilation output to track down the exact name of the kernel you intend to profile.

## Task 2: Implementing a shared memory tiling optimisation

### Shared memory and tiling

A GPU has two memory tiers that matter here. **Global memory** is large and
visible to every thread on the device, but slow to access relative to how
fast the GPU can compute. **Shared memory** is much faster, but tiny (tens
of KB), and private to one *thread block* — a group of threads the GPU
schedules together, which can synchronise with each other and share that
fast memory.

In `calc_acc`, every GPU thread owns one particle `i` and loops over all `n`
source particles `j`, reading `pos(j,:)` and `mass(j)` straight from global
memory every time. Threads in the same block work through that loop at
roughly the same pace, so at any given moment, many of them are asking
global memory for the *same* source particle at the *same* time — the same
value fetched from slow memory once per thread, instead of once per block.

**Tiling** makes the reuse explicit instead of hoping the cache absorbs it. The idea is:

Split the `n` source particles into small chunks ("tiles"), sized to match
the block. For each tile: every thread in the block cooperates to copy
exactly one particle from global memory into the block's shared memory (so
the whole tile costs one coordinated read per particle, not one per thread
per particle), then every thread reads that tile back out of shared memory
as many times as it needs, before the block moves on to the next tile. In
pseudocode:

```
for each tile of source particles:
    every thread loads one particle into the block's shared tile arrays
    <barrier: wait for the whole tile to be loaded>
    every thread accumulates its own acceleration from the tile
    <barrier: wait for everyone to finish reading before it's overwritten>
```

### Shared memory with OpenMP

OpenMP has no `__shared__` keyword the way CUDA does. To reach the same
hardware feature, you have to give up the convenient combined
`target teams distribute parallel do` directive you've used everywhere so
far, and separate it into two directives:

- `!$omp distribute` — spreads loop iterations across GPU thread **teams**.
  A team is OpenMP's name for what CUDA calls a thread block.
- `!$omp parallel`, nested inside the `distribute` loop's body — spreads
  work across the **threads within one team**.

We will declare the tile-staging arrays as ordinary local arrays, and list them in
a `private()` clause on `distribute`. That clause is what asks for one
instance of the array *per team*, shared by every thread in that team —
the compiler places an array like that in real GPU shared memory. You can
(and should) confirm this yourself: compile with `-Minfo=mp` and look for a
line like `Team private (..., pos_s, mass_s) located in CUDA shared
memory`. If you don't see "located in CUDA shared memory" there, the
arrays didn't end up where you think.

`!$omp barrier`, used inside the nested `parallel` region, is OpenMP's
version of CUDA's `__syncthreads()`: it forces every thread in the team to
reach that point before any of them continues. You need it twice per tile —
once after the load (nobody may read the tile until everyone's finished
writing it) and once after the accumulation (nobody may start the next
tile's load until everyone's finished reading this one).

### The task

Start from the solution to the previous exercise: `nbody_solution.f90`. `calc_acc_tiled` has been partially implemented for you, with the pieces you should fill in marked with `???` and TODO comments.

Build the tiled version with `make tiled`. This compiles `nbody_solution.f90`
with `-DTILED`, producing `build/main_gpu_tiled` and `build/test_gpu_tiled`. `-DTILED` forces the use of `calc_acc_tiled` instead of `calc_acc` however it should fail to compile! You'll have to address all the TODOs before it compiles.

Work through the TODOs in `calc_acc_tiled` from top to bottom. Each one
below gives you a question to think about, hints if you want them, and
the solution if you get stuck. Try to struggle with the problem yourself, or with a neighbour, before using the hints or solutions.

---

**`real(wp) :: ??? ! TODO create a shared array for the mass tile`** — what
should this declaration look like?

<details>
<summary>Hint 1</summary>

`pos_s` right above it creates one tile's worth of positions, one row per
particle in the tile. `mass` needs the same kind of array.

</details>

<details>
<summary>Hint 2</summary>

Mass has no x/y component, so this array only needs one dimension, sized
`TILE` — same as the first dimension of `pos_s`.

</details>

<details>
<summary>Solution</summary>

```fortran
real(wp) :: mass_s(TILE)
```

</details>

---

**`num_teams_needed = ??? ! TODO how many tiles do we need to cover n?`**

<details>
<summary>Hint</summary>

Each team handles one tile of `TILE` particles. `n` won't usually divide
evenly by `TILE`, so you need to round up, not down.

</details>

<details>
<summary>Solution</summary>

```fortran
num_teams_needed = (n + TILE - 1) / TILE
```

</details>

---

**`! TODO these omp directives are out of order! Fix them`** — this covers
two problems at once: the directive order and where `!$omp parallel` opens.

<details>
<summary>Hint</summary>

Re-read the section on shared memory above: `target teams` is what *creates* the
teams (CUDA blocks), so it has to be the outermost directive. `distribute`
then spreads the `do team_id` loop across those teams. `parallel` spreads
work across the threads *within* one team — so it needs to open once per
team, not once for the whole kernel.

</details>

<details>
<summary>Solution</summary>

```fortran
!$omp target teams num_teams(num_teams_needed) thread_limit(TILE)
!$omp distribute private(pos_s, mass_s)
do team_id = 0, num_teams_needed - 1
    !$omp parallel private(tid, i, ax, ay, t, tile_i, j, dx, dy, dist_sq, inv_dist_cube)
```

Note that `calc_acc_tiled` is called from inside a data region so the `pos`,
`mass` and `acc` arrays are already on the device, so there is no additional
data movement required. Only the per-team staging arrays need handling here,
using the `private()` clause.

</details>

---

**`i = ??? ! TODO calculate the global index for this thread`**

<details>
<summary>Hint 1</summary>

Team 0 owns particles `1..TILE`, team 1 owns `TILE+1..2*TILE`, and so on.
Within team `team_id`, thread `tid` (0-indexed) owns which particle in that
range?

</details>

<details>
<summary>Hint 2</summary>

`team_id * TILE` gets you to the start of this team's block of particles;
`tid` then offsets within it. Remember Fortran arrays are 1-indexed.

</details>

<details>
<summary>Solution</summary>

```fortran
i = team_id * TILE + tid + 1
```

</details>

---

**`! TODO do we need a barrier here?`** (the first one, right before loading
the tile)

<details>
<summary>Hint 1</summary>

A barrier protects shared data from being read too early or overwritten too
early. At this exact point in the loop, has anything unsafe happened to
`pos_s`/`mass_s` yet?

</details>

<details>
<summary>Hint 2</summary>

Look ahead to the *other two* barrier TODOs further down — one of them
already guarantees "nobody starts the next tile's load until everyone's
finished reading the previous tile." Does that barrier already cover this
spot?

</details>

<details>
<summary>Solution</summary>

No barrier needed — delete the TODO comment and leave this blank. The
barrier at the end of the loop body (see below) already ensures the tile
isn't overwritten until every thread has finished reading it, which is the
only thing that would make this spot unsafe.

</details>

---

**`tile_i = ??? ! TODO similar to i, calc global index for this thread's tile position`**

<details>
<summary>Hint</summary>

Same idea as computing `i`, but you're now indexing into tile `t` (the loop
variable) instead of team `team_id`.

</details>

<details>
<summary>Solution</summary>

```fortran
tile_i = t * TILE + tid + 1
```

</details>

---

**`! TODO do we need a barrier here?`** (after loading the tile, before
reading it)

<details>
<summary>Hint</summary>

Every thread in the team loads its own slot of `pos_s`/`mass_s` in
parallel — some finish before others. What happens if a fast thread starts
the accumulation loop below while a slower thread hasn't written its slot
yet?

</details>

<details>
<summary>Solution</summary>

Yes — add `!$omp barrier` here.

</details>

---

**`dy = ??? ! TODO`**

<details>
<summary>Hint</summary>

Look at the line directly above it, computing `dx`.

</details>

<details>
<summary>Solution</summary>

```fortran
dy = pos_s(j,2) - pos(i,2)
```

</details>

---

**`ax = ax + dx * mass??? * inv_dist_cube ! TODO how to access mass within the tile?`**

<details>
<summary>Hint 1</summary>

You should access the *shared* mass array, `mass_s`, here, but which index to use?

</details>

<details>
<summary>Hint 2</summary>

The index into the shared tile is `j`.

</details>

<details>
<summary>Solution</summary>

```fortran
ax = ax + dx * mass_s(j) * inv_dist_cube
ay = ay + dy * mass_s(j) * inv_dist_cube
```

</details>

---

**`! TODO do we need a barrier here?`** (after accumulating from the tile,
before looping back to load the next one)

<details>
<summary>Hint</summary>

Flip the question from the previous barrier: some threads may finish
reading the tile in the accumulation loop before others. What happens if a
fast thread jumps ahead to the next `t` iteration and starts overwriting
`pos_s`/`mass_s` while a slower thread is still reading the current values?

</details>

<details>
<summary>Solution</summary>

Yes — add `!$omp barrier` here.

</details>

---

**`! TODO I'm worried this will access out-of-bounds. Should there be a guard?`**
(just before `acc(i,1) = ax`)

<details>
<summary>Hint 1</summary>

`n` won't always be an exact multiple of `TILE`. In the last team, what does
`i` look like for a thread whose `tid` runs past the number of real
particles left?

</details>

<details>
<summary>Hint 2</summary>

You already guarded the tile load (`if (tile_i <= n)`) and the accumulation
loop (`if (i <= n)`) against this. This write needs the same treatment.

</details>

<details>
<summary>Solution</summary>

```fortran
if (i <= n) then
    acc(i,1) = ax
    acc(i,2) = ay
end if
```

</details>

---

Once every TODO is filled in, rebuild and test:

```sh
make tiled && build/test_gpu_tiled && build/main_gpu_tiled -n 10000
```

Check the
output for a line like `Team private (..., pos_s, mass_s) located in CUDA
shared memory` — if you don't see "located in CUDA shared memory", the
staging arrays didn't end up where you think, even if the tests pass.
Finally, compare `Mean time per step` against the untiled `build/main_gpu
-n 100000` from the previous exercise.

This is a very error-prone kind of code so do feel free to compare to the solution in `nbody_tiled_solution.f90` if things aren't going well. The point of this exercise is to learn how tiling works, not to bash your head against bugs (you can do that at your leisure).

---

**Profile the tiled version in the same way as in task 1.** 

Feel free to profile the solutions using:

```bash
./profile.sh build/solution_gpu
./profile.sh build/solution_gpu_tiled
```

... and the corresponding `ncu` calls detailed in task 1.

**What speedup did you achieve?**

---

Comparing the two `ncu` profiles, you should notice something memory-related that distinguishes the tiled version.

<details>
<summary>Solution: Warp cycles</summary>

**Non-tiled:**

```
Section: Warp State Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Warp Cycles Per Issued Instruction             cycle        14.94
Warp Cycles Per Executed Instruction           cycle        14.94
Avg. Active Threads Per Warp                                   32
Avg. Not Predicated Off Threads Per Warp                    31.80
---------------------------------------- ----------- ------------
```

**Tiled:**

```
Section: Warp State Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Warp Cycles Per Issued Instruction             cycle         8.81
Warp Cycles Per Executed Instruction           cycle         9.29
Avg. Active Threads Per Warp                                27.55
Avg. Not Predicated Off Threads Per Warp                    27.39
---------------------------------------- ----------- ------------
```

The number of cycles spent waiting has significantly decreased (by about the same factor as the decrease in overall runtime). This implies that our tiling worked! Fewer warps are now waiting on data accesses.

</details>

<details>
<summary>Solution: Occupancy</summary>

**Non-tiled:**

```
Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block           10
Block Limit Shared Mem                block           16
Block Limit Warps                     block           12
Theoretical Active Warps per SM        warp           40
Theoretical Occupancy                     %        83.33
Achieved Occupancy                        %        74.94
Achieved Active Warps Per SM           warp        35.97
------------------------------- ----------- ------------
```

**Tiled:**

```
Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block            6
Block Limit Shared Mem                block            8
Block Limit Warps                     block            9
Theoretical Active Warps per SM        warp           30
Theoretical Occupancy                     %        62.50
Achieved Occupancy                        %        57.86
Achieved Active Warps Per SM           warp        27.77
------------------------------- ----------- ------------
```

Unfortunately, our occupancy has been limited due to the increase in register use. Were we optimising further, reducing register use would be a decent thing to explore next.

Overall, the improvement in data reuse has balanced against more limited occupancy so we gain from this particular optimisation here.

</details>

---

You may also notice a decrease in memory and compute throughput! Since these metrics measure overall performance, it might be surprising that a decrease in these can coincide with an improvement in runtime. But optimising for runtime is complex so we cannot rely wholly on one metric, even an important one like throughput. We must use a *collection* of relevant metrics, our own understanding of the hardware, code and underlying algorithms, and intuition that develops with experience in profiling and GPU programming.

## Extension tasks

- **Switch between double and single precision with `make ... DOUBLE_PRECISION=true`.** How does the performance change? Is the tiling optimisation still useful?
- Consider how you might implement a slightly different algorithm. Everything in this exercise had one particle per thread. What do you think changes about the tiling logic — the indexing, the barriers, the padding — if a thread instead owned several particles? Why might one want to make this change?

### References

- [OpenMP Application Programming Interface, version 5.2](https://www.openmp.org/spec-html/5.2/openmp.html)
  — the authoritative reference for every directive and map type used here.
- [NVIDIA HPC SDK: OpenMP GPU Programming with the NVIDIA HPC Compilers](https://docs.nvidia.com/hpc-sdk/compilers/openmp-gpu/index.html)
  — `nvfortran`-specific detail on how `target` regions map onto CUDA
  concepts, including shared memory placement for `private` arrays.
