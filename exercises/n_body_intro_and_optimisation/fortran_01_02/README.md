# Introduction to OpenMP: Fortran N-Body

You are given a working, CPU-only n-body simulation in a single source file,
`nbody_start.f90`. It is correct and produces the right trajectories, but it
runs entirely on the host. Your job is to accelerate it on an Nvidia GPU
using OpenMP `target` offloading.

You will make all of your changes in `nbody_start.f90`. `nbody_solution.f90` in this
same directory is the finished solution — we recommend you attempt each task
before checking it.

This exercise assumes basic Fortran and some experience with OpenMP `parallel
do` on the CPU. No prior CUDA or GPU experience is needed.

There are two main tasks. Task 1 gets the hot loops running on the GPU.
Task 2 fixes the data transfers, which is where the actual speed shows up.

## What you'll learn

By the end of this exercise you should be able to:

- Offload a Fortran loop to an Nvidia GPU with OpenMP
- Recognise when `target` regions require data transfers and implement them with `!$omp target data`
- Choose the right OpenMP map type (`to`, `from`, `tofrom`, `alloc`) for a given array
- Confirm that code is running on the GPU

## Task 0: Understand the code

The simulation is a 2D direct-summation n-body: every particle feels the
gravitational pull of every other particle, so the acceleration calculation
is O(N^2).

| Subroutine / function       | What it does                                          |
|------------------------------|-------------------------------------------------------|
| `calc_acc`                   | Acceleration on every particle from every other one. **O(N^2), the hot loop, and your main target.** |
| `advance_pos`                 | Velocity-Verlet position update, O(N) per step.        |
| `run_sim`                     | Driver: builds the system, time-steps it, times it.    |
| `generate_random_star_system` | Build initial conditions — CPU-only setup code, not part of this exercise. |
| `test_calc_acc`, `test_advance_pos`, `test_calc_stable_orbit` | Unit tests, run automatically by the `TEST` build. |

Time integration is velocity-Verlet, which needs the previous position from
`pos_prev`. `pos_temp` is scratch space used to shuffle `pos` into
`pos_prev` after each update. You may be able to optimise this out. `epsilon` is a softening length that stops the
force blowing up when two particles are nearly coincident.

### Build and run

Two compilers are used in this exercise:

- **`gfortran`**, with no OpenMP flag, compiles `!$omp` lines as ordinary
  comments — this is how you get a pure CPU baseline, including from a file
  that already has offload directives in it.
- **`nvfortran`** with `-mp=gpu` is what actually offloads
  to the GPU. `-Minfo=mp` is useful; it prints, for every `target`
  region, whether it turned into a GPU kernel.

You can build and run the CPU version with 10000 particles with:

```sh
make cpu && build/test_cpu && build/main_cpu -n 10000
```

Note this builds two executables: the unit tests in `test_cpu` and the main simulation in `main_cpu`. By default the simulation runs for 10 steps. This can be increased with `--steps <number>` but this only makes the timestep runtime measurement slightly more accurate. Feel free to play with this parameter but the we find 10 steps is sufficient.

Similarly, for the GPU version (which currently runs on the CPU):

```sh
make gpu && build/test_gpu && build/main_gpu -n 10000
```

The output should look something like:

```
 Running with        10000  particles
 Complete:    10.0000000    
 Complete:    20.0000000    
 Complete:    30.0000019    
 Complete:    40.0000000    
 Complete:    50.0000000    
 Complete:    60.0000038    
 Complete:    70.0000000    
 Complete:    80.0000000    
 Complete:    90.0000000    
 Complete:    100.000000    
Time to complete:     3.8070 s
Mean time per step:     0.3807 s
```

Note the printed "Mean time per step". This is the main measurement you are aiming to improve.

`./test_cpu` runs the unit tests and aborts if any fail. Keep them passing
throughout — they're a cheap, fast check after every change. Bear in
mind they only exercise a few particles, so they can't catch every mistake
(see Task 2c).

## Task 1: Offload the compute kernels

There are three loops to offload, each marked with a `TODO (Task 1x)`
comment in `nbody_start.f90`. **Use OpenMP to parallelise these.**

**Confirm the compiler is offloading the loops.** The included compiler flag `-Minfo=mp` should report a `Generating "nvkernel_..."`
line for each of the three loops once all of Task 1 is done, and the unit
tests should still pass. The program is very likely **not faster yet**,
possibly slower than the CPU baseline. That's expected, not a bug — see
Task 2.

<details>
<summary>Hint</summary>

For each loop: add `!$omp target teams distribute parallel do` on the line
directly above the loop, and `!$omp end target teams distribute parallel
do` directly below it (after the matching `end do`/`enddo`).

</details>

---

One of these loops is slightly unusual for Fortran:

```fortran
do i = 1, n
  acc(i,1) = 0.0_wp
  acc(i,2) = 0.0_wp
enddo
```

The idiomatic way to write this entire loop would be using a slice like:

```fortran
! do i = 1, n
!   acc(i,1) = 0.0_wp
!   acc(i,2) = 0.0_wp
! enddo

acc(:,:) = 0.0_wp
```

**Why can't we write this while using OpenMP?**

<details>
<summary>Solution</summary>

OpenMP requires *loops*. We have to rewrite all slices as loops to add appropriate OpenMP directives to them.

</details>

---

## Task 2: Keep data on the GPU

With no data directives, every `target` region you added in Task 1 performs data transfers to or from the GPU every single time it runs.

**Which directive handles data transfers in OpenMP?**

<details>
<summary>Solution</summary>

`!$omp target data map(...)`

`!$omp end target data`

This directive opens a region that keeps its mapped
arrays resident on the device for as long as the region is open, regardless
of how many `target` kernels run inside it.

</details>

---

We want to use the `target data` directive to ensure that data stays on the GPU as much as possible during the simulation.

**Identify where you should put a `target data` directive to reduce data transfers associated with `calc_acc` and `advance_pos`**

<details>
<summary>Solution</summary>

Anywhere that `calc_acc` and `advance_pos` are called could be surrounded by `target data` directives, that is:

- The main loop
- The units tests
- The initial condition setup

However, the runtime is dominated by the main loop (and we're only timing that anyway!). So the key region that we need to ensure is covered by `target data` directives is the main loop:

```fortran
!$omp target data TODO map(...) map(...)
call system_clock(count_start, count_rate)

do while (current_step < n_steps)
    call calc_acc(acc, pos, mass)
    call advance_pos(acc, pos, pos_prev, pos_temp, dt)
    current_step = current_step + 1
    if (mod(current_step, print_every) .eq. 0) then
      print *, "Complete: ", real(current_step) / real(n_steps) * 100
    end if
end do
call system_clock(count_end)
!$omp end target data
```

Notice here that we've deliberately placed the `target data` directive before the initial timing call to avoid timing the data transfer itself.

Since both `calc_acc` and `advance_pos` are entirely offloaded to the GPU, we want to ensure data stays on the GPU for the entire main loop, so we surround the loop with the `target data` region.

</details>

---

Now we must decide which variables to map and exactly how they should be mapped.

Recall the `map` clause in the `target data map(...)` directive can take the following options:

- `map(to: [var1], [var2], ...)`
- `map(tofrom: [var1], [var2], ...)`
- `map(from: [var1], [var2], ...)`
- `map(alloc: [var1], [var2], ...)`

**Which variables must be mapped in the `target data` directive and which option should be used for each?**

<details>
<summary>Hint 1</summary>

The variables that should be mapped are all those used in the subroutines: `pos`, `mass`, `pos_prev`, `acc`, and `pos_temp`.

</details>

<details>
<summary>Hint 2</summary>

Strictly, you could choose to use `map(tofrom: ...)` for every single variable here. This would work in this code and probably wouldn't impact performance much. However, in a more complex code where data transfers are intended to happen as part of the main loop, instead of only before and after, using `tofrom` for every variable could be a bottleneck. So let's practice finding the right option here.

Try to consider which variables *need* to be copied in with `map(to: ...)`, which need to be copied out with `map(from: ...)` and which only need to be allocated with `map(alloc: ...)`. You may not need to use all these, and `tofrom` may also be useful.

</details>

<details>
<summary>Solution</summary>

Our solution is:

```fortran
!$omp target data map(tofrom: pos) map(to:mass, pos_prev) map(alloc: acc, pos_temp)
```

`pos` is generated on the host as initial conditions so must be copied in, but is also dumped after the main loop so must also be copied out. `mass` and `pos_prev` are generated on the host but are not dumped so can be copied as just `to`. The other variables do not need to be copied at all and can be simply `alloc`ed.

</details>

## Task 3: Confirm we're running on the GPU

Passing tests and a lower printed time are good signs, but they don't prove
the work is on the GPU rather than, say, silently falling back to the host.
Try the following:

- Run `nvidia-smi` in another terminal while `./main_gpu` is running (use a
  larger particle count, e.g. `-n 1000000`, so the run lasts long enough to
  observe). You should see GPU utilisation and the process listed.
- Set `OMP_TARGET_OFFLOAD=MANDATORY` before running. This tells the OpenMP
  runtime to abort with an error instead of silently running on the host if
  offload isn't actually happening for a `target` region.
- You could even try profiling the code properly with:
    ```bash
    nsys profile -o report build/main_gpu
    nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum report.nsys-rep
    ```

---

**Reflect on the difference in runtime between the explicitly mapped and unmapped versions.**

---

**Optional Task: Predict the performance increase**

We know that the GPU can perform more flops and has faster memory bandwidth than the CPU. If we find out exactly how many more flops, and how much faster bandwidth, we can make some rough prediction about our code's performance. 

**Find out which GPU you are currently using with `nvidia-smi` and which CPU with `lscpu`.**

**Research to find out the theoretical FLOPS and memory bandwidths of both processors.**

What is the factor increase in the FLOPS and memory bandwidth? How does this compare against the real speedup achieved?

Note: Modern GPUs provide most of their FLOPS in tensor cores, which we are not using here.

---

**Optional task**

**Implement `target data` directives around the uses of `calc_acc` and `advance_pos` in the unit tests and initial conditions setup**.

<details>
<summary>Solution</summary>

See the solution in `nbody_solution.f90`.

</details>


### Reflection

Take a moment to note down, or discuss with someone nearby:

- If you were handed a piece of GPU code you hadn't written, what's the
  quickest way to tell whether a repeated kernel call needs an explicit
  `target data` region around it?
- The compiler happily generates a kernel for a `target` region with the
  wrong map type — it isn't a compile error, and it might not even be a
  crash. Given that, what's your own answer to "how do I know my mapping is
  right"?

## Extension tasks

- **Explore the `collapse` clause.** The pairwise loop in `calc_acc` is a perfect square
  (`i` and `j` both run `1..n`), but you only parallelised the outer `i`
  loop. Look up the `collapse` clause and see whether collapsing both loops
  into a single parallel iteration space changes performance, and why (or
  why not). Is this kernel limited by the number of parallel iterations
  available, or by something else?
- **Explore `num_teams` / `thread_limit`.** These clauses let you control the
  GPU launch configuration explicitly instead of leaving it to the
  compiler. Sweep a few values and see whether you can beat the default.
- **Switch between double and single precision with `make ... DOUBLE_PRECISION=true`.** How does the performance change?
