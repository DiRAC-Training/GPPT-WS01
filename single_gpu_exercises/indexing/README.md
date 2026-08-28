# Indexing and coalesced memory

## Learning outcome

This exercise demonstrates the importance of coalescing memory 

## The code

The program uses a data structure with 256 columns and 2^20 rows.
It contains two kernels: both sum the rows of the data structure, but use different indexing methods.
Both produce the same correct answer, but access memory in different ways.

The memory access patterns for each are:

- `row_sum_strided` — each thread reads its row contiguously (`row*X + col`), so adjacent threads in a warp are X floats (1 KB) apart -> 32 separate cache lines per step, uncoalesced.
- `row_sum_coalesced` — indexing is transposed (`col*Y + row`), so adjacent threads are 1 float apart -> one contiguous 128-byte transaction per step, coalesced.

The two kernels run sequentially in the main program, so one profiler pass captures both outputs.

## Exercise

Using the Nsight profilers, you will compare the runtimes of the coalesced and strided kernels.

Begin by compiling the code:

```sh
./build.sh
```

### Run time from Nsight Compute

Firstly, we will extract the run times from the Nsight Compute profiler.

Run the profiler on each kernel using the commands below, and note down the run time for each kernel.

```sh
ncu -k row_sum_strided   ./a.out
ncu -k row_sum_coalesced ./a.out
```

### Run time from Nsight Systems

We can also extract the kernel durations from an Nsight Systems profile.
We do this by first running the Systems profiler to create a report object:

```sh
nsys profile -o indexing ./a.out
```

The `-o` option here names the output report `indexing`.
This is a useful option to keep track of your profiles.

We can then use the `stats` command of `nsys` to access the information in the report.
Using the `--report cuda_gpu_kern_sum` will produce a table that summarises the CUDA kernel launched during the run.
Run the following command, and check that the duration of your profiled kernels is similar to those found using Nsight Compute:

```sh
nsys stats --report cuda_gpu_kern_sum indexing.nsys-rep
```

Note down the `Total Time (ns)` values with those from Compute.
