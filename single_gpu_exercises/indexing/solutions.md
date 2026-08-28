# Indexing

Two kernels sum the rows of an `X` by `Y` array (256 columns, 2^20 rows) of
`1.0f`, one thread per row, so every row sum is exactly `X` (256) in both cases.
Only the memory access pattern differs:

- `row_sum_strided` — each thread reads its row contiguously (`row*X + col`), so
  adjacent threads in a warp are X floats (1 KB) apart -> 32 separate cache lines
  per step, uncoalesced.
- `row_sum_coalesced` — indexing is transposed (`col*Y + row`), so adjacent
  threads are 1 float apart -> one contiguous 128-byte transaction per step,
  coalesced.

Both kernels run one after the other, so a single profiler pass captures both.

## ncu

```sh
./build.sh
ncu -k row_sum_strided   ./a.out
ncu -k row_sum_coalesced ./a.out
```

### `row_sum_strided`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.21
    SM Frequency                    Mhz       764.99
    Elapsed Cycles                cycle    2,509,310
    Memory Throughput                 %        99.20
    DRAM Throughput                   %        29.06
    Duration                         ms         3.28
    L1/TEX Cache Throughput           %        99.61
    L2 Cache Throughput               %        43.32
    SM Active Cycles              cycle 2,497,709.57
    Compute (SM) Throughput           %         3.12
    ----------------------- ----------- ------------
```

### `row_sum_coalesced`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.21
    SM Frequency                    Mhz       764.97
    Elapsed Cycles                cycle      588,660
    Memory Throughput                 %        90.26
    DRAM Throughput                   %        90.26
    Duration                         us       769.50
    L1/TEX Cache Throughput           %        26.70
    L2 Cache Throughput               %        93.56
    SM Active Cycles              cycle   581,910.09
    Compute (SM) Throughput           %        13.32
    ----------------------- ----------- ------------
```

## nsys

```sh
nsys profile -o indexing ./a.out
nsys stats --report cuda_gpu_kern_sum indexing.nsys-rep
```

```sh
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                          Name
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ------------------------------------------------
     67.9        3,286,631          1  3,286,631.0  3,286,631.0  3,286,631  3,286,631          0.0  row_sum_strided(float *, float *)
     16.3          789,185          2    394,592.5    394,592.5      7,328    781,857    547,674.7  void cub::...for_each::static_kernel<...>  (device_vector fill)
     15.8          766,817          1    766,817.0    766,817.0    766,817    766,817          0.0  row_sum_coalesced(float *, float *)
```
