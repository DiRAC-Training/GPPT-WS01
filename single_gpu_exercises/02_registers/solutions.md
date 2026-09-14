# Registers

Each thread holds two `double` arrays of `N` cross-coupled temporaries and works
them for a fixed number of steps. `N` is the register-pressure lever: raise it
and each thread needs more live registers, so fewer blocks fit per SM and
occupancy falls. `iters = WORK / (N - 1)` keeps the total arithmetic constant, so
the only thing changing between runs is register pressure, not the work done.

Edit `N` in `main.cu`, rebuild, and profile:

```sh
./build.sh
ncu -k register_kernel ./a.out
```

Watch as `N` grows: `Registers Per Thread` climbs and both `Block Limit
Registers` and `Achieved Occupancy` fall. Across this sweep the arrays stay in
registers (no local-memory spill even at 206 registers/thread), so occupancy is
purely register-limited; pushing `N` higher still would eventually spill to
local memory (the sphNG failure mode).

## `N = 2`

`Registers Per Thread: 16`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           1.21
    SM Frequency                    Mhz         765.00
    Elapsed Cycles                cycle    121,617,485
    Memory Throughput                 %           0.02
    DRAM Throughput                   %           0.02
    Duration                         ms         158.98
    L1/TEX Cache Throughput           %           0.01
    L2 Cache Throughput               %           0.03
    SM Active Cycles              cycle 121,377,781.36
    Compute (SM) Throughput           %          99.80
    ----------------------- ----------- --------------
```

```sh
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp           64
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        97.47
    Achieved Active Warps Per SM           warp        62.38
    ------------------------------- ----------- ------------
```

## `N = 8`

`Registers Per Thread: 44`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           1.21
    SM Frequency                    Mhz         765.00
    Elapsed Cycles                cycle    225,840,907
    Memory Throughput                 %           0.01
    DRAM Throughput                   %           0.01
    Duration                         ms         295.22
    L1/TEX Cache Throughput           %           0.00
    L2 Cache Throughput               %           0.02
    SM Active Cycles              cycle 225,397,543.34
    Compute (SM) Throughput           %          99.80
    ----------------------- ----------- --------------
```

```sh
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block            5
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp           40
    Theoretical Occupancy                     %        62.50
    Achieved Occupancy                        %        61.31
    Achieved Active Warps Per SM           warp        39.24
    ------------------------------- ----------- ------------
```

## `N = 50`

`Registers Per Thread: 206`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           1.21
    SM Frequency                    Mhz         765.00
    Elapsed Cycles                cycle    245,699,320
    Memory Throughput                 %           0.01
    DRAM Throughput                   %           0.01
    Duration                         ms         321.18
    L1/TEX Cache Throughput           %           0.00
    L2 Cache Throughput               %           0.01
    SM Active Cycles              cycle 245,183,450.02
    Compute (SM) Throughput           %          97.79
    ----------------------- ----------- --------------
```

```sh
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block            1
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp            8
    Theoretical Occupancy                     %        12.50
    Achieved Occupancy                        %        12.10
    Achieved Active Warps Per SM           warp         7.74
    ------------------------------- ----------- ------------
```
