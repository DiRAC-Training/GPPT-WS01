# Branching

```sh
./build.sh
ncu -k branching_kernel ./a.out 1     # 50/50 split of threads within a warp
ncu -k branching_kernel ./a.out 32    # all the warp doing the same branch
```

## `./a.out 1`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.21
    SM Frequency                    Mhz       764.94
    Elapsed Cycles                cycle      622,706
    Memory Throughput                 %         2.48
    DRAM Throughput                   %         2.48
    Duration                         us       814.05
    L1/TEX Cache Throughput           %         0.95
    L2 Cache Throughput               %         2.99
    SM Active Cycles              cycle   618,898.86
    Compute (SM) Throughput           %        98.05
    ----------------------- ----------- ------------
```

## `./a.out 32`

```sh
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.21
    SM Frequency                    Mhz       764.95
    Elapsed Cycles                cycle      320,008
    Memory Throughput                 %         4.97
    DRAM Throughput                   %         4.97
    Duration                         us       418.34
    L1/TEX Cache Throughput           %         1.82
    L2 Cache Throughput               %         5.71
    SM Active Cycles              cycle   316,876.72
    Compute (SM) Throughput           %        95.88
    ----------------------- ----------- ------------
```

Highlight `Duration` and `Elapsed Cycles` that are almost half when switching from the divergent case where each warp has 16 threads (each taking an arm), to the coherent case where all 32 threads take the same arm.  

`ncu` also confirms this is a fully compute bound workload with no memory throughput and some other metrics you could go over that make it seem like they are very similar in terms of stats (identical occupancy and launch stats), yet are this far apart.

Adding the option `--metrics smsp__thread_inst_executed_per_inst_executed.ratio` also shows the ratio with 16.6 with divergent and 32 with coherent, so it all seems to be working.

**Note:** This shows something really cool where ncu in both case says "compute bound over 80%, you're awesome" (at least CLI):

```sh
INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.
To further improve performance, work will likely need to be shifted from the most utilized to another unit.
Start by analyzing workloads in the Compute Workload Analysis section.
```

Yet there is up to x2 speed-up on the table for the fully divergent case by grouping branches into warp-sized coherent bundles.
