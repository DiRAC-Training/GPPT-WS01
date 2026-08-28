# Launch parameters and occupancy

`main.cu` runs one axpy kernel (`x = 2*x + y`) with a single thread per element. The block size is passed as the only argument; the grid is derived from it, so every run does the same work and only the block geometry changes. It is deliberately minimal — no argument checks or error handling — so harden it if you see fit.

Build, then profile a chosen block size. The `-k` filter restricts the report to the axpy kernel, skipping the Thrust fill kernels used to initialise the data:

```sh
./build.sh
ncu -k axpy_kernel ./a.out <num>    # <num> = threads per block, e.g. 32 64 128 256 768 1024
```

Block size must be a multiple of 32 (the warp size) and at most 1024 threads — a fixed hardware limit, since a block must fit and run on a single SM.

## Block sizes to test
    
| Threads/block | Warps/block | Expected occupancy | What it shows                                                               |
|--------------:|------------:|-------------------:|-----------------------------------------------------------------------------|
|            32 |           1 |               ~50% | capped by the 32-blocks-per-SM limit, not warps — small blocks hurt         |
|            64 |           2 |              ~100% | just enough warps per block to fill the 64 slots                            |
|           128 |           4 |              ~100% | fewer, larger blocks, still 64 warps                                        |
|           256 |           8 |              ~100% | conventional full-occupancy case                                            |
|           768 |          24 |               ~75% | 64 is not divisible by 24, so only 2 blocks fit and 16 warp slots stay idle |
|          1024 |          32 |              ~100% | 2 blocks of 32 warps fill all 64 slots                                      |

## Archived occupancy (A100)

`Section: Occupancy` from `ncu`, recorded on an A100. The `Block Limit` rows cap the theoretical occupancy (smallest limit wins); achieved occupancy is what actually ran and varies between GPUs and runs.

| Metric                      |    32 |    64 |   128 |   256 |   768 |  1024 |
|-----------------------------|------:|------:|------:|------:|------:|------:|
| Block Limit SM              |    32 |    32 |    32 |    32 |    32 |    32 |
| Block Limit Registers       |   128 |    64 |    32 |    16 |     5 |     4 |
| Block Limit Shared Mem      |    32 |    32 |    32 |    32 |     8 |     8 |
| Block Limit Warps           |    64 |    32 |    16 |     8 |     2 |     2 |
| Theoretical Active Warps/SM |    32 |    64 |    64 |    64 |    48 |    64 |
| Theoretical Occupancy %     |    50 |   100 |   100 |   100 |    75 |   100 |
| Achieved Occupancy %        | 14.75 | 22.61 | 54.21 | 81.87 | 60.00 | 80.14 |
| Achieved Active Warps/SM    |  9.44 | 14.47 | 34.69 | 52.39 | 38.40 | 51.29 |

## Worked examples

**32 threads/block** (1 warp): SM 32, Registers 128, Shared 32, Warps 64 -> the warp limit allows 64 blocks, but the **32-blocks-per-SM cap** is the bottleneck -> `32 x 1 = 32 warps = 50%`. Small blocks lose occupancy not on warps, but because each block brings too few warps to fill the 64 slots.

**256 threads/block** (8 warps): SM 32, Registers 16, Shared 32, **Warps 8** -> min 8 -> `8 x 8 = 64 warps = 100%`. When the warp limit is the bottleneck, the warps sum to exactly 64 (it is `64 / warps_per_block`).

## Block limits calculations

```text
resident blocks = min(Block Limit SM, Registers, Shared Mem, Warps)
active warps    = resident blocks x warps per block
occupancy       = active warps / 64            # 64 = max warps per SM (A100)
```

- **Block Limit SM** = 32. A fixed hardware cap on resident blocks, independent of the kernel.
- **Block Limit Registers** = `floor(65536 / (16 x threads_per_block))` = `floor(128 / warps_per_block)`. 256 threads -> `128/8 = 16`; 768 threads -> `128/24 = 5`. (16 = `Registers Per Thread` from the ncu Launch Statistics.)
- **Block Limit Shared Mem** = `floor(carveout / shared_per_block)`. The kernel uses no shared memory, but the driver reserves ~1 KB/block, so against the 8 KB carveout of the 768/1024 runs it is `8192/1024 = 8`. It is never the bottleneck here (warps or the SM cap are always lower).
- **Block Limit Warps** = `floor(64 / warps_per_block)`. 256 threads = 8 warps -> `64/8 = 8`; 32 threads = 1 warp -> `64/1 = 64`.
