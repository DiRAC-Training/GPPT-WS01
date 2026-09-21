# Shared memory

This example demonstrates the impact of requesting different amounts of shared memory on the occupancy of a kernel.

## The code

In the `main` code, we define a single float vector on the host, `data_h`, populate it with random numbers between 0 and 1, and transfer a copy to the device vector, `data_d`.
We do this through the use of the `thrust` library [TODO: link to Moodle module 20 recap `thrust` section], which carries out data transfers automatically.  

We then call `shared_memory_kernel` to act on this data.
First, let's understand the launch parameters of the kernel.
We have defined our data size and block size (the number of threads per block) as global variables at the beginning of the file.
We use these to calculate how many blocks we require, rounding up so that we have enough blocks to cover every element in our input data even when the data size is not an exact multiple of the block size.
The number of blocks and the block size then make up the first two launch parameters for the kernel.  

For this example, we are also requesting shared memory.
The amount we are requesting is the `SHARED_STRIDES` global variable multiplied by the block size.
If `SHARED_STRIDES == 1`, for example, we will be requesting one float of shared memory per element in the block, with higher numbers requesting additional shared memory.

The kernel itself fills this shared memory in strides in each thread using the thread's data and thread ID.
The threads are synchronised, and then each element of data is summed from its neighbour's shared memory results.

## Suggested actions

To begin, compile and run the code as is.
You may use the build script provided:

```sh
./build.sh
./a.out
```

Now let's look at the Nsight compute output from the terminal for this code:

```sh
ncu a.out
```

Look in particular at the `Section: Occupancy` for the `shared_memory_kernel`.
You should see a table that resembles this one:

```sh
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            3
    Block Limit Warps                     block           16
    Theoretical Active Warps per SM        warp           12
    Theoretical Occupancy                     %        18.75
    Achieved Occupancy                        %        17.05
    Achieved Active Warps Per SM           warp        10.91
    ------------------------------- ----------- ------------
```

Note that this table was recorded on an A100, and your exact results may differ!
The lessons to be learned from the exercise will apply no matter what hardware you use, although some numbers might change.  

[TODO: update this table on final hardware from the automatic performance codes script. This table is from one of the A100s on COSMA]

Our theoretical occupancy for this program is only 18.75%.
We can see what is limiting occupancy by looking at the `Block Limit` metrics in the upper half of the table.
Each `Metric Value` tells us how many thread blocks can be resident on a single SM at once, based on a different limiting resource.
`Block Limit SM` is a fixed hardware cap on the number of blocks an SM can hold, independent of what the kernel uses; it is 32 for the A100.
`Block Limit Registers` is the SM's register file divided by the registers each block needs — the registers per thread (set by the compiler) multiplied by the threads per block — so it shrinks as either grows.
`Block Limit Shared Mem` is the SM's shared memory (up to 164 kB on the A100) divided by the amount each block requests, which here scales with `SHARED_STRIDES`.
`Block Limit Warps` is the SM's maximum number of warps (64 on the A100) divided by the warps per block. A warp is 32 threads, so with `BLOCK_SIZE = 128` that is 4 warps per block and a limit of 64 / 4 = 16.
The number of possible blocks running at once is therefore limited by the smallest value in this part of the table.  

In this instance, we are limited by the Shared Memory requested for the kernel launch.
Only three blocks can be resident on each SM thanks to the size of the required memory.  

Let's calculate our requested memory size to show why this is the case.  

We have requested 96 times the block size of floats for each block, 12288 floats per block.
A float is 4 bytes in total, meaning we are requesting 49152 bytes per block.
The A100 has 164 kB of shared memory per SM, and can therefore only accommodate 3 of the requested blocks before running out of available shared memory.
Filling the SM's warp slots would take 16 of these blocks — the `Block Limit Warps` value — but shared memory limits us to 3. Our theoretical occupancy is therefore $ 3 / 16 = 0.1875 $, the percentage stated in the table above.  

Try changing the size of `SHARED_STRIDES` and see what happens to the occupancy.  

## Expected results

If you have tested several values, you should have noticed that decreasing the size of `SHARED_STRIDES` increases the occupancy of the `shared_memory_kernel`.
You may also have noticed that above a `SHARED_STRIDES` of 96 the kernel fails to launch, as it requests more than the 48 kB of shared memory a block is allowed by default. The code checks the launch result and prints the CUDA error (`invalid argument`) before exiting.  

Find below some example values for the requested size of shared memory, and how it corresponds to occupancy on an A100:

| SHARED_STRIDES | Memory size per block | Blocks/SM | A100 Occupancy |
|----------------|----------------------:|-----------|---------------:|
| 96             |                  49kB | 3         |         18.75% |
| 64             |                  32kB | 4         |            25% |
| 32             |                  16kB | 9         |         56.25% |
| 24             |                  12kB | 12        |            75% |
| 20             |                  10kB | 14        |          87.5% |
| 18             |                   9kB | 16        |           100% |

Below a `SHARED_STRIDES` of 18, the requested shared memory per block is low enough that this is no longer the limiting factor determining the occupancy of the kernel.

## Conclusions

This example was not meant to deter you from using larger amounts of shared memory per block, but to be mindful of doing so.
Using more shared memory limits the number of processes that can run on the device in parallel, but if it is essential for your calculations, this lower occupancy can be an acceptable loss.
What works best for your code can only be determined through profiling and repeated optimisations.
