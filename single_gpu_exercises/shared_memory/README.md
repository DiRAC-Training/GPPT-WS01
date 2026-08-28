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

## Exercise

To begin, compile and run the code as is.

```sh
./build.sh
./a.out
```

You can use Nsight Compute to profile the performance of the kernel using the following command:

```sh
ncu -k shared_memory_kernel a.out
```

To test the performance of the kernel with different levels of shared memory, you will need to edit the `SHARED_STRIDES` variable at the beginning of `shared_mem.cu`.
A larger value here requests larger shared memory in the kernel.

Begin with the default value (`SHARED_STRIDE = 96`), and test a range of memory sizes.
For each value you test, note down from the Nsight Cmpute output:

- The `Shared Memory Configuration Size` from the `Launch Statistics` table,
- The `Block Limit Shared Mem` from the `Occupancy` table,
- The expected occupancy from the `Occupancy` table.

## Hints

Using values of `SHARED_STRIDES` too large will cause the program to crash because there is not enough shared memory for any blocks.
On an A100, values over 96 will cause the kernel to crash.

Once you reach a low enough request for shared memory, occupancy will no longer be limited by the request.
On the A100, this value corresponds to `SHARED_STRIDES=18`.
Test values between these, for example 96, 64, 32, 24, 20, and 18.