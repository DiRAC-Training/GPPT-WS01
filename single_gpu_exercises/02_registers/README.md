# Registers

## Learning outcomes

By the end of this exercise, the learner should understand the impact of running kernels that require many registers on the GPU's occupancy and potential performance.

## Introduction

Registers on the GPU are a finite resource.
If each thread requires too many registers, less threads can be resident and working on the GPU at once.
This kernel aims to show to impact of this 'register pressure' on the theoretical occupancy of the kernel.

## The code

The kernel itself takes 2 double arrays of size N and performs a fixed number of operations on them.
The work done is not important, but rather that the two arrays are cross-coupled and must be held in memory throughout the calculations.
This is done by using the GPU's register memory.
As we increase the size of N, we therefore use more registers per thread.

This exercise invites you to see how increasing the register use impacts the occupancy and performance of the kernel.

## Exercise

You will run the kernel with different values of `N`, and measure its occupancy and timing.

The code is compiled and run with the following commands:

```sh
./build.sh
./a.out
```

You can use the following command to produce a summary of the kernel's performance using Nsight Compute:

```sh
ncu -k register_kernel a.out
```

To change the size of the arrays in the kernel, you will need to edit the program and change the value of `N` set at the top of the file.
Once it's changed and compiled, you should be able to profile it again.

For each value of `N` that you test, you should note down:

- The registers per thread (from the `Launch statistics` section of the output),
- The kernel duration (found in the `Speed of Light Throughput` section),
- The expected occupancy for the kernel (from the `Occupancy` section).

## Hints

Start with `N=2`, then test higher numbers (for example 8 and 50).
Numbers that are too large will cause the memory to spill from registers to local memory, heavily impacting performance.
