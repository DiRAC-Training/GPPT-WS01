# Branching kernel

## Learning objective

This exercise will help the learner understand the impact that branching can have on kernel execution time.

## Introduction

This kernel demonstrates the impact of branching in kernels on their runtime.

## The code

Each thread in the kernel will take one of two branches, depending on the thread number and the input parameter provided at program launch.
The input parameter is a measure of the thread divergence.
Giving an argument of 1 means that 50% of the threads in the warp will take one branch, whilst the other takes the second branch.
Providing an argument of 32 means that all threads in the same warp will chose the same branch.
A number in between divides the threads in the warps proportionally.

## Exercise

Firstly, you should build and run the code to make sure it is working correctly.

```sh
./build.sh
./a.out 1
```

Here, we have chosen the input argument for the coherence parameter as 1.
This will create warp with divergent threads.

Profile this run with Nsights Compute:

```sh
ncu -k branching_kernel a.out 1
```

Take note of the runtime for this value of the input parameter.

Now run the profiler again, with a fully coherent warp:

```sh
ncu -k branching_kernel a.out 32
```

In this run, all threads in the warp will chose the same branch.
How does this impact the runtime of the kernel?

Further learning: see how the runtime is impacted by partial coherence within a warp.