# Launch parameters and occupancy

This exercise will:
- Introduce the learner to the Nsight profilers, Nsight System and Nsight Compute.
- Demonstrate how changing the block size of a kernel can impact its occupancy on the GPU.

## The code

This program is designed to show the impact of kernel launch parameters on the occupancy of that kernel.

`main.cu` runs one axpy kernel (`x = 2*x + y`) with a single thread per element.
The block size is passed as the only argument; the grid is derived from it, so every run does the same work and only the block geometry changes.

## Exercises

### Exercise 1: Running Nsight System to analysis the workflow of this code

The first exercise of the workshop will be to run the Nsight System profiler on this axpy kernel, and investigate the output.

Firstly, you will need to build the code.

```sh
./build.sh
```

Once this has built, check that running the code works.
The code takes one command line argument, which is the size of the blocks to run on GPU.
For now, you should use 32 as your input for the program.

```sh
./a.out 32
```

The program should complete silently.
If you experience any other results, talk to one of our lovely helpers!

Now, let's profile the code.
We'll do this in two parts: creating the profile report using the `nsys` CLI program, then analysing this report in the Systems GUI.

First, create the profile:

```sh
nsys profile a.out 32
```

Now, open the report in the Systems GUI:

```sh
nsys-ui report1.nsys-rep
```

Take some time to familiarise yourself with the output.

In the timeline, identify:
1) The CUDA hardware track:
  a) Memory transfers,
  b) Kernel executions.
2) The CUDA API track:
  b) The cudaMemCpy calls to and from the device.
  a) The `axpy_kernel` launch.

From the CUDA API track, record the order and time of the CUDA calls in the program.

## Exercise 2: Running Nsight Compute

With the code already built, we can run Nsight Compute on the program directly in the command line.
We will use the `-k` argument to filter the Compute output to only include the kernel of interest, in this case the `axpy_kernel`.

```sh
ncu -k axpy_kernel a.out 32
```

Identify and note down the duration of the kernel.

*Hint*: The `Duration` metric in the `Speed of Light Throughhput` table is the execution time of the kernel.

## Exercise 3: Inspecting occupancy as a function of the block size

Now, you will change the block size through the program's input parameter, and see how this impacts occupancy of the kernel on the GPU.

Run Nsight Compute profiler on the kernel again, with a range of input block sizes.

```sh
ncu -k axpy_kernel a.out <num>
```

The block size must be a multiple of the number of the warp size (i.e. 32).
We recommend testing the block sizes 32, 64, 128, 256, 768 and 1024.
Note that the block size must fit within one SM, and so has a theoretical largest size.
For an A100, this size is 1024.

In the table provided in the shared document, record the input parameter, and the measured values of `Block Limit Warps` and `Expected occupancy`.