# Data lifetime

## Learning outcome

This program demonstrates the cost of unnecessary data transfers.
The learner will see how extra data transfers can be less performant than inefficient code in certain scenarios.

## Introduction

During a code's operation, there will be tasks that are better suited for running on the GPU, and tasks that would be better handled by the CPU.
Depending on the ordering and criticality of these tasks, sometimes it can be more efficient to port the CPU code to the device rather than incur additional data transfers.

## The code

The code in this directory is one such example.

The program's workflow goes as such:

- Carry out a forward FFT on the GPU using the cuFFT library,
- Carry out a normalisation on this data (as cuFFT does not narmalise its output),
- Then carry out an inverse FFT.

This trivial operation reconstructs the input signal, allowing the program to validate itself.

The normalisation is a memory-bound, one-multiply-per-element operation that would not be justified offloading to the GPU on its own.
THe program therefore presents two versions of the above workflow:

- `run_naive`, which copies the spectrum back to the host, normalises it there, and returns it to the GPU.
- `run_resident`, which performs the normalisation on the GPU between the FFTs.

## Exercise

You will examine the different in run time between the two options using the Nsight Systems profiler.

First, build the program:

```bash
./build.sh
```

Now, generate the Nsight Systems report for the program:

```sh
nsys profile -o data_lifetime ./a.out
```

Remember that the `-o` option lets you name the output report.
Look at the output in the Nsights Compute GUI.

```sh
nsys-ui data_lifetime.nsys-rep
```

Identify the normalisation kernel, and compare it visually to the data transfers.

We can quantify the time using the `stats` command of `nsys`.
Using the `--report` option, we can make summary reports of the kernel executions in the program using the `cuda_gpu_kern_sum` option, and of the data transfers by using the `cuda_gpu_mem_time_sum` option.
Running the two commands below will give summary tables that tell us how long these actions took:

```sh
nsys stats --report cuda_gpu_kern_sum data_lifetime.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum data_lifetime.nsys-rep
```

Note down the average time a data transfer takes, along with the time of the `normalise` kernel.