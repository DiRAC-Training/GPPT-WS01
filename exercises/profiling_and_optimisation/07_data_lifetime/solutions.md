# Data lifetime: the same trivial operation, with and without a round trip

Workflow: forward FFT, multiply by 1/N (cuFFT's inverse transform is unnormalised), inverse FFT — which reconstructs the input signal, so the program verifies itself.

The normalisation is a memory-bound, one-multiply-per-element operation that would not be justified offloading to the GPU on its own. `run_naive` copies the spectrum back to the host, normalises it there, and returns it to the GPU; `run_resident` performs it on the GPU between the FFTs.

## Profiling

```bash
./build.sh
nsys profile -o data_lifetime ./a.out
nsys stats --report cuda_gpu_kern_sum data_lifetime.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum data_lifetime.nsys-rep
```

```sh
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ------------------------------------
     95.6        2,000,548          8  250,068.5  249,601.0   227,873   271,585     21,636.6  void regular_fft_factor<...>  (cuFFT)
      4.4           92,896          1   92,896.0   92,896.0    92,896    92,896          0.0  normalise(double2 *, double, int)
```

```sh
 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Operation
 --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ----------------------------
     58.2       21,262,050      6  3,543,675.0  3,674,118.0      1,824  8,871,150  3,448,662.2  [CUDA memcpy Host-to-Device]
     41.8       15,252,761      3  5,084,253.7  5,084,104.0  5,083,945  5,084,712        404.8  [CUDA memcpy Device-to-Host]
```

In the timeline, compare the two variants: the resident normalise kernel is nearly invisible next to the FFTs, while the detour's two extra 64 MB transfers (bracketing a gap of CPU work) cost more than the FFTs themselves. Operations run cheapest wherever the data already lives.
