# GPPT N-body porting & profiling exercises

This code is a 2D direct-summation n-body simulation: every particle feels the
gravitational pull of every other particle, so the acceleration calculation
is O(N^2). The time advancement is performed with a 2nd-order velocity-Verlet method.

Before running any exercises, you should `git clone <URL>` this repo to CSD3.

**Important**: [Guidance on running on CSD3](https://github.com/DiRAC-Training/GPPT-WS01/blob/main/guidance_on_csd3.md)

## Porting exercises (Day 1)

- [C++ CUDA Porting](https://github.com/DiRAC-Training/gppt-ws01-nbody/tree/main/cuda_01_introduction)
- [Fortran OpenMP Porting](https://github.com/DiRAC-Training/gppt-ws01-nbody/tree/main/fortran_01_02)

## Profiling & optimisation exercises (Day 2)

- [Fortran OpenMP](https://github.com/DiRAC-Training/gppt-ws01-nbody/blob/main/fortran_01_02/02_profiling_and_optimisation.md)
- [C++ CUDA](https://github.com/DiRAC-Training/gppt-ws01-nbody/tree/main/cuda_02_optimisation)

# Some helpful information

## Checking whole-code correctness

Both codes will dump out a CSV containing some trajectories called `final.csv` which can be copied and checked against with a command like `diff final_copy.csv final.csv`. Although comparing two runs on different hardware is unlikely to be useful, you can usually compare two runs on the same hardware when making small changes to have some confidence you haven't affected the numerical output.

## Visualising output

You may find it useful to visualise the output of the simulation. We have provided a Python visualisation script in `../visualisation/plot.py` that can be used to render CSV outputs to PNG images:

```bash
../visualisation/plot.py *.csv
```

If the Python packages are not already available, see the README.md in the visualisation folder for instructions on getting this script running.

There is also a video plotting tool that plots all CSVs in the current folder and outputs an MP4 animation of the simulation:

```bash
../visualisation/plot_video.sh
```

## WARNING: Timing can be tricky

Do not rely **solely** on the wall-clock number the program prints on its own to
decide whether a change helped. You are on a shared node, so your runtime will be affected by other users on the system.

## References

- Nyland, Harris and Prins,
  [*Fast N-Body Simulation with CUDA*](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda),
  GPU Gems 3 chapter 31 (NVIDIA, 2007).
- Volkov,
  [*Better Performance at Lower Occupancy*](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf),
  GTC 2010.
