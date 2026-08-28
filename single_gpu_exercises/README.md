# Profiling and performance kernel exercises

Welcome to the profiling and performance exercises directory!

Each sub-directory contains a self-contained and documented exercise that will teach an aspect of optimisation in GPU code, and demonstrate how its impact can be seen in the Nsight profiling tools.

If you are new to profilers, we recommend beginning with the `launch_param_sizes` exercise, as this begins with an introduction to running the profiler tools themselves.
Otherise, the exercises can be carried out in any order, though we would recommend the following order, as it lines up with what was taught at the GPPT workshop:

- `launch_param_sizes`,
- `registers`,
- `shared_memory`,
- `indexing`,
- `branching`,
- `grouping_transfers`,
- `data_lifetime`.

The `README.md` in each directory will discuss the concept that the code is demonstrating, and contains a guided exercise that will help the learner understand how we can use the profiler outputs to interpret the possible optimisation.
The `solutions.md` contain further insight into what is happening in the code, as well as expected outputs for the exercises.

The numbers in the solutions were generated on an A100 GPU, so if you are using different hardware your exact values may vary, but the principles will remain.
