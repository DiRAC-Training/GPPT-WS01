# CUDA n-body exercise: presenter and helper notes

These are notes for running and supporting the exercise. The learner
instructions are in [`nbody_exercise.md`](nbody_exercise.md). Learners work in
`nbody.cu`; the other source variants are reference material for presenters and
helpers and should not be presented as alternative starting points.

## Existing A100 measurements

These figures were collected on a COSMA A100. They are curated results, not
complete `ncu` reports. All durations are for one kernel launch selected with
`--launch-skip 1 --launch-count 1`, not the total run time of the simulation.

### One-wave case: N = 221,184

This particle count is 108 SMs × 2048 resident threads, so the tuned launches
fit in one execution wave.

| Kernel | Block/tile size |  Duration |
|--------|----------------:|----------:|
| Naive  |              32 | 136.46 ms |
| Naive  |              64 | 135.60 ms |
| Naive  |             128 | 135.59 ms |
| Naive  |             256 | 133.37 ms |
| Naive  |             512 | 130.87 ms |
| Naive  |            1024 | 121.60 ms |
| Tiled  |              32 | 122.09 ms |
| Tiled  |              64 | 121.20 ms |
| Tiled  |             128 | 119.81 ms |
| Tiled  |             256 | 120.14 ms |

The naive block-size sweep improves by 10.9%, with the best measured value at
1024. The best tiled result is at 128, but it is only 1.5% faster than the tuned
naive result at this particle count. In a single wave, blocks progress through
the interaction loop in step and the naive kernel benefits unusually strongly
from cache broadcast and reuse.

### Multi-wave case: N = 1,000,000

| Kernel            | Duration | Compute (SM) | Memory | L1/TEX | Achieved occupancy |
|-------------------|---------:|-------------:|-------:|-------:|-------------------:|
| Naive, block 1024 |   2.73 s |       83.14% | 41.57% | 45.96% |             94.38% |
| Tiled, tile 128   |   2.45 s |       92.94% | 24.35% | 24.58% |             90.11% |

At 1,000,000 particles, tiling is 11.4% faster than the fully tuned naive
kernel. Both launches now require about 4.5 waves. As blocks finish and new ones
start, the naive blocks drift to different positions in the interaction loop
and lose the implicit cache locality seen in the one-wave case. The tiled
kernel keeps its reuse within each block and continues to scale close to the
expected O(N^2) work; the naive result is about 9.8% slower than quadratic
scaling from the smaller case predicts.

The current code runs 10 steps, which keeps a live profile around 20s per-run.

### Other useful observations

- In the archived 100-step run, four hundred transfer calls account for about
  63 ms. The transfer fix is important for correct data ownership, but changes
  total runtime by only about 0.1% and does not change `calc_acc` Duration.
- Identical whole-program runs on the shared node varied from roughly 40 to 59
  seconds, while `ncu` Duration repeated to about 0.14%.
- Theoretical occupancy reaches 100% early in the naive block-size sweep and
  does not explain the later runtime improvements. Compute (SM) Throughput
  tracks Duration more usefully here.
- The tiled kernel pays two barriers per tile. Its performance turns upward at
  larger tiles even when an individual memory metric continues to improve.
- At one million particles, the slower naive kernel reports higher achieved
  occupancy than the tiled kernel. Treat `ncu` optimisation suggestions as
  hypotheses at best here.
- Earlier FP64 measurements were much slower than the current FP32/`rsqrtf`
  baseline where there was far less scope for these optimisations to
  materialise.
