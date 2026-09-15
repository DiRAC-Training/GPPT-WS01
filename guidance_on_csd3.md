# CSD3

On CSD3 we have 5 nodes in a SLURM queue and 1 with direct access.

## Direct access

Direct access is via:

```bash
ssh gpu-q-11
```

The GPUs have been split into 28 total virtual GPUs. There are many people running on these devices so you can set your GPU manually with:

```bash
CUDA_VISIBLE_DEVICES=<number between 0 and 27> ./<exe>
```

## Queue access

You can submit an executable directly with:

```bash
srun --gres=gpu:1 --ntasks-per-node 1 -p ampere -t 00:02:00 -A DIRAC-DT001-GPU -q intr ./<executable>
```

## Modules

You may find issues with the default modules. We recommend sourcing the relevant module file in this repo:

```bash
wget https://raw.githubusercontent.com/DiRAC-Training/GPPT-WS01/refs/heads/main/csd3-modules-cpp-cuda.sh
source csd3-modules-cpp-cuda.sh
```
