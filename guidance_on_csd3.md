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
# Using the Nsight Systems GUI from CSD3

Sometimes viewing reports from `nsys` is better done in the GUI. There are a few ways we can do this:

1. **Recommended** [Install Nsight systems locally](https://developer.nvidia.com/nsight-systems/get-started) and use it to view reports generated on CSD3. See below for guidance on downloading these reports.
2. Use X forwarding in SSH to run `nsys-ui` directly on CSD3:
    ```bash
    ssh -X ...
    nsys-ui report.nsys-rep
    ```
    This can be slow but may be enough for a quick glance at the visual timeline.

**Accessing reports from CSD3:**

1. **Recommended** Use `sshfs` to mount a folder on CSD3 to your local filesystem:
    ```bash
    mkdir csd3_mnt
    sshfs <USERNAME>@login.hpc.cam.ac.uk:/home/<USERNAME> csd3_mnt
    ```
2. Use `scp`, `rsync` or some other file transfer tool to your local machine:
    ```bash
    scp <USERNAME>@login.hpc.cam.ac.uk:/home/<USERNAME>/workshop/report.nsys-rep <destination>
    ```

# Running `ncu` on CSD3

If you're logged into the direct access node (`gpu-q-11`) and you try to run `ncu` you might get the error:

```
==ERROR== Cannot lock GPU clock frequencies on MIG! Try locking the clocks externally (e.g. using 'nvidia-smi --lock-gpu-clocks=tdp,tdp') or profile without fixed frequencies (see '--clock-control').
```

You can add the following flag to avoid the error:

```bash
ncu --clock-control none <executable>
```

OR submit to the queue:

```bash
srun --gres=gpu:1 --ntasks-per-node 1 -p ampere -t 00:02:00 -A DIRAC-DT001-GPU -q intr ncu ...
```
