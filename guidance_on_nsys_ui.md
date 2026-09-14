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
