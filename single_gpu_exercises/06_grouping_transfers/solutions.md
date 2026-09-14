# Grouping transfers

Move the same 128 MB from host to device, split into a growing number of
smaller transfers. The bytes are identical every pass; only the number of
`cudaMemcpy` calls changes. Splitting the transfer costs time on two fronts: the
GPU copy engine reaches a lower effective bandwidth on small transfers, and the
host pays a fixed launch overhead on every synchronous `cudaMemcpy` call.

## Profiling

Each pass uses a distinct transfer size, so grouping the memcpy events by `bytes`
gives one row per pass. Joining the host-side API calls
(`CUPTI_ACTIVITY_KIND_RUNTIME`) to the GPU transfers
(`CUPTI_ACTIVITY_KIND_MEMCPY`) by `correlationId` separates the two costs:
`gpu_ns` is the copy engine busy time, `api_ns` is the host time inside the
blocking call, and `api_ns - gpu_ns` is hence the launch overhead from the multiple copies.

```sh
./build.sh
nsys profile -o grouping_transfers ./a.out
nsys export --type sqlite --output grouping_transfers.sqlite grouping_transfers.nsys-rep
sqlite3 -header -column grouping_transfers.sqlite \
  "SELECT m.bytes,
          COUNT(*) AS calls,
          SUM(r.end - r.start) AS api_ns,
          SUM(m.end - m.start) AS gpu_ns,
          SUM(r.end - r.start) - SUM(m.end - m.start) AS overhead_ns
   FROM CUPTI_ACTIVITY_KIND_RUNTIME r
   JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON r.correlationId = m.correlationId
   GROUP BY m.bytes ORDER BY m.bytes;"
```

```sh
bytes      calls  api_ns    gpu_ns    overhead_ns
---------  -----  --------  --------  -----------
128000     1000   27835646  13919100  13916546
1280000    100    11496019  10038196  1457823
12800000   10     9811789   9649268   162521
128000000  1      9663991   9612501   51490
```

The Nsight Systems GUI timeline shows the same story visually. One solid block for
the single transfer, versus loads of tiny blocks (with gaps) for the
many-transfer cases. Because the copies are synchronous the CUDA API row spans
each whole transfer, so the launch overhead shows up as the idle gaps between
transfers on the GPU memory blocks.  

For the video, I suggest you zoom into the 1000 zone (last part of the small blocks) and look at the gaps between the red blocks, which is the overhead measured with squlite.
