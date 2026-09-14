# Grouping transfers

## Learning outcome

This exercise demonstrates how the overheads associated with launching data transfers contribute to a longer runtime compared to fewer, larger transfers.

## The code

This program does not launch any kernels, but instead is designed to show the difference between grouping data transfers together and making many smaller transfers.

The program moves the same 128 MB data object from host to device several times, each time dividing the transfer up into a different number of chunks.
The number of these chunks is defined at the beginning of the program.
The transfer happens first as a single transfer, then divided into 10 chunks, then 100 and finally 1000.
Feel free to experiment with these numbers if you so wish.

## Exercise

You will create an Nsight Systems profile of this program, and investigate the output to determine the time taken for each set of transfers.

First, build the program:

```sh
./build.sh
```

Create the Systems profile report:

```sh
nsys profile -o grouping_transfers ./a.out
```

Remember that the `-o` option allows you to define the name of the output report.
Now we can inspect the timeline for a visual representation of the transfer times:

```sh
nsys-ui grouping_transfers.nsys-rep
```

Look over the timeline.
Identify where the different chunked

In particular, try zooming in on the smaller transfers; you will see blank space between each transfer.
These correspond to unavoidable memory movement overheads, and are the main contribution to the overall slowing of the process when carrying out many small transfers.

We can quantify this overhead using another part of the Systems functionality, by turning the reports output into an SQL database and constructing a table from the data.

We can create the database using the `export` command of `nsys`:

```sh
nsys export --type sqlite --output grouping_transfers.sqlite grouping_transfers.nsys-rep
```

We can then run any SQL commands we might want to on the generated database.
For the purpose of this exercise, we will calculate two numbers for each set of transfers: the total runtime from beginning to end of the transfer, and the total amount of time that the GPU is active during these transfers.
The second number corresponds to the time that is spent transferring data, so we can calculate the overhead by subtracting that value from the total runtime.
This can be executed with the following SQL command:

```sh
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

Produce this table for yourself, and see how the timing evolves.