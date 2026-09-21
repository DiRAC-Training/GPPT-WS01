#!/usr/bin/env bash

set -e

bin=$1
bname=$(basename "$bin")
repname="report_$bname"

if test ! -f "$repname.nsys-rep"; then
  echo "Profiling $bin"
  echo
  nsys profile -o "$repname" --force-overwrite=true "$bin" -n 100000
else
  echo "Skipping profile. To re-run:"
  echo "rm $repname.nsys-rep"
  echo
fi

nsys stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum "$repname.nsys-rep"
