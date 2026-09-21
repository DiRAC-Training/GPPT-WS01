all: cpu gpu

ifdef DOUBLE_PRECISION
FLAGS:=
else
FLAGS:=-DSINGLE_PRECISION
endif

GPU_COMPILE:=nvfortran -cpp -mp=gpu -Minfo=mp -g -O3
# GPU_COMPILE:=flang -cpp -O3 -fopenmp -march=native --offload-arch=sm_80 -fopenmp-version=50

cpu: nbody_start.f90 | build
	gfortran -cpp -DMAIN $(FLAGS) -O2 nbody_start.f90 -o build/main_cpu
	gfortran -cpp -DTEST $(FLAGS) -O2 nbody_start.f90 -o build/test_cpu

gpu: nbody_start.f90 | build
	$(GPU_COMPILE) -DMAIN $(FLAGS) nbody_start.f90 -o build/main_gpu
	$(GPU_COMPILE) -DTEST $(FLAGS) nbody_start.f90 -o build/test_gpu

tiled: nbody_solution.f90 | build
	$(GPU_COMPILE) -DMAIN  $(FLAGS) -DTILED nbody_solution.f90 -o build/main_gpu_tiled
	$(GPU_COMPILE) -DTEST $(FLAGS) -DTILED nbody_solution.f90 -o build/test_gpu_tiled

solutions: nbody_solution.f90 nbody_tiled_solution.f90 | build
	gfortran -cpp -DMAIN $(FLAGS) -O2 nbody_solution.f90 -o build/solution_cpu
	$(GPU_COMPILE) -DMAIN $(FLAGS) nbody_solution.f90 -o build/solution_gpu
	$(GPU_COMPILE) -DMAIN $(FLAGS) -DTILED nbody_tiled_solution.f90 -o build/solution_gpu_tiled

build:
	mkdir build

clean:
	rm -f *.mod *.csv
	rm -rf build

.PHONY: all tiled clean
