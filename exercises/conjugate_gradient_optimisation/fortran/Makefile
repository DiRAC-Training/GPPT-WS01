GPU_COMPILE:=nvfortran -mp=gpu -gpu=cc80 -Minfo=mp -g -O3
# GPU_COMPILE:=flang -O3 -fopenmp -march=native --offload-arch=sm_80 -fopenmp-version=50 # Uncomment for flang

.PHONY: cpu gpu fixme cublas cublas_sym clean

all: cpu gpu cublas cublas_sym

BEFORE = precision.f90
AFTER  = test.f90 main.f90

cpu: $(BEFORE) solver.f90 $(AFTER) | build
	gfortran -O3 -march=native $^ -Jbuild -o build/cpu

gpu: $(BEFORE) solver_gpu.f90 $(AFTER) | build
	$(GPU_COMPILE) $^ -o build/gpu

cublas: $(BEFORE) cublas.f90 solver_cublas.f90 $(AFTER) | build
	$(GPU_COMPILE) $^ -cudalib=cublas -o build/cublas

cublas_sym: $(BEFORE) cublas.f90 solver_cublas_sym.f90 $(AFTER) | build
	$(GPU_COMPILE) $^ -cudalib=cublas -o build/cublas_sym

build:
	mkdir -p build

clean:
	rm -rf build *.mod *.o
