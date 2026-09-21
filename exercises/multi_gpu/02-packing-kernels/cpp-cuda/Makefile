THIS_TASK := 02
OUTPUT_NAME := packing.$(THIS_TASK)__$(shell date '+%Y%m%d-%H%M')

NP ?= 4
SUBMIT_CMD ?= srun --gres=gpu:4 --ntasks-per-node 4 -p ampere -t 00:02:00 -A dirac-dt001-gpu

EXEC=packing.ex
MPICXX=mpicxx
export OMPI_CXX=nvcc # Open MPI specific
TARGET_GPU_ARCH?=80
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS	:= $(GENCODE_SM$(TARGET_GPU_ARCH))
NVCC_FLAGS = -lineinfo $(GENCODE_FLAGS) -std=c++17 --forward-unknown-to-host-compiler -x cu

$(EXEC): Makefile packing.cpp
	$(MPICXX) -ccbin=g++ $(NVCC_FLAGS) packing.cpp -o $(EXEC)

.PHONY.: clean sanitize run profile test
clean:
	rm -f $(EXEC) *.nsys-rep packing.*.compute-sanitizer.log

sanitize: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) compute-sanitizer --log-file $(OUTPUT_NAME).%q{SLURM_PROCID}.compute-sanitizer.log ./$(EXEC)

run: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) ./$(EXEC)

test: $(EXEC)
	mpiexec -n $(NP) ./$(EXEC) -num_iters 10

profile: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,cuda -o $(OUTPUT_NAME).%q{SLURM_PROCID} ./$(EXEC)
