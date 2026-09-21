THIS_TASK := 02
OUTPUT_NAME := packing.$(THIS_TASK)__$(shell date '+%Y%m%d-%H%M')

NP ?= 4
SUBMIT_CMD ?= srun --gres=gpu:4 --ntasks-per-node 4 -p ampere -t 00:02:00 -A dirac-dt001-gpu

EXEC=packing.ex
MPIFC=mpifort
export OMPI_FC=nvfortran # Open MPI specific
TARGET_GPU_ARCH?=80
NVFC_FLAGS = -mp=gpu -gpu=cc$(TARGET_GPU_ARCH) -Mpreprocess
ifdef USE_DOUBLE
        NVFC_FLAGS += -DUSE_DOUBLE
endif

$(EXEC): Makefile packing.F90
	$(MPIFC) $(NVFC_FLAGS) packing.F90 -o $(EXEC)

.PHONY.: clean sanitize run profile test
clean:
	rm -f $(EXEC) *.o *.mod *.nsys-rep packing.*.compute-sanitizer.log

sanitize: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) compute-sanitizer --log-file $(OUTPUT_NAME).%q{SLURM_PROCID}.compute-sanitizer.log ./$(EXEC)

run: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) ./$(EXEC)

test: $(EXEC)
	mpiexec -n $(NP) ./$(EXEC) -num_iters 10

profile: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,openmp,cuda -o $(OUTPUT_NAME).%q{SLURM_PROCID} ./$(EXEC) -num_iters 10
