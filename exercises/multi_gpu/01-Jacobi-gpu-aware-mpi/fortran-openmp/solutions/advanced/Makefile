# Original work Copyright (c) 2017-2024, NVIDIA CORPORATION.
# Modified work Copyright (c) 2026 Miren Radia
THIS_TASK := 01
OUTPUT_NAME := jacobi.$(THIS_TASK)__$(shell date '+%Y%m%d-%H%M')

NP ?= 4
SUBMIT_CMD ?= srun --gres=gpu:4 --ntasks-per-node 4 -p ampere -t 00:02:00 -A dirac-dt001-gpu

EXEC=jacobi.ex
MPIFC=mpifort
export OMPI_FC=nvfortran # Open MPI specific
TARGET_GPU_ARCH?=80
NVFC_FLAGS = -mp=gpu -gpu=cc$(TARGET_GPU_ARCH) -Mpreprocess
ifdef USE_DOUBLE
        NVFC_FLAGS += -DUSE_DOUBLE
endif

$(EXEC): Makefile jacobi.F90
	$(MPIFC) $(NVFC_FLAGS) jacobi.F90 -o $(EXEC)

.PHONY.: clean sanitize run profile test
clean:
	rm -f $(EXEC) *.o *.mod *.nsys-rep jacobi.*.compute-sanitizer.log

sanitize: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) compute-sanitizer --log-file $(OUTPUT_NAME).%q{SLURM_PROCID}.compute-sanitizer.log ./$(EXEC) -niter 10

run: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) ./$(EXEC)

test: $(EXEC)
	mpiexec -n $(NP) ./$(EXEC) -niter 10

profile: $(EXEC)
	$(SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,openmp,cuda -o $(OUTPUT_NAME).%q{SLURM_PROCID} ./$(EXEC) -niter 10
