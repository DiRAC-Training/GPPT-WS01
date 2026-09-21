CXX = nvc++

HEADERS = $(wildcard include/*.hpp)

BLOCK_SIZE ?= 128
EXTRA_FLAGS="-DBLOCK_SIZE=$(BLOCK_SIZE)"

.PHONY: cpu cuda cuda_optimised cublas cublas_sym clean fmt all solutions

all: cpu fixme
solutions: cpu cuda cuda_optimised cublas cublas_sym

# Regular C++ .cpp compilation
build/%.o: src/%.cpp $(HEADERS) | build
	$(CXX) -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"$(CUDA_PATH)/include" $(EXTRA_FLAGS) -c $< -o $@

# CPU version
cpu: build/main.o build/test.o build/solver.o | build
	$(CXX) -fopenmp $^ -o build/$@

# .cu compilation
build/%.o: src/%.cu $(HEADERS) | build
	nvcc -O3 -arch=native -Iinclude/ $(EXTRA_FLAGS) -c $< -o $@

# The learner's version!
fixme: build/solver_FIXME.o build/main.o build/solution/test.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

## Solutions

# Redefine compilation rules just for solution files (there is a more elegant way to do this but we're keeping this simple)
build/solution/%.o: solution/%.cpp $(HEADERS) | build
	$(CXX) -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"$(CUDA_PATH)/include" $(EXTRA_FLAGS) -c $< -o $@
build/solution/%.o: solution/%.cu $(HEADERS) | build
	nvcc -O3 -arch=native -Iinclude/ $(EXTRA_FLAGS) -c $< -o $@

# These could be made more maintainable but let's keep it simple!
cuda: build/solution/solver_cuda.o build/solution/main.o build/solution/test.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cuda_optimised: build/solution/solver_cuda_opt.o build/solution/main.o build/solution/test.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cublas: build/solution/solver_cublas.o build/solution/main.o build/solution/test.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

cublas_sym: build/solution/solver_cublas_sym.o build/solution/main.o build/solution/test.o | build
	$(CXX) -L"$(CUDA_PATH)/lib" -lcublas -lcudart $^ -o build/$@

## Utility

fmt:
	clang-format --style=llvm -i src/*.cpp src/*.cu solution/*.cpp solution/*.cu $(HEADERS)

clean:
	rm -rf build

build: build/solution
	mkdir -p build

build/solution:
	mkdir -p build/solution
