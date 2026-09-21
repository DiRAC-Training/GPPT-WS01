all: nbody_cpu nbody_gpu

nbody_cpu: nbody.cpp util.hpp
	g++ -O3 -march=native nbody.cpp -o nbody_cpu

nbody_port: nbody_port.cu util.hpp
	nvcc -O3 nbody_port.cu -o nbody_port

nbody_gpu: nbody.cu util.hpp
	nvcc -O3 nbody.cu -o nbody_gpu

fmt:
	clang-format -i --style=llvm *.cpp *.cu *.hpp

clean:
	rm -f nbody_cpu nbody_gpu nbody_port *.csv *.png

.PHONY: all fmt
