all: nbody

ifdef BLOCK_SIZE
EXTRA_FLAGS="-DBLOCK_SIZE=$(BLOCK_SIZE)"
endif

nbody: nbody.cu util.hpp
	nvcc -O3 -lineinfo -g $(EXTRA_FLAGS) -o nbody nbody.cu

nbody_tiled: nbody_tiled.cu util.hpp
	nvcc -O3 -lineinfo -g $(EXTRA_FLAGS) -o nbody_tiled nbody_tiled.cu

fmt:
	clang-format -i --style=llvm *.cu

clean:
	rm -f nbody nbody_tiled *.csv *.png

.PHONY: all fmt clean
