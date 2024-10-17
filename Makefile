TARGETS = saxpy hello add add_2d

TARGETS_SRC=$(addprefix ./src/,$(notdir $(TARGETS)))
TARGETS_BIN=$(addprefix ./build/,$(notdir $(TARGETS)))
NVCC = /usr/local/cuda-12.6/bin/nvcc -gencode arch=compute_89,code=sm_89 -g 
all: build $(TARGETS_BIN)
build: 
	mkdir -p build

clean: 
	rm -rf build

build/%: src/%.cu
	$(NVCC) -o $@ $<
#	# $@ => file name as in %
#	# $< => dependency as in %.cu

TARGET_BIN = ./build/add_2d
TARGET_ITER = 1
TARGET_CMD = ${TARGET_BIN} 
PROF_OPT :=
null :=
space := ${null} ${null}
PROF_OPT += dram__bytes_read.sum
PROF_OPT +=,dram__bytes_read.sum.per_second
PROF_OPT +=,dram__bytes_write.sum
PROF_OPT +=,dram__bytes_write.sum.per_second
PROF_OPTION =$(subst ${space},${null},$(PROF_OPT))
#PROF_OPT = ${PROF_OPT},
#PROF_OPT = ${PROF_OPT},

nsys: $(TARGET_BIN)
	nsys profile -o ./${TARGET_BIN}.${TARGET_ITER}.prof --cuda-graph-trace=node --cudabacktrace=all --force-overwrite=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --gpu-metrics-devices=cuda-visible --gpu-metrics-set=ad10x-gfxt ${TARGET_CMD}

ncu: $(TARGET_BIN)
	ncu -f --set full --target-processes all -o ./${TARGET_BIN}.${TARGET_ITER}.prof ${TARGET_CMD}
ncu-prof: $(TARGET_BIN)
	ncu --print-summary per-gpu --metrics ${PROF_OPTION} ${TARGET_CMD}
