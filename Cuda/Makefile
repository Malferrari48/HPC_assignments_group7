ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

ifndef EXERCISE
EXERCISE=cuda_dynprog.cu
endif

BUILD_DIR ?= ./

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

OPT:=-O3 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math `pkg-config --cflags --libs opencv4`

CXXFLAGS:=$(OPT) -I. $(EXT_CXXFLAGS)
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp

SRCS:= ../PolybenchC/polybench.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(EXERCISE:%=$(BUILD_DIR)/%.o)
EXE=$(EXERCISE:.cu=.exe)

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo LD_LIBRARY_PATH=$(CUDA_HOME)/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} $(CUDA_HOME)/bin/nvprof ./$(EXE)

clean:
	-rm -f $(BUILD_DIR)/*.exe $(BUILD_DIR)/*.o

MKDIR_P ?= mkdir -p
