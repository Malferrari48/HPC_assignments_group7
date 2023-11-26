CC=clang
LD=ld
OBJDUMP=objdump

OPT=-O3 -g
OMP=-fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
CFLAGS=$(OPT) $(OMP) -I. $(EXT_CFLAGS) --cuda-path=/usr/local/cuda-10
LDFLAGS=-lm $(EXT_LDFLAGS)

SRCS=../../PolybenchC/polybench.c
SRCS += $(wildcard *.c)
EXE=$(basename $(SRCS)).exe
HEADERS = $(wildcard *.h)

.PHONY: all exe clean

all : exe

exe : $(EXE)

$(EXE) : $(SRCS)
	$(CC) $(CFLAGS) $(INCPATHS) $^ -o $@ $(LDFLAGS)

clean :
	-rm -vf -vf $(EXE) *~

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} /usr/local/cuda/bin/nvprof ./$(EXE) $(EXT_ARGS)