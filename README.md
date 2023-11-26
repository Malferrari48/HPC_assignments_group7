# HPC OpenMP assignment Group 7

## Authors
Francesco Malferrari, Gianluca Siligardi and Andrea Somenzi

## Description
This is the exam project for UniMORE High Performance Computing course.
The goal is to minimize the execution time of the DynProg solver of the Polybench C library through:
* Profiling and bottleneck research
* Code rewriting
* Parallelization with
    * OpenMP
    * CUDA
    * HLS

## Folders in the repository
* `/dynprog` :
    original version of the file assigned
* `/fast_dynprog` :
    the fastest version of the written dynprog.
    It has been uploaded for completeness, however it has an error due to the [precision of floating point type](https://stackoverflow.com/questions/48088766/c-double-multiplication-have-different-result-when-order-of-variables-is-chang).
* `/OpenMP`

## OpenMP folder
This folder contains the sources of the first assignment:
* the rewritten DynProg `rew_dynprog.c` in `/rew_dynprog`
* the OpenMP parallelization of the rewritten DynProg `parallel_dynprog.c` in `/parallel_dynprog`
* the OpenMP parallelization over Nvidia Jetson Nano GPU of the rewritten DynProg `gpu_dynprog.c` in `/gpu_dynprog`

### Visuals
`/OpenMP` folder contains also the presentation of the results of the first assignment.

## How to compile
Into each directory run this command:
``` bash
make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```
You can also specify the dataset size adding the flag *DATASET* and writing one of the listed in the header file
`/dynprog/dynprog.h`, for example:
``` bash
make EXT_CFLAGS="-DLARGE_DATASET -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```
Alternatively, if you want to use datasets not present in the header file, you can replace the flag *DATASET* 
whith the following flags setted with numbers of your interest:
``` bash
-DLENGTH=10000 -DTSTEPS=20
```
Before running this commands, to compile `gpu_dynprog.c`, you must run
``` bash
module load clang/11.0.0 cuda/10.0
```
to load the cuda library.