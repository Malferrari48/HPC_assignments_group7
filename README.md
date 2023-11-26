# HPC OpenMP assignment Group 7

## Authors
Francesco Malferrari, Gianluca Siligardi and Andrea Somenzi

## Description
This is the exam project for UniMORE High Performance Computing course.
The goal is to minimize the execution time of the DynProg solver of the Polybench C library through:
* Profiling and bottleneck research;
* Code rewriting;
* Parallelization with
    * OpenMP
    * CUDA
    * HLS

## Folders in the repository
* /dynprog :
    riginal version of the file assigned;
* /fast_dynprog :
    the fastest version of the written dynprog.
    It has been uploaded for completeness, however it has an error due to the [precision of floating point type](https://stackoverflow.com/questions/48088766/c-double-multiplication-have-different-result-when-order-of-variables-is-chang).
* /OpenMP

## OpenMP folder


## How to compile
* To compile the accelerator file you have to write this command:
```
make EXERCISE=dynprog_GPU.c EXT_CFLAGS="-DEXTRALARGE_DATASET -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```
where in *EXERCISE* you put the file to compile and in *DATASET* you can put the datasets available in the header file.
Remember to write before
```
module load clang/11.0.0 cuda/10.0
```
to load the cuda library.

* To compile the other files you have to write this command:
```
make EXT_CFLAGS="-DNTHREADS=4 -DEXTRALARGE_DATASET -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```
where you can change the number of threads in *THREADS* and the dataset in *DATASET*.

* Alternatively, you should use datasets not present in the header file replacing the flag *DATASET* whith the following:
```
-DLENGTH=10000 -DTSTEPS=20
```