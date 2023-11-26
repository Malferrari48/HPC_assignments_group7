# HPC OpenMP assignment Group 7

## Authors
Francesco Malferrari, Gianluca Siligardi and Andrea Somenzi

## Files in the repository
* dynprog_original.c --> Original file
* dynprog_rewritten.c --> File rewritten and optimized by us
* dynprog_OpenMP --> File optimized using OpenMP
* dynprog_accelerator --> File optimized using accelerator
* Makefile

## How to compile
To compile the accelerator file you have to write this command:
```
make EXERCISE=dynprog_GPU.c EXT_CFLAGS="-DEXTRALARGE_DATASET -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```
where in *EXERCISE* you put the file to compile and in *DATASET* you can put the datasets available in the header file.
Remember to write before
```
module load clang/11.0.0 cuda/10.0
```
to load the cuda library.

To compile the other files you have to write this command:
```
make EXT_CFLAGS="-DNTHREADS=4 -DEXTRALARGE_DATASET -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS" clean all run
```
where you can change the number of threads in *THREADS* and the dataset in *DATASET*.

