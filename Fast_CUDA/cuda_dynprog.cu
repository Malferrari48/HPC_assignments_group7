#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* Include polybench common header. */
#include "../PolybenchC/polybench.h"

/* Include benchmark-specific header. */
/* Default data type is int, default size is 50. */
#include "../dynprog/dynprog.h"

__device__ void warpReduce(volatile double* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

static void print_array(DATA_TYPE out)
{
  fprintf(stderr, DATA_PRINTF_MODIFIER, out);
  fprintf(stderr, "\n");
}

// Funzione per sommare gli elementi di un array in modo parallelo (riduzione)
__global__ void reduceSum(double *input, double *output, int size) {
  extern __shared__ double sharedData[];

  int tid = threadIdx.x;
  int gid = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  // Carica i dati nell'array condiviso
  if (gid < size) {
    sharedData[tid] = input[gid] + input[gid+blockDim.x];
  }
  else {
    sharedData[tid] = 0;
  }

  __syncthreads();

  // Esegui la riduzione nella memoria condivisa
  for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
    if (tid < offset) {
      sharedData[tid] += sharedData[tid + offset];
    }
    __syncthreads();
  }

  if (tid < 32) 
    warpReduce(sharedData, tid);

  // Il thread 0 di ogni blocco scrive il risultato parziale nel blocco corrispondente dell'array di output
  if (tid == 0) { /*
    for (int i = 0; i < 4; ++i) {
      printf("%f ", input[i]);
      printf("\n");
    } */
    output[blockIdx.x] = sharedData[0];
    /* printf("Output: %f\n", output[blockIdx.x]); */
  }
}

__global__ void kernel2(double *c, double *part_sum, double *W, int i)
{
  int thid = threadIdx.x;
  
  if(thid == 0) {
    c[i] = part_sum[0] + W[i];
  }
}

int main() {
  /* Retrieve problem size. */
  int length = LENGTH;
  int tsteps = TSTEPS;

  const int blockSize = 256;
  int numBlocks = (length + blockSize - 1) / blockSize;

  /* Variable declaration/allocation. */
  DATA_TYPE *h_out_l;
  //DATA_TYPE h_sum_c;
  //POLYBENCH_1D_ARRAY_DECL(h_c, DATA_TYPE, LENGTH, length);
  //POLYBENCH_1D_ARRAY_DECL(h_W, DATA_TYPE, LENGTH, length);
  //DATA_TYPE h_c[length];
  //DATA_TYPE h_W[length];
  
  DATA_TYPE *h_c = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * length);
  DATA_TYPE *h_W = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * length);
  
  /* Initialize array(s). */
  for (int i = 0; i < length; ++i) 
  {
    h_c[i] = 0.0f;
    h_W[i] = ((DATA_TYPE) -i) / length;
  }
  
  /* Start timer. */
  polybench_start_instruments;

  // Alloca spazio per l'array su GPU
  DATA_TYPE *d_c, *d_out_l, *d_W;
  cudaMalloc((void**)&d_c, length * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_out_l, numBlocks * sizeof(DATA_TYPE));
  cudaMalloc((void**)&d_W, length * sizeof(DATA_TYPE));

  // Copia l'array da CPU a GPU
  cudaMemcpy(d_c, h_c, length * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W, length * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

  for(int i = 1; i <= length; i++) 
  {
    int numBlocks = (i + blockSize - 1) / blockSize;
    // Esegui la riduzione
    reduceSum<<<numBlocks, blockSize, blockSize * sizeof(DATA_TYPE)>>>(d_c, d_out_l, i);
    kernel2<<<1, 1>>>(d_c, d_out_l, d_W, i);
  }
  
  // Alloca spazio per il risultato finale su CPU
  h_out_l = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * numBlocks);

  // Copia il risultato da GPUresultU
  cudaMemcpy(h_c, d_c, length * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
  
  DATA_TYPE out = h_c[length - 1];
  
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  
  /* Prevent dead-code elimination. All live-out_l data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(out));

  /* Be clean. */
  //POLYBENCH_FREE_ARRAY(h_c);
  //POLYBENCH_FREE_ARRAY(h_W);

  // Deallocazione della memoria
  free(h_out_l);
  free(h_W);
  free(h_c);
  cudaFree(d_W);
  cudaFree(d_out_l);
  cudaFree(d_c);

  return 0;
}