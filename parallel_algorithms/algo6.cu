// prefix sum when array size > block size.

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void addPrev(float *input, float* output, float *aux, int len)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*BLOCK_SIZE*2 + tx;
  float aux_val = 0.0f;
  if(bx > 0)
    aux_val = aux[bx-1];
  if(idx < len)
  {
    output[idx] = input[idx] + aux_val;
  }
  if(idx + BLOCK_SIZE < len)
    output[idx+BLOCK_SIZE] = input[idx+BLOCK_SIZE] + aux_val;
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*BLOCK_SIZE*2 + tx;
  __shared__ float prefix[2*BLOCK_SIZE];
  if(idx < len){
    prefix[tx] = input[idx]; 
  }
  else{
    prefix[tx] = 0.0f;
  }
  if(idx + BLOCK_SIZE < len){
    prefix[tx + BLOCK_SIZE] = input[idx + BLOCK_SIZE]; 
  }
  else{
    prefix[tx + BLOCK_SIZE] = 0.0f;
  }

  // step1
  for(int stride = 1; stride <= BLOCK_SIZE; stride*=2)
  {
    __syncthreads();
    int index = (tx+1)*stride*2 - 1;
    if((index < 2*BLOCK_SIZE) && (index - stride >=0))
    {
      prefix[index] += prefix[index-stride];
    }
  }

  // step 2
  for(int stride = BLOCK_SIZE/2; stride >= 1; stride/=2)
  {
    __syncthreads();
    int index = (tx+1)*stride*2 - 1;
    if((index + stride) < 2*BLOCK_SIZE)
    {
      prefix[index + stride] += prefix[index];
    }
  }

  __syncthreads();

  if(idx < len){
    input[idx] = prefix[tx];
    // printf("Thread idx: %d, idx: %d, output: %d \n", tx, idx, output[idx]);
  }
  if(idx + BLOCK_SIZE < len){
    input[idx + BLOCK_SIZE] = prefix[tx + BLOCK_SIZE];
  }

  if(tx == 0 && output)
  {
    // Here we use the output parameter as the aux array
    output[bx] = prefix[2*BLOCK_SIZE-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float* deviceAux;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  int nblocks = ceil(1.0*numElements/(2*BLOCK_SIZE));
  cudaMalloc((void **)&deviceAux, nblocks*sizeof(float)); 
  //@@ Initialize the grid and block dimensions here
  dim3 gridDim(nblocks, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<gridDim, blockDim>>>(deviceInput, deviceAux, numElements);
  cudaDeviceSynchronize();

  scan<<<1, blockDim>>>(deviceAux, NULL, nblocks);
  cudaDeviceSynchronize();

  addPrev<<<gridDim, blockDim>>>(deviceInput, deviceOutput, deviceAux, numElements);
  cudaDeviceSynchronize();
  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  // for(int i=0; i<numElements; i++)
  //   wbLog(TRACE, i, " ", hostOutput[i]);

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

