// prefix sum reduction using brent-kung algorithm

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the correct index
  __shared__ float prefix[2*BLOCK_SIZE];
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int idx = bx*BLOCK_SIZE*2 + tx;
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
  for(int stride=BLOCK_SIZE; stride>=1; stride/=2)
  {
    __syncthreads();
    if(tx<stride){
      prefix[tx] += prefix[tx+stride];
    }
  }
  
  if(tx==0){
    output[bx] = prefix[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  float *in_d, *out_d;
  cudaMalloc((void **) &in_d, numInputElements*sizeof(float));
  cudaMalloc((void **) &out_d, numOutputElements*sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(in_d, hostInput, numInputElements*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Copy input memory to the GPU


  //@@ Initialize the grid and block dimensions here
  dim3 gridDim(ceil(1.0*numInputElements/(2*BLOCK_SIZE)), 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  //@@ Launch the GPU Kernel and perform CUDA computation
  total<<<gridDim, blockDim>>>(in_d, out_d, numInputElements);
  
  cudaDeviceSynchronize();  
  //@@ Copy the GPU output memory back to the CPU
  cudaMemcpy(hostOutput, out_d, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);
  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  //@@ Free the GPU memory
  cudaFree(in_d);
  cudaFree(out_d);

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

