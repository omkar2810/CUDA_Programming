// Vector Addition
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  if(i<len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  float *in1_d, *in2_d, *out_d;
  int total_size = inputLength*sizeof(float);
  cudaMalloc((void **) &in1_d, total_size);
  cudaMalloc((void **) &in2_d, total_size);
  cudaMalloc((void **) &out_d, total_size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(in1_d, hostInput1, total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(in2_d, hostInput2, total_size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(1.0*inputLength/256), 1, 1);
  dim3 DimBlock(256, 1, 1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid, DimBlock>>>(in1_d, in2_d, out_d, inputLength);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, out_d, total_size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(in1_d);
  cudaFree(in2_d);
  cudaFree(out_d);
  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
