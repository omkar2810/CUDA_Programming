// 3D Convolution using shared memory tiling.

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
const int TILE_WIDTH = 8;
const int RADIUS = 1;
const int MASK_WIDTH = 3;

//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_out = blockIdx.x*TILE_WIDTH + tx;
  int y_out = blockIdx.y*TILE_WIDTH + ty;
  int z_out = blockIdx.z*TILE_WIDTH + tz;

  int x_in = x_out-RADIUS;
  int y_in = y_out-RADIUS;
  int z_in = z_out-RADIUS;
  const int block_width = TILE_WIDTH + 2*RADIUS;

  __shared__ float tile[block_width][block_width][block_width];

  if((x_in>=0 && x_in<x_size)&&(y_in>=0 && y_in<y_size)&&(z_in>=0 && z_in<z_size))
  {
    tile[tx][ty][tz] = input[z_in*(x_size*y_size) + y_in*x_size + x_in];
  }
  else
  {
    tile[tx][ty][tz] = 0.0f;
  }
  __syncthreads();


  if((tx<TILE_WIDTH) && (ty<TILE_WIDTH) && (tz<TILE_WIDTH))
  {
    float result = 0.0f;
    for(int i=0; i<MASK_WIDTH;i++)
    {
      for(int j=0; j<MASK_WIDTH; j++)
      {
        for(int k=0; k<MASK_WIDTH; k++)
        {
          result += Mc[i][j][k]*tile[tx+i][ty+j][tz+k];
        }
      }
    }
    if((x_out<x_size)&&(y_out<y_size)&&(z_out<z_size))
    {
      output[z_out*(x_size*y_size) + y_out*x_size + x_out] = result;
    }
  }

  }

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  float * input_d, *output_d;
  cudaMalloc((void **) &input_d, (inputLength-3)*sizeof(float));
  cudaMalloc((void **) &output_d, (inputLength-3)*sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength*sizeof(float));
  cudaMemcpy(input_d, &hostInput[3], (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(1.0*x_size/TILE_WIDTH), ceil(1.0*y_size/TILE_WIDTH), ceil(1.0*z_size/TILE_WIDTH));
  dim3 DimBlock(TILE_WIDTH+2*RADIUS, TILE_WIDTH+2*RADIUS, TILE_WIDTH+2*RADIUS);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(input_d, output_d, z_size, y_size, x_size);
  cudaDeviceSynchronize();


  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)

  cudaMemcpy(&hostOutput[3], output_d, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);


  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

//   for(int i=0; i<10; i++)
//     wbLog(TRACE, hostOutput[i]);

  //@@ Free device memory
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(Mc);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

