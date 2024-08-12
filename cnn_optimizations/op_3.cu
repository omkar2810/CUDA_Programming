#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16


__global__ void unrollAndMatrixMultiplyShared(const float *input_kernel, const float *input_images, float *output_images,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int Batch, int K, int Channel, int Height, int Width) {
  #define in_4d(i3, i2, i1, i0) input_images[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH]; 

  int global_batch = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int width_tiles = ceil((1.0*numCColumns)/TILE_WIDTH);
  int row = blockIdx.y*blockDim.y + ty;
  int col = blockIdx.x*blockDim.x + tx; 
  float pval = 0.0f;
  for (int q = 0; q < (numAColumns-1)/TILE_WIDTH + 1; q++) 
  { 
    if ((row < numARows) && (q*TILE_WIDTH+tx < numAColumns))
      subTileA[ty][tx] = input_kernel[row*numAColumns + q*TILE_WIDTH+tx];
    else
      subTileA[ty][tx] = 0.0f;
    
    if ((col < numBColumns) && (q*TILE_WIDTH+ty < numBRows))
    {
      int h_unroll = (q*TILE_WIDTH+ty); // 6
      int w_unroll = col; // 1
      int channel = h_unroll/(K*K); // 1
      int Width_out = Width - K + 1;
      int h = w_unroll/Width_out; // 0
      int w = w_unroll%Width_out; // 1
      int p = (h_unroll - channel*K*K)/K; // (6-4)/2 = 1  
      int _q = (h_unroll - channel*K*K)%K; // (6-4)%2 = 0
      subTileB[ty][tx] = in_4d(global_batch, channel, h+p, w+_q);
    }
    else
      subTileB[ty][tx] = 0.0f;

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
      pval += subTileA[ty][k] * subTileB[k][tx];
    __syncthreads();

  }

  if ((row<numCRows)&&(col<numCColumns))
  {
    output_images[global_batch*(numCRows*numCColumns) + row*numCColumns+col] = pval;
  }

  #undef in_4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int in_size = Batch*Channel*Height*Width;
    int out_size = Batch*Map_out*Height_out*Width_out;
    int mask_size = Map_out*Channel*K*K;

    cudaMalloc((void **)device_input_ptr, in_size*sizeof(float));
    cudaMalloc((void **)device_output_ptr, out_size*sizeof(float));
    cudaMalloc((void **)device_mask_ptr, mask_size*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, in_size*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_output_ptr, host_output, out_size*sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size*sizeof(float), cudaMemcpyHostToDevice);    

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int width_tiles = ceil((1.0*Width_out)/TILE_WIDTH);
    int height_tiles = ceil((1.0*Height_out)/TILE_WIDTH);

    int Width_unroll = Height_out*Width_out;
    int Height_unroll = K*K*Channel;
    int Height_kernel = Map_out;
    int Width_kernel = K*K*Channel;

    int Height_output = Height_kernel;
    int Width_output = Width_unroll;

    int unroll_size = Batch*Height_unroll*Width_unroll;

    dim3 gridDim(ceil(1.0*Width_output/TILE_WIDTH), ceil(1.0*Height_output/TILE_WIDTH), Batch);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    unrollAndMatrixMultiplyShared<<<gridDim, blockDim>>>(device_mask, device_input, device_output, Height_kernel, Width_kernel, Height_unroll, Width_unroll, Height_output, Width_output, Batch, K, Channel, Height, Width);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int out_size = Batch*Map_out*Height_out*Width_out;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, out_size*sizeof(float), cudaMemcpyDeviceToHost);    

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}