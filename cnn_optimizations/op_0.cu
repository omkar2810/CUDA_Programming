#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, int batch_offset)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int width_tiles = ceil((1.0*Width_out)/TILE_WIDTH);
    int m = blockIdx.x;
    int x_out = (blockIdx.y%width_tiles)*TILE_WIDTH + threadIdx.x;
    int y_out = (blockIdx.y/width_tiles)*TILE_WIDTH + threadIdx.y;
    int batch = blockIdx.z + batch_offset;
    if(x_out>=Width_out || y_out >= Height_out)
        return;   
    if(batch>=Batch)
      return;
    
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    float result = 0.0f;
    for(int c=0; c<Channel; c++)
    {
        for(int h=0; h<K; h++)
        {
            for(int w=0; w<K; w++)
            {
                result += in_4d(batch, c, y_out+h, x_out+w)*mask_4d(m, c, h, w);
            }
        }
    }
    out_4d(batch, m, y_out, x_out) = result;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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
    const int num_streams = 4;
    int mini_batch_size = 250;
    int mini_batch_per_stream = ceil(Batch*1.0/num_streams*mini_batch_size);
    float * host_output_temp = (float *) host_output;

    cudaHostRegister((void *)host_input, in_size*sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((void *)host_output_temp, out_size*sizeof(float), cudaHostRegisterDefault);
    
    cudaStream_t streams[num_streams];
    for(int i=0; i<num_streams; i++)
      cudaStreamCreate(&streams[i]);

    
    cudaMalloc((void **)device_input_ptr, in_size*sizeof(float));
    cudaMalloc((void **)device_output_ptr, out_size*sizeof(float));

    cudaMalloc((void **)device_mask_ptr, mask_size*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size*sizeof(float), cudaMemcpyHostToDevice);  


    int width_tiles = ceil((1.0*Width_out)/TILE_WIDTH);
    int height_tiles = ceil((1.0*Height_out)/TILE_WIDTH);
    int grid_z = mini_batch_size;
    int grid_y = width_tiles*height_tiles;
    int grid_x = Map_out;
    dim3 gridDim(grid_x, grid_y, grid_z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    int start_idx = 0;
    for(int mini_batch_idx=0; mini_batch_idx < mini_batch_per_stream; mini_batch_idx++)
    {
      for(int stream_idx=0; stream_idx < num_streams; stream_idx++)
      {
        if(start_idx>=Batch)
          break;
        // printf("%d %d\n", start_idx, stream_idx);
        int mem_start_input = start_idx*Channel*Height*Width;
        int mem_start_output = start_idx*Map_out*Height_out*Width_out;
        int copy_size = min(mini_batch_size, Batch-start_idx);
        int mini_batch_in_size = copy_size*Channel*Height*Width;
        int mini_batch_out_size = copy_size*Map_out*Height_out*Width_out;

        cudaMemcpyAsync(*device_input_ptr+mem_start_input, &host_input[mem_start_input], mini_batch_in_size*sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]);
        conv_forward_kernel<<<gridDim, blockDim, 0, streams[stream_idx]>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K, start_idx);
        cudaMemcpyAsync(&host_output_temp[mem_start_output], *device_output_ptr+mem_start_output, mini_batch_out_size*sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]); 
        start_idx += mini_batch_size;
      }
    }
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);
    // cudaMemcpy(*device_output_ptr, host_output, out_size*sizeof(float), cudaMemcpyHostToDevice);    
    cudaHostUnregister((void *)host_input);  
    cudaHostUnregister((void *)host_output_temp);  

    for(int i=0; i<num_streams; i++)
      cudaStreamDestroy(streams[i]);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
   return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  return;
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