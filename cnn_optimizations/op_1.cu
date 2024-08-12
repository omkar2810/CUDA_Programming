#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x_out = (blockIdx.y%width_tiles)*TILE_WIDTH + threadIdx.x;
    int y_out = (blockIdx.y/width_tiles)*TILE_WIDTH + threadIdx.y;
    int batch = blockIdx.z; 
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    extern __shared__  float tile[];
    int tile_size = TILE_WIDTH + 2*(K-1)/2;
    bool valid_thread = (x_out<Width) && (y_out < Height);
    for(int c=0; c<Channel; c++)
    {
        if(valid_thread)
            tile[c*(tile_size*tile_size) + ty*tile_size + tx] = in_4d(batch, c, y_out, x_out);
        else
            tile[c*(tile_size*tile_size) + ty*tile_size + tx] = 0.0f; 
    }
    __syncthreads();
    if (tx>=TILE_WIDTH||ty>=TILE_WIDTH)
        return;
    if(x_out>=Width_out || y_out >= Height_out)
        return;  
    // Insert your GPU convolution kernel code here
    float result = 0.0f;
    for(int c=0; c<Channel; c++)
    {
        for(int h=0; h<K; h++)
        {
            for(int w=0; w<K; w++)
            {
                // result += in_4d(batch, c, y_out+h, x_out+w)*mask_4d(m, c, h, w);
                // if(batch==0)
                // {
                //     float val1 = in_4d(batch, c, y_out+h, x_out+w);
                //     float val2 = tile[c*(tile_size*tile_size) + (ty+h)*tile_size + tx+w];
                //     if(val1!=val2)
                //         printf("tx: %d, ty: %d, c: %d, x_out: %d, y_out: %d, val1: %f, val2: %f \n", tx, ty, c, x_out, y_out, val1, val2);
                // }
                result += tile[c*(tile_size*tile_size) + (ty+h)*tile_size + tx+w]*mask_4d(m, c, h, w);

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
    int grid_z = Batch;
    int grid_y = width_tiles*height_tiles;
    int grid_x = Map_out;
    int radius = (K-1)/2;
    int block_size = TILE_WIDTH + 2*radius;
    int shared_memory_size = (Channel*block_size*block_size)*sizeof(float);
    dim3 gridDim(grid_x, grid_y, grid_z);
    dim3 blockDim(block_size, block_size, 1);
    conv_forward_kernel<<<gridDim, blockDim, shared_memory_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

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