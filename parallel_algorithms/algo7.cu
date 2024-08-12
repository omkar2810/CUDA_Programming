// Histogram Equalization for images

#include <wb.h>

#define HISTOGRAM_LENGTH 256
//@@ insert code here
#define BLOCK_SIZE 512
#define uc unsigned char
__global__ void convert_to_char(float* input, uc* output, int len)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*blockDim.x*3 + 3*tx;
  if(idx<len)
  {
    output[idx] = (uc)(255*input[idx]);
    output[idx+1] = (uc)(255*input[idx+1]);
    output[idx+2] = (uc)(255*input[idx+2]);
  }

}

__global__ void convert_to_float(uc* input, float* output, int len)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*blockDim.x*3 + 3*tx;
  if(idx<len)
  {
    output[idx] = (float)(input[idx]/255.0);
    output[idx+1] = (float)(input[idx+1]/255.0);
    output[idx+2] = (float)(input[idx+2]/255.0);
  }

}

__global__ void convert_to_grayscale(uc* input, uc* output, int len)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*blockDim.x*3 + 3*tx;
  int output_idx = idx/3;
  if(idx<len)
  {
    uc r = input[idx];
    uc g = input[idx+1];
    uc b = input[idx+2];
    uc grayscale_val = (uc)(0.21*r + 0.71*g + 0.07*b);
    output[output_idx] = grayscale_val; 
  }
}

__global__ void create_historgram(uc* input, unsigned int* histogram, int len)
{
  __shared__ unsigned int private_histogram[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*blockDim.x + tx;
  if (idx >= len)
    return;

  if(tx < HISTOGRAM_LENGTH)
    private_histogram[tx] = 0;

  __syncthreads();
  unsigned int val = (unsigned int)input[idx];

  atomicAdd(&(private_histogram[val]), 1); 

  __syncthreads();

  if(tx < HISTOGRAM_LENGTH)
    atomicAdd(&(histogram[tx]), private_histogram[tx]);

  // if(tx<HISTOGRAM_LENGTH && bx < 10)
  //   printf("%d:%f, ", tx, histogram[tx]);

}

__global__ void scan(unsigned int *input, float* output, int image_size) {
  const int block_size = HISTOGRAM_LENGTH/2;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*block_size*2 + tx;

  __shared__ float prefix[2*block_size];
  
  prefix[tx] = (input[idx]*1.0)/image_size; 
  prefix[tx + block_size] = (input[idx + block_size]*1.0)/image_size; 

  // step1
  for(int stride = 1; stride <= block_size; stride*=2)
  {
    __syncthreads();
    int index = (tx+1)*stride*2 - 1;
    if((index < 2*block_size) && (index - stride >=0))
    {
      prefix[index] += prefix[index-stride];
    }
  }

  // step 2
  for(int stride = block_size/2; stride >= 1; stride/=2)
  {
    __syncthreads();
    int index = (tx+1)*stride*2 - 1;
    if((index + stride) < 2*block_size)
    {
      prefix[index + stride] += prefix[index];
    }
  }

  __syncthreads();

  output[idx] = prefix[tx];
  output[idx + block_size] = prefix[tx + block_size];
}

__global__ void equalize(uc *input, float* cdf, int len, float cdfmin)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*blockDim.x*3 + 3*tx;
  __shared__ float private_histogram[HISTOGRAM_LENGTH];
  
  if(tx<256)
    private_histogram[tx] = cdf[tx];
  __syncthreads();

  if(idx>=len)
    return;

  for(int i=0;i<3;i++)
  {
    int temp_idx = idx + i;
    int pixel_val = input[temp_idx];
    float x = (float) 255.0*(private_histogram[pixel_val] - cdfmin)/(1.0 - cdfmin);

    float clamp_val = min(max(x, 0.0), 255.0);
    input[temp_idx] = (uc) clamp_val;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  // Read Image
  int total_size = imageHeight*imageWidth*imageChannels;
  int grayscale_size = total_size/3;
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(total_size*sizeof(float));

  //@@ insert code here
  float *deviceInput;
  float * deviceOutput;
  uc *charImage;
  uc *grayScaleImage;
  unsigned int* histogram;
  float* cdf;
 
  cudaMalloc((void **)&deviceInput, total_size*sizeof(float));
  cudaMalloc((void **)&deviceOutput, total_size*sizeof(float));

  // Convert float to char
  cudaMalloc((void **)&charImage, total_size*sizeof(uc));
  cudaMemcpy(deviceInput, hostInputImageData, total_size*sizeof(float), cudaMemcpyHostToDevice);
  dim3 gridDim(ceil(total_size*1.0/(BLOCK_SIZE*3)), 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);
  convert_to_char<<<gridDim, blockDim>>>(deviceInput, charImage, total_size);
  cudaDeviceSynchronize();
  // --- debug
  // unsigned char* cpu_charImage = (unsigned char*)malloc(total_size*sizeof(uc));
  // cudaMemcpy(cpu_charImage, charImage, total_size*sizeof(uc), cudaMemcpyDeviceToHost);
  // for(int i=45200; i<45300;i++)
  // {
  //   uc uchardata = (uc)(hostInputImageData[i]*255.0);
  //   printf("%f %d %d\n", hostInputImageData[i], uchardata, cpu_charImage[i]);
  
  // }
    
  
  // Convert rgb to grayscale
  cudaMalloc((void **)&grayScaleImage, grayscale_size*sizeof(uc));
  convert_to_grayscale<<<gridDim, blockDim>>>(charImage, grayScaleImage, total_size);
  cudaDeviceSynchronize();
  // --- debug
  // unsigned char* cpu_gsImage = (unsigned char*)malloc(total_size*sizeof(uc));
  // cudaMemcpy(cpu_gsImage, grayScaleImage, grayscale_size*sizeof(uc), cudaMemcpyDeviceToHost);
  // int total_zero = 0;
  // for(int i=0; i<grayscale_size;i++)
  // {
  //   int r = cpu_charImage[3*i];
  //   int g = cpu_charImage[3*i+1];
  //   int b = cpu_charImage[3*i+2];
  //   int gs =(uc)(0.21*r + 0.71*g + 0.07*b);
  //   if(gs==0)
  //     printf("zero val%d", i);
  //   if(cpu_gsImage==0)
  //     total_zero +=1;
  //   if(gs!=cpu_gsImage[i])
  //     printf("error %d %d %d %d %d\n", r, g, b, gs, cpu_gsImage[i]);
  // }


  //Create Histogram
  cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  create_historgram<<<gridDim, blockDim>>>(grayScaleImage, histogram, grayscale_size);
  cudaDeviceSynchronize();
  // --- debug
  // float* cpu_histogram = (float *)malloc(HISTOGRAM_LENGTH*sizeof(float));
  // for(int i=0;i<HISTOGRAM_LENGTH ;i++)
  //   cpu_histogram[i] = 0.0f;
  // for(int i=0;i<grayscale_size;i++)
  //   cpu_histogram[cpu_gsImage[i]] += 1;
  // unsigned int* cpu_histogram_temp  = (unsigned int *)malloc(HISTOGRAM_LENGTH*sizeof(unsigned int));
  // cudaMemcpy(cpu_histogram_temp, histogram, HISTOGRAM_LENGTH*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  // for(int i=0; i<HISTOGRAM_LENGTH; i++)
  //   printf("%d:%f, %d\n", i, cpu_histogram_temp[i], cpu_histogram[i]);
  // printf("\n");


  // Compute Histrogram CDF
  cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH*sizeof(float));
  scan<<<1, HISTOGRAM_LENGTH/2>>>(histogram, cdf, grayscale_size);
  cudaDeviceSynchronize();
  float* cpu_cdf = (float *)malloc(HISTOGRAM_LENGTH*sizeof(float));
  cudaMemcpy(cpu_cdf, cdf, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyDeviceToHost);
  // --- debug

  // for(int i=0; i<HISTOGRAM_LENGTH; i++)
  //   printf("%d:%f, ", i, cpu_cdf[i]);
  // printf("\n");
    
  // Equalize Image
  float cdfmin = cpu_cdf[0];
  equalize<<<gridDim, blockDim>>>(charImage, cdf, total_size, cdfmin);
  cudaDeviceSynchronize();

  // unsigned char* cpu_eqImage = (unsigned char*)malloc(total_size*sizeof(uc));
  // cudaMemcpy(cpu_eqImage, charImage, total_size*sizeof(uc), cudaMemcpyDeviceToHost);
  // for(int i=0; i<100;i++)
  // {
  //   printf("%d %f %d\n", cpu_charImage[i], cpu_cdf[cpu_charImage[i]], cpu_eqImage[i]);
  // }

  // Convert to float
  convert_to_float<<<gridDim, blockDim>>>(charImage, deviceOutput, total_size);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutput, total_size*sizeof(float), cudaMemcpyDeviceToHost);

  
  wbImage_setData(outputImage, hostOutputImageData);


  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(charImage);
  cudaFree(grayScaleImage);
  cudaFree(histogram);
  cudaFree(cdf);

  free(hostInputImageData);
  free(hostOutputImageData);
  free(cpu_cdf);
  return 0;
}

