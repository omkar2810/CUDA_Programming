// Basic Matrix Multiplication

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  if((row<numCRows)&&(col<numCColumns))
  {
    float pval = 0;
    for(int k=0; k<numAColumns; k++)
    {
      // A[row, k]*B[k,col]
      pval += A[row*numAColumns+k]*B[k*numBColumns+col];
    }
    C[row*numCColumns+col] = pval;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  int a_size = numARows*numAColumns;
  int b_size = numBRows*numBColumns;
  int c_size = numCRows*numCColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(c_size*sizeof(float));

  //@@ Allocate GPU memory here
  float *a_d, *b_d, *c_d;
  cudaMalloc((void **) &a_d, a_size*sizeof(float));
  cudaMalloc((void **) &b_d, b_size*sizeof(float));
  cudaMalloc((void **) &c_d, c_size*sizeof(float));
  // //@@ Copy memory to the GPU here
  cudaMemcpy(a_d, hostA, a_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, hostB, b_size*sizeof(float), cudaMemcpyHostToDevice);

  // //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(1.0*numCColumns/32), ceil(1.0*numCRows/32), 1);
  dim3 DimBlock(32, 32, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(a_d, b_d, c_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, c_d, c_size*sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

