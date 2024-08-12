## CUDA Programming

### GPU Alogirthms

The folder `gpu_algorithm` contains the CUDA implementation for various popular parallel algorithms.

- `algo1.cu` - vector addition.
- `algo2.cu` - generalized matrix-matrix multiplication.
- `algo3.cu` - GEMM with shared memory tiling.
- `algo4.cu` - 3D convolution with shared memory tiling.
- `algo5.cu` - Brent-Kung algorithm for prefix sum.
- `algo6.cu` - Heirarchical prefix sum.
- `algo7.cu` - Histogram equalization for images.
- `algo8.cu` - Sparse matrix multiplication using Jagged Diagonal Storage (JDS) representation. 
 
### CNN Optimizations
The folder `cnn_optmizations` an experimental project to evaluate the effectivness of various kernel and host optmizations to improve the efficiency of the convolution layers of a CNN.
The complete code for the CNN is not included, and the files contain the optimized convolution forward function.

- `op_0.cu` - Streams optmization to improve CPU-GPU overlap. 
- `op_1.cu` - Shared memory tiling to reduce global memory look-ups.
- `op_2.cu` -  Converting convolutions into a matrix multiplication using unrolling.
- `op_3.cu` - Implementing a single step unrolling and matrix multiplication algorithm.
- `op_4.cu` - Storing the convolution kernel in constant memory for faster access.
- `op_5.cu` - `__restrict__` keyword and loop unrolling.
- `op_6.cu` - Combining different optmizations based on layer sizes.
- `report.pdf` - A comprehensive study of the effectiveness of the above optmizations by analyzing time, memory usage, and API calls using Nvidia profiling tools.



## Acknowledgement 
The boilerplate code was provided by UIUC CS483 teaching staff.
