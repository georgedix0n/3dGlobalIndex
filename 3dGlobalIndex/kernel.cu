
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_gid_calculation_3d(int *data)
{
	int tid = ((blockDim.x * threadIdx.y) + threadIdx.x) + ((blockDim.x * blockDim.y) * threadIdx.z);


	int blockId = blockIdx.x + (gridDim.x * blockIdx.y) + (gridDim.x * gridDim.y * blockIdx.z);

	int num_threads_in_block = blockId * blockDim.x * blockDim.y;

	int block_offset = num_threads_in_block * blockDim.z;


	int gid = tid + block_offset;
	printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d, threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, gid : %d - data :%d \n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, gid, data[gid]);

}

int main()
{
	int array_size = 64;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = { 780, 484, 58, 722, 359, 255, 173, 696, 209, 562, 161, 57, 37, 262, 398, 266, 845, 864, 337, 197, 510, 961, 124, 990, 753, 84, 673, 183, 204, 966, 708, 939, 772, 28, 98, 211, 53, 471, 803, 498, 697, 416, 763, 588, 950, 776, 404, 819, 452, 14, 487, 203, 390, 205, 387, 550, 219, 794, 974, 490, 538, 913, 13, 251 };

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(2, 2, 2);
	dim3 grid(2, 2, 2);

	unique_gid_calculation_3d << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}