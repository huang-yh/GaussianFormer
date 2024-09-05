#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"


__forceinline__ __device__ void getRect(const int* p, int max_radius, uint3& rect_min, uint3& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)(p[0] - max_radius))),
		min(grid.y, max((int)0, (int)(p[1] - max_radius))),
        min(grid.z, max((int)0, (int)(p[2] - max_radius)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)(p[0] + max_radius + 1))),
		min(grid.y, max((int)0, (int)(p[1] + max_radius + 1))),
        min(grid.z, max((int)0, (int)(p[2] + max_radius + 1)))
	};
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif