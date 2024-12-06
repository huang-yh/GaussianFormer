/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
        const int P,
        const int* points_xyz,
        const int* radii,
        const dim3 grid,
        uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
        const int N,
        const float* pts,
        const int* points_int,
        const dim3 grid,
        const uint2* ranges,
        const uint32_t* point_list,
        const float* means3D,
        const float* cov3D,
        const float* opas,
        const float* semantic,
        float* out_logits,
        float* out_bin_logits,
        float* out_density,
        float* out_probability);
}


#endif