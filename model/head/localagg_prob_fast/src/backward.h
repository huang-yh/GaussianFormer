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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
        const int P,
        const uint32_t* offsets,
        const uint32_t* point_list_keys_unsorted,
        const int* voxel2pts,
        const float* pts,
        const float* means3D,
        const float* cov3D,
        const float* opas,
        const float* semantic,
	const float* logits,
	const float* bin_logits,
	const float* density,
        const float* probability,
        const float* logits_grad,
        const float* bin_logits_grad,
        const float* density_grad,
        float* means3D_grad,
        float* opas_grad,
        float* semantics_grad,
        float* cov3D_grad);

	void preprocess(
        const int N,
        const int* points_xyz,
        const dim3 grid,
        int* voxel2pts);
}

#endif