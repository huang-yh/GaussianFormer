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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace LocalAggregator
{
	class Aggregator
	{
	public:

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int N,
			const float* pts,
			const int* points_int,
			const float* means3D,
			const int* means3D_int,
			const float* opas,
			const float* semantics,
			const float* cov3D,
			const int* radii,
			const int H,
			const int W,
			const int D,
			float* out_logits,
			float* out_bin_logits,
			float* out_density,
			float* out_probability,
			bool debug = false);

		static void backward(
			const int P, int R, int N,
			const int H, int W, int D,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const int* points_int,
			int* voxel2pts,
			const float* pts,
			const float* means3D,
			const float* cov3D,
			const float* opas,
			const float* semantics,
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
			float* cov3D_grad,
			bool debug = false);
	};
};

#endif