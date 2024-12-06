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

#include <torch/extension.h>
#include "aggregator_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    const int P,
    const int* points_xyz,
    const uint32_t* offsets,
    uint32_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int* radii,
    const dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint3 rect_min, rect_max;

    getRect(points_xyz + 3 * idx, radii + 3 * idx, rect_min, rect_max, grid);
    
    for (int x = rect_min.x; x < rect_max.x; x++)
    {
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int z = rect_min.z; z < rect_max.z; z++)
            {
                uint32_t key = x * grid.y * grid.z + y * grid.z + z;
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(
	int L, 
	uint32_t* point_list_keys, 
	uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint32_t currtile = point_list_keys[idx];
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1];
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


LocalAggregator::GeometryState LocalAggregator::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

LocalAggregator::ImageState LocalAggregator::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

LocalAggregator::BinningState LocalAggregator::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int LocalAggregator::Aggregator::forward(
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
	bool debug)
{
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(H * W * D);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, H * W * D);

	dim3 grid(H, W, D);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D_int,
		radii,
		grid,
		geomState.tiles_touched
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		means3D_int,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		grid)
	CHECK_CUDA(, debug);

	// int bit = getHigherMsb(H * W * D);
	int bit = 0;

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, H * W * D * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	CHECK_CUDA(FORWARD::render(
        N,
		pts,
		points_int,
		grid,
		imgState.ranges,
		binningState.point_list,
		means3D,
		cov3D,
		opas,
		semantics,
		out_logits,
		out_bin_logits,
		out_density,
		out_probability), debug);
	
	// return num_rendered;
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void LocalAggregator::Aggregator::backward(
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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, H * W * D);

	const dim3 grid(H, W, D);

	CHECK_CUDA(BACKWARD::preprocess(
		N,
		points_int,
		grid,
		voxel2pts
	), debug)

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	CHECK_CUDA(BACKWARD::render(
		P,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		voxel2pts,
		pts,
		means3D,
		cov3D,
		opas,
		semantics,
		logits,
		bin_logits,
		density,
		probability,
		logits_grad,
		bin_logits_grad,
		density_grad,
		means3D_grad,
		opas_grad,
		semantics_grad,
		cov3D_grad), debug)
}