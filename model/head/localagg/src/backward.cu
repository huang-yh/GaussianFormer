#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	int voxel_idx = points_xyz[3 * idx] * grid.y * grid.z + points_xyz[3 * idx + 1] * grid.z + points_xyz[3 * idx + 2];
	voxel2pts[voxel_idx] = idx;
}


template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const int P,
	const uint32_t* __restrict__ offsets,
	const uint32_t* __restrict__ point_list_keys_unsorted,
	const int* __restrict__ voxel2pts,
	const float* __restrict__ pts,
	const float* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ opacity,
	const float* __restrict__ semantic,
	const float* __restrict__ out_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opacity_grad,
	float* __restrict__ semantics_grad,
	float* __restrict__ cov3D_grad)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
	    return;

	uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
	uint32_t end = offsets[idx];
	
	const float3 means = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float3 cov1 = {cov3D[6 * idx], cov3D[6 * idx + 1], cov3D[6 * idx + 2]};
	const float3 cov2 = {cov3D[6 * idx + 3], cov3D[6 * idx + 4], cov3D[6 * idx + 5]};
	const float opa = opacity[idx];
	float sem[CHANNELS] = {0};
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		sem[ch] = semantic[idx * CHANNELS + ch];
	}

	float means_grad[3] = {0};
	float opa_grad = 0;
	float semantic_grad[CHANNELS] = {0};
	float cov_grad[6] = {0};

	for (int i = start; i < end; i++)
	{
		int voxel_idx = point_list_keys_unsorted[i];
		int pts_idx = voxel2pts[voxel_idx];
		if (pts_idx >= 0)
		{
			float3 d = {means.x - pts[pts_idx * 3], means.y - pts[pts_idx * 3 + 1], means.z - pts[pts_idx * 3 + 2]};
			float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
			power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
			power = exp(power);
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				float curr_out_grad = power * out_grad[pts_idx * CHANNELS + ch];
				opa_grad += sem[ch] * curr_out_grad;
				semantic_grad[ch] += opa * curr_out_grad;
				float cov_grad_coeff = opa * sem[ch] * curr_out_grad;
				cov_grad[0] += -0.5f * cov_grad_coeff * d.x * d.x;
				cov_grad[1] += -0.5f * cov_grad_coeff * d.y * d.y;
				cov_grad[2] += -0.5f * cov_grad_coeff * d.z * d.z;
				cov_grad[3] += -1.0f * cov_grad_coeff * d.x * d.y;
				cov_grad[4] += -1.0f * cov_grad_coeff * d.y * d.z;
				cov_grad[5] += -1.0f * cov_grad_coeff * d.x * d.z;
				means_grad[0] += -1.0f * cov_grad_coeff * (cov1.x * d.x + cov2.x * d.y + cov2.z * d.z);
				means_grad[1] += -1.0f * cov_grad_coeff * (cov1.y * d.y + cov2.x * d.x + cov2.y * d.z);
				means_grad[2] += -1.0f * cov_grad_coeff * (cov1.z * d.z + cov2.y * d.y + cov2.z * d.x);
			}
		}
	}

	means3D_grad[idx * 3] = means_grad[0];
	means3D_grad[idx * 3 + 1] = means_grad[1];
	means3D_grad[idx * 3 + 2] = means_grad[2];
	opacity_grad[idx] = opa_grad;
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		semantics_grad[idx * CHANNELS + ch] = semantic_grad[ch];
	}
	for (int ch = 0; ch < 6; ch++)
	{
		cov3D_grad[idx * 6 + ch] = cov_grad[ch];
	}
}


void BACKWARD::render(
	const int P,
	const uint32_t* offsets,
	const uint32_t* point_list_keys_unsorted,
	const int* voxel2pts,
	const float* pts,
	const float* means3D,
	const float* cov3D,
	const float* opacity,
	const float* semantic,
	const float* out_grad,
	float* means3D_grad,
	float* opacity_grad,
	float* semantics_grad,
	float* cov3D_grad)
{
	renderCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		offsets,
		point_list_keys_unsorted,
		voxel2pts,
		pts,
		means3D,
		cov3D,
		opacity,
		semantic,
		out_grad,
		means3D_grad,
		opacity_grad,
		semantics_grad,
		cov3D_grad);
}

void BACKWARD::preprocess(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	preprocessCUDA << <(N + 255) / 256, 256 >> > (
		N,
		points_xyz,
		grid,
		voxel2pts
	);
}