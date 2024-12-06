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
	const float* __restrict__ opas,
	const float* __restrict__ semantic,
	const float* __restrict__ logits,
	const float* __restrict__ bin_logits,
	const float* __restrict__ density,
	const float* __restrict__ probability,
	const float* __restrict__ logits_grad,
	const float* __restrict__ bin_logits_grad,
	const float* __restrict__ density_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opas_grad,
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
	const float opa = opas[idx];
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
			float deter = cov1.x * cov1.y * cov1.z + 2 * cov2.x * cov2.y * cov2.z - cov1.x * cov2.y * cov2.y - cov1.y * cov2.z * cov2.z - cov1.z * cov2.x * cov2.x;
			float prob = powf(2 * 3.1415926535, -1.5) * powf(deter, 0.5) * power;
			float power_grad = 0.;
			float deter_grad = 0.;
			float prob_grad = 0.;
			float prob_sum = probability[pts_idx];

			if (prob_sum > 1e-9) {
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					semantic_grad[ch] += logits_grad[pts_idx * CHANNELS + ch] * prob * opa / prob_sum;
					prob_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * opa / prob_sum;
					opa_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * prob / prob_sum;
				}
			} 
			power_grad += prob_grad * powf(2 * 3.1415926535, -1.5) * powf(deter, 0.5);
			power_grad += (1 - bin_logits[pts_idx]) / (1 - power + 1e-9) *  bin_logits_grad[pts_idx];
			power_grad += density_grad[pts_idx];
			deter_grad += prob_grad * prob / 2 / deter;

			means_grad[0] -= power_grad * power * (cov1.x * d.x + cov2.x * d.y + cov2.z * d.z);
			means_grad[1] -= power_grad * power * (cov2.x * d.x + cov1.y * d.y + cov2.y * d.z);
			means_grad[2] -= power_grad * power * (cov2.z * d.x + cov2.y * d.y + cov1.z * d.z);

			cov_grad[0] += power_grad * power * (- 0.5 * d.x * d.x) + deter_grad * (cov1.y * cov1.z - cov2.y * cov2.y);
			cov_grad[1] += power_grad * power * (- 0.5 * d.y * d.y) + deter_grad * (cov1.x * cov1.z - cov2.z * cov2.z);
			cov_grad[2] += power_grad * power * (- 0.5 * d.z * d.z) + deter_grad * (cov1.x * cov1.y - cov2.x * cov2.x);
			cov_grad[3] += power_grad * power * (- d.x * d.y) + 2 * deter_grad * (cov2.y * cov2.z - cov1.z * cov2.x);
			cov_grad[4] += power_grad * power * (- d.y * d.z) + 2 * deter_grad * (cov2.x * cov2.z - cov1.x * cov2.y);
			cov_grad[5] += power_grad * power * (- d.x * d.z) + 2 * deter_grad * (cov2.x * cov2.y - cov1.y * cov2.z);
		}
	}

	means3D_grad[idx * 3] = means_grad[0];
	means3D_grad[idx * 3 + 1] = means_grad[1];
	means3D_grad[idx * 3 + 2] = means_grad[2];
	opas_grad[idx] = opa_grad;
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
		opas,
		semantic,
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