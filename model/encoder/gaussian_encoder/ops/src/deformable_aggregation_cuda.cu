#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>


__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    float *output_ptr = output + idx;
    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride
        + groups_index
    );

    float result = 0;
    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        const float loc_w = sample_location[loc_offset];
        const float loc_h = sample_location[loc_offset + 1];
        
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;

                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const float *weights_ptr = (
                    weights + weight_offset + scale_index * num_groups
                    + weight_cam_stride * cam_index
                );
                result += bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_ptr) * *weights_ptr;
            }
        }
    }
    *output_ptr = result;
}


__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;
    const float grad = grad_output[idx];

    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride
        + groups_index
    );

    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        const float loc_w = sample_location[loc_offset];
        const float loc_h = sample_location[loc_offset + 1];
        
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;

                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const int weights_ptr = weight_offset + scale_index * num_groups + weight_cam_stride * cam_index;
                const float weight = weights[weights_ptr];

                float *grad_location_ptr = grad_sampling_location + loc_offset;
                float *grad_weights_ptr = grad_weights + weights_ptr;
                bilinear_sampling_grad(
                    mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
                    value_ptr,
                    grad,
                    grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
                );
            }
        }
    }
}


void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/512)), 512>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups
    );
}


void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_pts,
  int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    deformable_aggregation_grad_kernel
        <<<(int)ceil(((double)num_kernels/512)), 512>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups
    );
}
