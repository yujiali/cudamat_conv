/**
 * CUDA kernels for convolution.
 *
 * Yujia Li, 03/2015
 */

#include "cudamat_conv_kernels.cuh"

__global__ void kConvolveV1(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int n_ftr, int ftr_h, int ftr_w) {

    const int target_h = im_h - ftr_h + 1;
    const int target_w = im_w - ftr_w + 1;
    const int target_c_size = target_h * target_w;
    const int target_im_size = target_c_size * n_ftr;

    const int image_c_size = im_h * im_w;
    const int image_im_size = image_c_size * c;

    const int ftr_c_size = ftr_h * ftr_w;
    const int ftr_im_size = ftr_c_size * c;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid / target_im_size < n) {
        int t_w = tid % target_w;
        int t_h = (tid % target_c_size) / target_w;
        int t_c = (tid % target_im_size) / target_c_size;
        int t_n = tid / target_im_size;

        float s = 0;

        for (int k = 0; k < c; k++)
            for (int i = 0; i < ftr_h; i++)
                for (int j = 0; j < ftr_w; j++) {
                    s += image[t_n * image_im_size + k * image_c_size + (t_h + i) * im_w + (t_w + j)] * \
                         filter[t_c * ftr_im_size + k * ftr_c_size + i * ftr_w + j];
                }

        target[tid] = s;

        tid += gridDim.x * blockDim.x;
    }
}

__global__ void kConvolveV2(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int n_ftr, int ftr_h, int ftr_w) {
    __shared__ float partial[CONV_SHARED_MEMORY_SIZE];
    const int target_h = im_h - ftr_h + 1;
    const int target_w = im_w - ftr_w + 1;
    const int target_c_size = target_h * target_w;
    const int target_im_size = target_c_size * n_ftr;

    const int image_c_size = im_h * im_w;
    const int image_im_size = image_c_size * c;

    const int ftr_c_size = ftr_h * ftr_w;
    const int ftr_im_size = ftr_c_size * c;
    const int ftr_full_size = ftr_im_size * n_ftr;

    const shared_batch_size = CONV_SHARED_MEMORY_SIZE / ftr_im_size * ftr_im_size;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid / target_im_size < n) {
        int t_w = tid % target_w;
        int t_h = (tid % target_c_size) / target_w;
        int t_c = (tid % target_im_size) / target_c_size;
        int t_n = tid / target_im_size;

        int ftr_idx = 0;
        while (ftr_idx < ftr_full_size) {

            // collectively load one batch of filters
            __syncthreads();
            for (int i = threadIdx.x; i < shared_batch_size && i < ftr_full_size - ftr_idx; i += blockDim.x)
                partial[i] = filter[t_c * ftr_im_size + ftr_idx + i];
            __syncthreads();

            ftr_idx += shared_batch_size;
        }

        float s = 0;

        for (int k = 0; k < c; k++)
            for (int i = 0; i < ftr_h; i++)
                for (int j = 0; j < ftr_w; j++) {
                    s += image[t_n * image_im_size + k * image_c_size + (t_h + i) * im_w + (t_w + j)] * \
                         filter[t_c * ftr_im_size + k * ftr_c_size + i * ftr_w + j];
                }

        target[tid] += s;

        tid += gridDim.x * blockDim.x;
    }
}

__global__ void kConvolve(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int ftr_h, int ftr_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    
}

