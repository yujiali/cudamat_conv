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

    // const int shared_batch_size = CONV_SHARED_MEMORY_SIZE / ftr_im_size * ftr_im_size;
    const int shared_batch_size = CONV_SHARED_MEMORY_SIZE;

    int ftr_idx = 0;
    int tid;
    int f_w, f_h, f_c, f_n;
    int t_w, t_h, t_n, t_base_idx, i_base_idx;

    while (ftr_idx < ftr_full_size) {
        // load one batch of filter
        __syncthreads();
        for (int i = threadIdx.x; i < shared_batch_size && i + ftr_idx < ftr_full_size; i += blockDim.x)
            partial[i] = filter[i + ftr_idx];
        __syncthreads();

        tid = threadIdx.x + blockIdx.x * blockDim.x;

        while (tid < target_c_size * n) {
            t_w = tid % target_w;
            t_h = (tid % target_c_size) / target_w;
            t_n = tid / target_c_size;

            t_base_idx = t_n * target_im_size + t_h * target_w + t_w;
            i_base_idx = t_n * image_im_size + t_h * im_w + t_w;

            int last_f_n = -1;
            float s = 0;

            for (int i = 0; i < shared_batch_size && i + ftr_idx < ftr_full_size; i++) {
                f_w = (i + ftr_idx) % ftr_w;
                f_h = ((i + ftr_idx) % ftr_c_size) / ftr_w;
                f_c = ((i + ftr_idx) % ftr_im_size) / ftr_c_size;
                f_n = (i + ftr_idx) / ftr_im_size;
                if (i > 0 && f_n != last_f_n) {
                    target[t_base_idx + last_f_n * target_c_size] += s;
                    s = 0;
                    last_f_n = f_n;
                }

                s += partial[i] * image[i_base_idx + f_c * image_c_size + f_h * im_w + f_w];
            }
            if (s != 0)
                target[t_base_idx + last_f_n * target_c_size] += s;

            tid += gridDim.x * blockDim.x;
        }
        ftr_idx += shared_batch_size;
    }
}


__global__ void kConvolveV3(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int n_ftr, int ftr_h, int ftr_w) {

    __shared__ float target_cache[CONV_TILE_SIZE][CONV_TILE_SIZE];
    __shared__ float filter_cache[CONV_MAX_FILTER_SIZE][CONV_MAX_FILTER_SIZE];
    __shared__ float image_cache[CONV_TILE_SIZE + CONV_MAX_FILTER_SIZE - 1][CONV_TILE_SIZE + CONV_MAX_FILTER_SIZE - 1];

    const int target_h = im_h - ftr_h + 1;
    const int target_w = im_w - ftr_w + 1;
    const int target_c_size = target_h * target_w;
    const int target_im_size = target_c_size * n_ftr;

    const int image_c_size = im_h * im_w;
    const int image_im_size = image_c_size * c;

    const int ftr_c_size = ftr_h * ftr_w;
    const int ftr_im_size = ftr_c_size * c;

    const int n_blocks_w = (target_w + blockDim.x - 1) / blockDim.x;
    const int n_blocks_h = (target_h + blockDim.y - 1) / blockDim.y;
    const int n_blocks_c = n_blocks_h * n_blocks_w;
    const int n_blocks_im = n_blocks_c * n_ftr;

    int bid = blockIdx.x;

    // loop over all output blocks
    while (bid < n * n_blocks_im) {
        int b_n = bid / n_blocks_im;
        int b_c = (bid % n_blocks_im) / n_blocks_c;
        int b_h = (bid % n_blocks_c) / n_blocks_w;
        int b_w = bid % n_blocks_w;

        int i_base_h = b_h * blockDim.y;
        int i_base_w = b_w * blockDim.x;

        // reset all targests
        // __syncthreads();
        target_cache[threadIdx.y][threadIdx.x] = 0;

        // loop over input channels
        for (int k = 0; k < c; k++) {
            int i_base = b_n * image_im_size + k * image_c_size;
            int f_base = b_c * ftr_im_size + k * ftr_c_size;

            // load data cache first

            for (int h = threadIdx.y; h < blockDim.y + ftr_h - 1 && h < im_h; h += blockDim.y)
                for (int w = threadIdx.x; w < blockDim.x + ftr_w - 1 && w < im_w; w += blockDim.x)
                    image_cache[h][w] = image[i_base + (i_base_h + h) * im_w + (i_base_w + w)];
            // __syncthreads();

            for (int h = threadIdx.y; h < ftr_h; h += blockDim.y)
                for (int w = threadIdx.x; w < ftr_w; w += blockDim.x)
                    filter_cache[h][w] = filter[f_base + h * ftr_w + w];
            __syncthreads();

            if (i_base_h + threadIdx.y < target_h && i_base_w + threadIdx.x < target_w) {
                float s = 0;
                for (int h = 0; h < ftr_h; h++)
                    for (int w = 0; w < ftr_w; w++)
                        s += image_cache[threadIdx.y + h][threadIdx.x + w] * filter_cache[h][w];
                target_cache[threadIdx.y][threadIdx.x] += s;
            }
            __syncthreads();
        }

        // write to output
        if (i_base_h + threadIdx.y < target_h && i_base_w + threadIdx.x < target_w)
            target[b_n * target_im_size + b_c * target_c_size + (i_base_h + threadIdx.y) * target_w + (i_base_w + threadIdx.x)] \
                = target_cache[threadIdx.y][threadIdx.x];

        bid += blockDim.x;
    }
}

__global__ void kConvolve(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int ftr_h, int ftr_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    
}

