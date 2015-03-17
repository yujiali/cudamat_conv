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

        while (tid / target_c_size < n) {
            t_w = tid % target_w;
            t_h = (tid % target_c_size) / target_w;
            t_n = tid / target_c_size;

            t_base_idx = t_n * target_im_size + t_h * target_w + t_w;
            i_base_idx = t_n * image_im_size + t_h * im_w + t_w;

            // changed this - should save some memory access, need testing
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

            tid += gridDim.x * blockDim.x;
        }
        ftr_idx += shared_batch_size;
    }
}


__global__ void kConvolveV3(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int n_ftr, int ftr_h, int ftr_w) {

    __shared__ float filter_cache[CONV_HALF_SHARED_MEMORY_SIZE];
    __shared__ float image_cache[CONV_HALF_SHARED_MEMORY_SIZE];
    const int target_h = im_h - ftr_h + 1;
    const int target_w = im_w - ftr_w + 1;
    const int target_c_size = target_h * target_w;
    const int target_im_size = target_c_size * n_ftr;

    const int image_c_size = im_h * im_w;
    const int image_im_size = image_c_size * c;
    const int image_full_size = image_im_size * n;

    const int ftr_c_size = ftr_h * ftr_w;
    const int ftr_im_size = ftr_c_size * c;
    const int ftr_c_full_size = ftr_c_size * n_ftr;

    const int cache_size = CONV_HALF_SHARED_MEMORY_SIZE;

    const int i_cache_w = blockDim.x - ftr_w + 1;
    const int i_cache_h = ftr_h;
    const int i_cache_size = i_cache_w * i_cache_h;

    int tid_base = blockIdx.x * blockDim.x;

    // TODO
    while (tid_base < target_im_size * n) {
        // load input
        int t_w = tid_base % target_w;
        int t_h = (tid_base % target_c_size) / target_w;
        int t_c = (tid_base % target_im_size) / target_c_size;
        int t_n = tid_base / target_im_size;

        int i_base = t_n * image_im_size + t_c * image_c_size + t_h * im_w + t_w;

        __syncthreads();
        for (int i = threadIdx.x; i < i_cache_size; i += blockDim.x)
            image_cache[i] = image[i_base + (i / i_cache_w) * im_w + (i % i_cache_w)];

        // load filter

        tid_base += gridDim.x * blockDim.x;
    }



    while (tid < image_im_size * n) {
        // load input
        __syncthreads();
        const int im_batch_size = MIN(cache_size, image_full_size - tid);
        for (int i = threadIdx.x; i < im_batch_size; i += blockDim.x)
            image_cache[i] = image[tid + i];
        __syncthreads();

        int i_w = tid

        int c_start = (tid % image_im_size) / image_c_size;
        int c_end = ((tid + batch_size - 1) % image_im_size) / image_c_size;
        if (c_end < c_start)
            c_end += c;

        for (int f_c = c_start; f_c <= c_end; f_c++) {
            int k = f_c % c;
            int fid = ftr_c_size * k;

            while (fid < ftr_im_size * n_ftr) {
                // load filter data
                __syncthreads();
                const int ftr_batch_size = MIN(cache_size, ftr_c_full_size - fid);
                for (int i = threadIdx.x; i < ftr_batch_size; i += blockDim.x) {
                    int hw_id = (fid + i) % ftr_c_size;
                    int n_id = (fid + i) / ftr_c_size;
                    filter_cache[i] = filter[n_id * ftr_im_size + k * ftr_c_size + hw_id];
                }
                __syncthreads();

                // compute outputs using image_cache and filter_cache
                for (int i = 0; i < ftr_batch_size; i++) {
                    int f_w = (fid + i) % ftr_w;
                    int f_h = ((fid + i) % ftr_c_size) / ftr_w;
                    int f_n = (fid + i) / ftr_c_size;

                    target;
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }

}

__global__ void kConvolve(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int ftr_h, int ftr_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    
}

