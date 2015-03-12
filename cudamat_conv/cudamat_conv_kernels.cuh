/**
 * CUDA kernels for convolution.
 *
 * Yujia Li, 03/2015
 */

#ifndef _CUDAMAT_CONV_KERNELS_CUH_
#define _CUDAMAT_CONV_KERNELS_CUH_

#define CONV_BLOCK_SIZE         256
#define CONV_MAX_NUM_BLOCKS     512

#ifndef MIN
#define MIN(x,y) \
    ({ __typeof__ (x) _x = (x); \
       __typeof__ (y) _y = (y); \
       _x > _y ? _y : _x; })
#endif

#ifndef MAX
#define MAX(x,y) \
    ({ __typeof__ (x) _x = (x); \
       __typeof__ (y) _y = (y); \
       _x > _y ? _y : _x; })

#endif

/**
 * Testing convolution code, no padding, stride=1.
 */
__global__ void kConvolveV1(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int n_ftr, int ftr_h, int ftr_w);

__global__ void kConvolve(float* image, float* filter, float* target,
        int n, int c, int im_h, int im_w, int ftr_h, int ftr_w, int pad_h, int pad_w, int stride_h, int stride_w);

#endif // _CUDAMAT_CONV_KERNELS_CUH_
