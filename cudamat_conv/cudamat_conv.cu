/**
 * cudamat-conv: GPU acceleration of convolution operations based on cudamat.
 *
 * Yujia Li, 03/2015
 */

#include <cuda_runtime.h>
#include <cublas.h>

#include "cudamat_conv.cuh"
#include "cudamat.cuh"
#include "cudamat_kernels.cuh"
#include "cudamat_conv_kernels.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------ utility functions -------------------------------- //

int tensor_size(cudamat_4d_tensor* t) {
    return t->n * t->h * t->w * t->c;
}


// ------------------------ memory management -------------------------------- //

int tensor_allocate_memory_on_device(cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);
    if (t_size <= 0)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasStatus stat = cublasAlloc(t_size, sizeof(float), (void**)&(t->data_device));
    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
        return CUBLAS_ERROR;

    t->on_device = 1;
    return CUDAMAT_CONV_SUCCESS;
}

int tensor_free_memory_on_device(cudamat_4d_tensor* t) {
    if (t->on_device) {
        cublasStatus stat = cublasFree(t->data_device);
        if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
            return CUBLAS_ERROR;

        t->on_device = 0;
    }

    return CUDAMAT_CONV_SUCCESS;
}


// ------------------------ move data around -------------------------------- //

int tensor_copy_to_device(cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);
    if (t_size <= 0)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!(t->on_device)) {
        int err_code = tensor_allocate_memory_on_device(t);
        if (err_code)
            return err_code;
        t->on_device = 1;
    }

    cublasStatus stat = cublasSetVector(t_size, sizeof(float), t->data_host, 1, t->data_device, 1);
    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
        return CUBLAS_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

int tensor_copy_to_host(cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);
    if (t_size <= 0)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!(t->on_device))
        return ERROR_NOT_ON_DEVICE;

    cublasStatus stat = cublasGetVector(t_size, sizeof(float), t->data_device, 1, t->data_host, 1);
    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
        return CUBLAS_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

int tensor_copy_on_device(cudamat_4d_tensor* src, cudamat_4d_tensor* dst) {
    int src_size = tensor_size(src);
    int dst_size = tensor_size(src);

    if (src_size != dst_size || src_size <= 0 || dst_size <= 0)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!(src->on_device))
        return ERROR_NOT_ON_DEVICE;

    if (!(dst->on_device)) {
        int err_code = tensor_allocate_memory_on_device(dst);
        if (err_code)
            return err_code;
        dst->on_device = 1;
    }

    cublasScopy(src_size, src->data_device, 1, dst->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

// ------------------------ initialization -------------------------------- //

int tensor_init_empty(cudamat_4d_tensor* t, int n, int c, int h, int w) {
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;

    t->on_device = 0;

    return CUDAMAT_CONV_SUCCESS;
}

int tensor_init_with_array(cudamat_4d_tensor* t, float* data, int n, int c, int h, int w) {
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;

    t->on_device = 0;

    t->data_host = data;

    return CUDAMAT_CONV_SUCCESS;
}


int tensor_fill_with_rand(rnd_struct* rnd_state, cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);
    if (t_size <= 0)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!(t->on_device))
        return ERROR_NOT_ON_DEVICE;

    kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, t->data_device, t_size);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

int tensor_fill_with_randn(rnd_struct* rnd_state, cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);
    if (t_size <= 0)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!(t->on_device))
        return ERROR_NOT_ON_DEVICE;

    kRandomGaussian<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, t->data_device, t_size);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}


// ------------------------ algebraic operations -------------------------------- //

int tensor_convolve(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output,
        cudamat_convolution_descriptor* desc) {
    if (!(input->on_device) || !(filter->on_device) || !(output->on_device))
        return ERROR_NOT_ON_DEVICE;
    if (input->c != filter->c || output->c != filter->n || output->n != input->n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (input->h + desc->pad_h * 2 < filter->h || input->w + desc->pad_w * 2 < filter->w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (output->h != input->h + desc->pad_h * 2 - filter->h + 1 || output->w != input->w + desc->pad_w * 2 - filter->w + 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    const int block_size = CONV_BLOCK_SIZE;
    const int output_size = tensor_size(output);
    const int n_blocks = MIN((output_size + block_size - 1) / block_size, CONV_MAX_NUM_BLOCKS);

    cudaError_t err = cudaMemset(output->data_device, 0, tensor_size(output) * sizeof(float));
    if (err != cudaSuccess || checkCUDAError())
        return CUDA_ERROR;

    kConvolveV1<<<n_blocks, block_size>>>(input->data_device, filter->data_device, output->data_device,
        input->n, input->c, input->h, input->w, filter->n, filter->h, filter->w);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

int tensor_convolve2(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output,
        cudamat_convolution_descriptor* desc) {
    if (!(input->on_device) || !(filter->on_device) || !(output->on_device))
        return ERROR_NOT_ON_DEVICE;
    if (input->c != filter->c || output->c != filter->n || output->n != input->n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (input->h + desc->pad_h * 2 < filter->h || input->w + desc->pad_w * 2 < filter->w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (output->h != input->h + desc->pad_h * 2 - filter->h + 1 || output->w != input->w + desc->pad_w * 2 - filter->w + 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    const int block_size = CONV_BLOCK_SIZE;
    const int output_size = tensor_size(output) / output->c;
    const int n_blocks = MIN((output_size + block_size - 1) / block_size, CONV_MAX_NUM_BLOCKS);

    cudaError_t err = cudaMemset(output->data_device, 0, tensor_size(output) * sizeof(float));
    if (err != cudaSuccess || checkCUDAError())
        return CUDA_ERROR;

    kConvolveV2<<<n_blocks, block_size>>>(input->data_device, filter->data_device, output->data_device,
        input->n, input->c, input->h, input->w, filter->n, filter->h, filter->w);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}


int tensor_convolve3(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output,
        cudamat_convolution_descriptor* desc) {
    if (!(input->on_device) || !(filter->on_device) || !(output->on_device))
        return ERROR_NOT_ON_DEVICE;
    if (input->c != filter->c || output->c != filter->n || output->n != input->n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (input->h + desc->pad_h * 2 < filter->h || input->w + desc->pad_w * 2 < filter->w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (output->h != input->h + desc->pad_h * 2 - filter->h + 1 || output->w != input->w + desc->pad_w * 2 - filter->w + 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 block_dim(CONV_TILE_SIZE, CONV_TILE_SIZE);
    const int n_blocks = MIN(output->n * output->c * ((output->h + CONV_TILE_SIZE - 1) / CONV_TILE_SIZE) * \
                         ((output->w + CONV_TILE_SIZE -1) / CONV_TILE_SIZE), CONV_MAX_NUM_BLOCKS);

    cudaError_t err = cudaMemset(output->data_device, 0, tensor_size(output) * sizeof(float));
    if (err != cudaSuccess || checkCUDAError())
        return CUDA_ERROR;

    kConvolveV3<<<n_blocks, block_dim>>>(input->data_device, filter->data_device, output->data_device,
        input->n, input->c, input->h, input->w, filter->n, filter->h, filter->w);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

#ifdef __cplusplus
}
#endif

