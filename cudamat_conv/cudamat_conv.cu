/**
 * cudamat-conv: GPU acceleration of convolution operations based on cudamat.
 *
 * Yujia Li, 03/2015
 */

#include <cuda_runtime.h>
#include <cublas.h>

#include "cudamat_conv.cuh"
#include "cudamat.cuh"

#define CUBLAS_FUNCTION_CALL()

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
    }

    cublasScopy(src_size, src->data_device, 1, dst->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return CUDAMAT_CONV_SUCCESS;
}

// ------------------------ initialization -------------------------------- //

int tensor_init_empty(cudamat_4d_tensor* t, int n, int h, int w, int c) {
    t->n = n;
    t->h = h;
    t->w = w;
    t->c = c;

    t->on_device = 0;

    return CUDAMAT_CONV_SUCCESS;
}

int tensor_init_with_array(cudamat_4d_tensor* t, float* data, int n, int h, int w, int c) {
    t->n = n;
    t->h = h;
    t->w = w;
    t->c = c;

    t->on_device = 0;

    t->data_host = data;

    return CUDAMAT_CONV_SUCCESS;
}


int tensor_fill_with_rand(cudamat_4d_tensor* t) {
    // TODO
    return CUDAMAT_CONV_SUCCESS;
}

int tensor_fill_with_randn(cudamat_4d_tensor* t) {
    // TODO
    return CUDAMAT_CONV_SUCCESS;
}


// ------------------------ algebraic operations -------------------------------- //

int tensor_convolve(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output) {
    if (!(input->on_device) || !(filter->on_device) || !(output->on_device))
        return ERROR_NOT_ON_DEVICE;
    if (input->c != filter->c || output->c != filter->n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    return CUDAMAT_CONV_SUCCESS;
}

#ifdef __cplusplus
}
#endif

