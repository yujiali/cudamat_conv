/**
 * cudamat-conv: GPU acceleration of convolution operations based on cudamat.
 *
 * Yujia Li, 03/2015
 */

#ifndef _CUDAMAT_CONV_CUH_
#define _CUDAMAT_CONV_CUH_

#define CUDAMAT_CONV_SUCCESS    0

/**
 * 4D tensors for image data.
 */
struct cudamat_4d_tensor {
    float *data_host;       // pointer to host data
    float *data_device;     // pointer to device data
    int on_device;          // 1 if a copy of data is on device, 0 otherwise
    int n;                  // number of images
    int h;                  // height of each feature map
    int w;                  // width of each feature map
    int c;                  // number of channels/feature maps
};

// ------------------------ utility functions -------------------------------- //

/**
 * Return the number of elements in the tensor.
 */
int tensor_size(cudamat_4d_tensor* t);



// ------------------------ memory management -------------------------------- //

/**
 * Allocate memory on device for the given tensor. The on_device bit will be
 * set to 1 after this call.
 */
int tensor_allocate_memory_on_device(cudamat_4d_tensor* t);

/**
 * Free the device memory allocated via tensor_allocate_memory_on_device. The
 * on_device bit will be set to 0 after this call.
 */
int tensor_free_memory_on_device(cudamat_4d_tensor* t);



// ------------------------ move data around -------------------------------- //

/**
 * Copy data from host memory to device memory, allocate device memory if data
 * is not already on device, assume host memory is already allocated.
 */
int tensor_copy_to_device(cudamat_4d_tensor* t);

/**
 * Copy data from device memory to host memory, assume data is already on
 * device, and host memory is already allocated.
 */
int tensor_copy_to_host(cudamat_4d_tensor* t);

/**
 * Copy data from device memory to device memory. src and dst should have the
 * same size.  Allocate device memory for dst if it is not already allocated.
 */
int tensor_copy_on_device(cudamat_4d_tensor* t_src, cudamat_4d_tensor* t_dst);



// ------------------------ initialization -------------------------------- //

/**
 * Initialize an empty cudamat_4d_tensor instance t, with the specified sizes.
 *
 * on_device bit will be set to 0. data_device and data_host will not be set.
 */
int tensor_init_empty(cudamat_4d_tensor* t, int n, int h, int w, int c);

/**
 * Initialize an empty cudamat_4d_tensor instance t, with the specified sizes
 * and a pointer to some host data.
 *
 * on_device bit will be set to 0.  data_device will not be set.
 */
int tensor_init_with_array(cudamat_4d_tensor* t, float* data, int n, int h, int w, int c);

/**
 * Fill the data_device for the given tensor with pseudo random numbers 
 * uniformly distributed in [0,1].
 */
int tensor_fill_with_rand(cudamat_4d_tensor* t);

/**
 * Fill the data_device for the given tensor with pseudo random numbers from a
 * standard normal distribution.
 */
int tensor_fill_with_randn(cudamat_4d_tensor* t);



// ------------------------ algebraic operations -------------------------------- //

int tensor_convolve(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output);

#endif  // _CUDAMAT_CONV_CUH_

