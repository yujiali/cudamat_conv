/**
 * Helper functions for testing.
 *
 * Yujia Li, 03/2015
 */

#ifndef _TEST_HELPER_H_
#define _TEST_HELPER_H_

#include "../cudamat_conv.cuh"


void _test_create_tensor(cudamat_4d_tensor* t, int n, int c, int h, int w);
void _test_free_tensor(cudamat_4d_tensor* t);

void _test_fill_tensor_with_constant(cudamat_4d_tensor* t, float value);
void _test_fill_tensor_with_rand_positive(cudamat_4d_tensor* t);
void _test_fill_tensor_with_rand_pos_neg(cudamat_4d_tensor* t);
void _test_fill_tensor_with_toy_data(cudamat_4d_tensor* t);

void _test_print_small_tensor(cudamat_4d_tensor* t, const char* t_name);

void _test_tensor_convolve(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output);

#endif // _TEST_HELPER_H_
