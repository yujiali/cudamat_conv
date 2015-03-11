/**
 * Helper functions for testing.
 *
 * Yujia Li, 03/2015
 */

#include <cstdlib>
#include <cstdio>

#include "../cudamat_conv.cuh"
#include "test_helper.h"

void _test_create_tensor(cudamat_4d_tensor* t, int n, int c, int h, int w) {
    tensor_init_empty(t, n, c, h, w);
    int t_size = tensor_size(t);

    t->data_host = (float*) malloc(sizeof(float) * t_size);

    _test_fill_tensor_with_toy_data(t);
}

void _test_free_tensor(cudamat_4d_tensor* t) {
    tensor_free_memory_on_device(t);
    free(t->data_host);
    tensor_init_empty(t, 0, 0, 0, 0);
}

void _test_fill_tensor_with_constant(cudamat_4d_tensor* t, float value) {
    int t_size = tensor_size(t);

    for (int i = 0; i < t_size; i++)
        t->data_host[i] = value;
}

void _test_fill_tensor_with_rand_positive(cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);

    for (int i = 0; i < t_size; i++)
        t->data_host[i] = (float) rand() / RAND_MAX;
}

void _test_fill_tensor_with_rand_pos_neg(cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);

    for (int i = 0; i < t_size; i++)
        t->data_host[i] = 2.0 * rand() / RAND_MAX - 1;
}

void _test_fill_tensor_with_toy_data(cudamat_4d_tensor* t) {
    int t_size = tensor_size(t);

    for (int i = 0; i < t_size; i++)
        t->data_host[i] = i % 5;
}

void _test_print_small_tensor(cudamat_4d_tensor* t, const char* t_name) {
    printf("<tensor %s>\n", t_name);

    const int c_size = t->h * t->w;
    const int im_size = t->c * c_size;

    for (int n = 0; n < t->n; n++) {
        printf("\n*** n=%d ***\n", n);
        for (int row = 0; row < t->h; row++) {
            for (int c = 0; c < t->c; c++) {
                printf("     ");
                for (int col = 0; col < t->w; col++)
                    printf("%10.5g", t->data_host[n * im_size + c * c_size + row * t->w + col]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

void _test_tensor_convolve(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output) {
    // TODO
}

