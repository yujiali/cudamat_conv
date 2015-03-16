/**
 * Helper functions for testing.
 *
 * Yujia Li, 03/2015
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "../cudamat_conv.cuh"
#include "test_helper.h"

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

void _test_print_a_few_elements(cudamat_4d_tensor* t, const char* t_name, int n, bool is_first) {
    printf("<tensor %s>\n", t_name);
    const int t_size = tensor_size(t);

    if (!is_first)
        printf("...");
    else
        printf("   ");

    for (int i = 0; i < MIN(n, t_size); i++)
        if (is_first)
            printf("  %10g", t->data_host[i]);
        else
            printf("  %10g", t->data_host[t_size-i]);
    
    if (is_first)
        printf("   ...");

    printf("\n\n");
}

void _convolve(float* input, float* filter, float* output, 
        int input_n, int input_c, int input_h, int input_w,
        int ftr_n, int ftr_h, int ftr_w) {
    const int input_c_size = input_h * input_w;
    const int input_im_size = input_c_size * input_c;
    const int ftr_c_size = ftr_h * ftr_w;
    const int ftr_im_size = ftr_c_size * input_c;
    const int output_h = input_h - ftr_h + 1;
    const int output_w = input_w - ftr_w + 1;
    const int output_c_size = output_h * output_w;
    const int output_im_size = output_c_size * ftr_n;

    for (int n = 0; n < input_n; n++)
        for (int f = 0; f < ftr_n; f++)
            for (int h = 0; h < output_h; h++)
                for (int w = 0; w < output_w; w++) {
                    float s = 0;
                    for (int c = 0; c < input_c; c++)
                        for (int i = 0; i < ftr_h; i++)
                            for (int j = 0; j < ftr_w; j++)
                                s += input[n * input_im_size + c * input_c_size + (h + i) * input_w + (w + j)] * \
                                     filter[f * ftr_im_size + c * ftr_c_size + i * ftr_w + j];
                    output[n * output_im_size + f * output_c_size + h * output_w + w] = s;
                }
}

void _test_tensor_convolve(cudamat_4d_tensor* input, cudamat_4d_tensor* filter, cudamat_4d_tensor* output) {
    _test_create_tensor(output, input->n, filter->n, input->h - filter->h + 1, input->w - filter->w + 1);
    _convolve(input->data_host, filter->data_host, output->data_host,
            input->n, input->c, input->h, input->w, filter->n, filter->h, filter->w);
}


void _test_create_convolution_descriptor(
        cudamat_convolution_descriptor* d, int pad_h, int pad_w, int pad_type, int stride_h, int stride_w) {
    d->pad_h = pad_h;
    d->pad_w = pad_w;
    d->pad_type = pad_type;
    d->stride_h = stride_h;
    d->stride_w = stride_w;
}

void _test_tensor_to_cudamat(cudamat_4d_tensor* t, cudamat* mat) {
    mat->data_host = t->data_host;
    mat->data_device = t->data_device;
    mat->on_device = t->on_device;
    mat->on_host = 1;
    mat->size[0] = t->n;
    mat->size[1] = t->c * t->h * t->w;
    mat->is_trans = 0; // 0 or 1
    mat->owns_data = 1;
}

float _test_compute_l2_difference(cudamat_4d_tensor* t1, cudamat_4d_tensor* t2) {
    double diff = 0;
    int t_size = tensor_size(t1);
    if (tensor_size(t2) != t_size)
        return -1;

    for (int i = 0; i < t_size; i++)
        diff += (t1->data_host[i] - t2->data_host[i]) * (t1->data_host[i] - t2->data_host[i]);

    return sqrt(diff / t_size);
}

