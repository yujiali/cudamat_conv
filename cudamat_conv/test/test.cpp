#include <cstdio>
#include <ctime>

#include "test_helper.h"
#include "cgpulock.h"
#include "../cudamat_conv.cuh"

void test_helper_io() {
    cudamat_4d_tensor t;
    _test_create_tensor(&t, 2, 3, 3, 2);
    _test_print_small_tensor(&t, "init");

    _test_fill_tensor_with_constant(&t, 0);
    _test_print_small_tensor(&t, "zero");

    _test_fill_tensor_with_rand_positive(&t);
    _test_print_small_tensor(&t, "rand");

    _test_fill_tensor_with_rand_pos_neg(&t);
    _test_print_small_tensor(&t, "rand [-1,1]");

    _test_fill_tensor_with_toy_data(&t);
    _test_print_small_tensor(&t, "toy");

    _test_free_tensor(&t);
}

void test_data_transfer() {
    cudamat_4d_tensor t1, t2;

    _test_create_tensor(&t1, 2, 3, 3, 2);
    _test_fill_tensor_with_rand_positive(&t1);

    _test_create_tensor(&t2, 2, 3, 3, 2);
    _test_fill_tensor_with_constant(&t2, 0);

    _test_print_small_tensor(&t1, "t1");
    _test_print_small_tensor(&t2, "t2");

    tensor_copy_to_device(&t1);
    tensor_copy_on_device(&t1, &t2);
    tensor_copy_to_host(&t2);

    _test_print_small_tensor(&t1, "t1");
    _test_print_small_tensor(&t2, "t2");

    _test_free_tensor(&t1);
    _test_free_tensor(&t2);
}

void test_init() {
    cudamat_4d_tensor t;
    rnd_struct r;
    init_random(&r, 0, "../rnd_multipliers_32bit.txt");

    _test_create_tensor(&t, 2, 3, 3, 2);
    tensor_copy_to_device(&t);
    _test_print_small_tensor(&t, "init");

    tensor_fill_with_rand(&r, &t);
    tensor_copy_to_host(&t);
    _test_print_small_tensor(&t, "rand");

    tensor_fill_with_randn(&r, &t);
    tensor_copy_to_host(&t);
    _test_print_small_tensor(&t, "randn");

    _test_free_tensor(&t);
}

void test_convolution() {
    cudamat_4d_tensor in, ftr, cpu_out, gpu_out;

    _test_create_tensor(&in, 2, 2, 4, 4);
    _test_create_tensor(&ftr, 2, 2, 3, 3);

    _test_fill_tensor_with_toy_data(&in);
    // _test_fill_tensor_with_constant(&ftr, 1);
    _test_fill_tensor_with_toy_data(&ftr);

    _test_print_small_tensor(&in, "in");
    _test_print_small_tensor(&ftr, "filter");

    _test_tensor_convolve(&in, &ftr, &cpu_out);

    _test_print_small_tensor(&cpu_out, "cpu_out");

    _test_create_tensor(&gpu_out, 2, 2, 2, 2);
    _test_fill_tensor_with_constant(&gpu_out, 0);

    tensor_copy_to_device(&in);
    tensor_copy_to_device(&ftr);
    tensor_copy_to_device(&gpu_out);

    cudamat_convolution_descriptor d;

    _test_create_convolution_descriptor(&d, 0, 0, 0, 1, 1);
    tensor_convolve(&in, &ftr, &gpu_out, &d);

    tensor_copy_to_host(&gpu_out);
    _test_print_small_tensor(&gpu_out, "gpu_out");

    _test_free_tensor(&in);
    _test_free_tensor(&ftr);
    _test_free_tensor(&cpu_out);
    _test_free_tensor(&gpu_out);
}

void test_convolution_speed() {
    cudamat_4d_tensor in, ftr, cpu_out, gpu_out;

    int n = 256;
    int c = 3;
    int h = 128;
    int w = 128;
    int n_ftr = 256;
    int ftr_h = 3;
    int ftr_w = 3;

    _test_create_tensor(&in, n, c, h, w);
    _test_create_tensor(&ftr, n_ftr, c, ftr_h, ftr_w);

    printf("OK[1]\n");

    // _test_fill_tensor_with_rand_positive(&in);
    // _test_fill_tensor_with_rand_positive(&ftr);
    // _test_fill_tensor_with_constant(&ftr, 1);

    _test_fill_tensor_with_toy_data(&in);
    _test_fill_tensor_with_toy_data(&ftr);

    // _test_print_small_tensor(&in, "in");
    // _test_print_small_tensor(&ftr, "filter");
    
    printf("OK[2]\n");
    
    clock_t t_start = clock();
    _test_tensor_convolve(&in, &ftr, &cpu_out);
    clock_t t_end = clock();

    printf("cpu time: %.5f\n", (float)(t_end - t_start) / CLOCKS_PER_SEC);

    // _test_print_small_tensor(&cpu_out, "cpu_out");

    t_start = clock();
    _test_create_tensor(&gpu_out, n, n_ftr, h - ftr_h + 1, w - ftr_w + 1);
    _test_fill_tensor_with_constant(&gpu_out, 0);

    printf("OK[3]\n");

    tensor_copy_to_device(&in);
    tensor_copy_to_device(&ftr);
    tensor_copy_to_device(&gpu_out);

    cudamat_convolution_descriptor d;

    printf("OK[4]\n");

    _test_create_convolution_descriptor(&d, 0, 0, 0, 1, 1);
    tensor_convolve2(&in, &ftr, &gpu_out, &d);
    // tensor_convolve(&in, &ftr, &gpu_out, &d);

    printf("OK[5]\n");

    tensor_copy_to_host(&gpu_out);
    // _test_print_small_tensor(&gpu_out, "gpu_out");
    t_end = clock();
    
    printf("gpu time: %.5f\n", (float)(t_end - t_start) / CLOCKS_PER_SEC);
    printf("diff: %.8f\n", _test_compute_l2_difference(&cpu_out, &gpu_out));


    _test_print_a_few_elements(&cpu_out, "cpu", 10, true);
    _test_print_a_few_elements(&cpu_out, "cpu", 10, false);
    _test_print_a_few_elements(&gpu_out, "gpu", 10, true);
    _test_print_a_few_elements(&gpu_out, "gpu", 10, false);

    _test_free_tensor(&in);
    _test_free_tensor(&ftr);
    _test_free_tensor(&cpu_out);
    _test_free_tensor(&gpu_out);
}

void test_compare_gpu_convolution_speed() {
    cudamat_4d_tensor in, ftr, conv1_out, conv2_out;

    int n = 256;
    int c = 3;
    int h = 128;
    int w = 128;
    int n_ftr = 256;
    int ftr_h = 3;
    int ftr_w = 3;

    cudamat_convolution_descriptor d;
    _test_create_convolution_descriptor(&d, 0, 0, 0, 1, 1);

    _test_create_tensor(&in, n, c, h, w);
    _test_create_tensor(&ftr, n_ftr, c, ftr_h, ftr_w);

    printf("OK[1]\n");

    // _test_fill_tensor_with_rand_positive(&in);
    // _test_fill_tensor_with_rand_positive(&ftr);
    // _test_fill_tensor_with_constant(&ftr, 1);

    _test_fill_tensor_with_toy_data(&in);
    _test_fill_tensor_with_toy_data(&ftr);

    printf("OK[2]\n");
    
    tensor_copy_to_device(&in);
    tensor_copy_to_device(&ftr);

    printf("OK[3]\n");

    clock_t t_start = clock();
    _test_create_tensor(&conv1_out, n, n_ftr, h - ftr_h + 1, w - ftr_w + 1);
    printf("Copy error: %d\n", tensor_copy_to_device(&conv1_out));

    printf("OK[4]\n");


    printf("in->on_device: %d\n", in.on_device);
    printf("ftr->on_device: %d\n", ftr.on_device);
    printf("conv1_out->on_device: %d\n", conv1_out.on_device);

    // printf("Error code: %d\n", tensor_convolve(&in, &ftr, &conv1_out, &d));
    printf("Error code: %d\n", tensor_convolve2(&in, &ftr, &conv1_out, &d));

    printf("OK[5]\n");

    tensor_copy_to_host(&conv1_out);
    clock_t t_end = clock();
    
    printf("conv1 time: %.5f\n", (float)(t_end - t_start) / CLOCKS_PER_SEC);

    t_start = clock();
    _test_create_tensor(&conv2_out, n, n_ftr, h - ftr_h + 1, w - ftr_w + 1);
    printf("Copy error: %d\n", tensor_copy_to_device(&conv2_out));

    printf("OK[6]\n");

    printf("in->on_device: %d\n", in.on_device);
    printf("ftr->on_device: %d\n", ftr.on_device);
    printf("conv2_out->on_device: %d\n", conv2_out.on_device);

    printf("Error code: %d\n", tensor_convolve2(&in, &ftr, &conv2_out, &d));
    // printf("Error code: %d\n", tensor_convolve(&in, &ftr, &conv2_out, &d));

    printf("OK[7]\n");

    tensor_copy_to_host(&conv2_out);
    t_end = clock();
    
    printf("conv2 time: %.5f\n", (float)(t_end - t_start) / CLOCKS_PER_SEC);

    printf("diff: %.8f\n", _test_compute_l2_difference(&conv1_out, &conv2_out));

    _test_print_a_few_elements(&conv1_out, "conv1", 10, true);
    _test_print_a_few_elements(&conv1_out, "conv1", 10, false);
    _test_print_a_few_elements(&conv2_out, "conv2", 10, true);
    _test_print_a_few_elements(&conv2_out, "conv2", 10, false);

    _test_free_tensor(&in);
    _test_free_tensor(&ftr);
    _test_free_tensor(&conv1_out);
    _test_free_tensor(&conv2_out);
}

int main() {
    int board_id = obtainGpuLock();
    cublas_init();

    // test_helper_io();
    // test_data_transfer();
    // test_init();
    // test_convolution();
    // test_convolution_speed();
    test_compare_gpu_convolution_speed();

    cublas_shutdown();
    freeGpuLock(board_id);
    return 0;
}

