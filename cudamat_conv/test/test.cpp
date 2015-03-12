#include <cstdio>

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
    cudamat_4d_tensor in, ftr, out;

    _test_create_tensor(&in, 1, 1, 4, 5);
    _test_create_tensor(&ftr, 1, 1, 3, 3);

    _test_fill_tensor_with_toy_data(&in);
    _test_fill_tensor_with_toy_data(&ftr);

    _test_print_small_tensor(&in, "in");
    _test_print_small_tensor(&ftr, "filter");

    _test_tensor_convolve(&in, &ftr, &out);

    _test_print_small_tensor(&out, "out");

    _test_free_tensor(&in);
    _test_free_tensor(&ftr);
    _test_free_tensor(&out);
}

int main() {
    int board_id = obtainGpuLock();
    cublas_init();

    // test_helper_io();
    // test_data_transfer();
    // test_init();
    test_convolution();

    cublas_shutdown();
    freeGpuLock(board_id);
    return 0;
}

