#ifndef _CUDAMAT_CUH_
#define _CUDAMAT_CUH_

#define SYNC_THREADS 1

#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9

// added by Yujia Li
#define ERROR_MEMORY_ERROR -10

#ifdef __cplusplus
extern "C" {
#endif

struct cudamat {
    float* data_host;
    float* data_device;
    int on_device;
    int on_host;
    int size[2];
    int is_trans; // 0 or 1
    int owns_data;
};

struct rnd_struct {
    unsigned int* dev_mults;
    unsigned long long* dev_words;
};

bool check_cublas_error();
bool checkCUDAError();

const char* get_last_cuda_error();
int cublas_init();
int cublas_shutdown();
int cuda_set_device(int deviceId);
int init_random(rnd_struct* rnd_state, int seed, const char* cudamatpath);


int get_leading_dimension(cudamat* mat);
int get_nonleading_dimension(cudamat* mat);
void set_transpose(cudamat* mat, int is_trans);
char get_transpose_char(cudamat* mat);
void cuda_sync_threads();


int allocate_device_memory(cudamat* mat);
int copy_to_host(cudamat* mat);
int copy_to_device(cudamat* mat);
int copy_on_device(cudamat* mat1, cudamat* mat2);
int get_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end);
int set_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end);
int copy_transpose(cudamat* source, cudamat* target);
int free_device_memory(cudamat* mat);
int reshape(cudamat* mat, unsigned int m, unsigned int n);
int get_slice(cudamat* source, cudamat* target, unsigned int first_col, unsigned int last_col);
int get_vector_slice(cudamat* source, cudamat* target, unsigned int first_ind, unsigned int last_ind);


void init_from_array(cudamat* mat, float* data, int m, int n);
int init_empty(cudamat* mat, int m, int n);


int fill_with_rand(rnd_struct* rnd_state, cudamat* mat);
int fill_with_randn(rnd_struct* rnd_state, cudamat* mat);


int add_col_vec(cudamat* mat, cudamat* vec, cudamat* target);
int add_col_mult(cudamat* mat, cudamat* vec, cudamat* target, float mult);
int add_row_vec(cudamat* mat, cudamat* vec, cudamat* target);
int mult_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target);
int mult_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target);
int divide_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target);
int divide_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target);
int less_than(cudamat* mat1, cudamat* mat2, cudamat* target);
int less_than_scalar(cudamat* mat, float val, cudamat* target);
int greater_than(cudamat* mat1, cudamat* mat2, cudamat* target);
int greater_than_scalar(cudamat* mat, float val, cudamat* target);
int equals(cudamat* mat1, cudamat* mat2, cudamat* target);
int equals_scalar(cudamat* mat, float val, cudamat* target);
int minimum(cudamat* mat1, cudamat* mat2, cudamat* target);
int minimum_scalar(cudamat* mat, float val, cudamat* target);
int maximum(cudamat* mat1, cudamat* mat2, cudamat* target);
int maximum_scalar(cudamat* mat, float val, cudamat* target);
int min_by_axis(cudamat* mat, cudamat* target, int axis);
int max_by_axis(cudamat* mat, cudamat* target, int axis);
int argmin_by_axis(cudamat* mat, cudamat* target, int axis);
int argmax_by_axis(cudamat* mat, cudamat* target, int axis);
int sign(cudamat* mat, cudamat* target);
int apply_sigmoid(cudamat* mat, cudamat* target);
int apply_tanh(cudamat* mat, cudamat* target);
int apply_soft_threshold(cudamat* mat, float alpha, cudamat* target);
int apply_abs(cudamat* mat, cudamat* target);
int apply_log_1_plus_exp(cudamat* mat, cudamat* target);
int apply_log(cudamat* mat, cudamat* target);
int apply_exp(cudamat* mat, cudamat* target);
int apply_gamma(cudamat* mat, cudamat* target);
int apply_lgamma(cudamat* mat, cudamat* target);
int apply_sqrt(cudamat* mat, cudamat* target);
int apply_pow(cudamat* mat, float pow, cudamat* target);
int apply_pow_matrix(cudamat* mat, cudamat* pow, cudamat* target);
int reciprocal(cudamat* mat, cudamat* target);
int dot(cudamat* mat1, cudamat* mat2, cudamat* target, float beta, float alpha);
float vdot(cudamat* mat1, cudamat* mat2, int* err_code);
int add_mult(cudamat* mat1, cudamat* mat2, float alpha);
int add_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int subtract_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int divide_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int mult_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int assign_scalar(cudamat* mat, float alpha);
int mult_by_scalar(cudamat* mat, float alpha, cudamat* target);
int divide_by_scalar(cudamat* mat, float alpha, cudamat* target);
int add_scalar(cudamat* mat, float alpha, cudamat* target);
float euclid_norm(cudamat* mat, int* err_code);
float manhattan_norm(cudamat* mat, int* err_code);
int selectRows(cudamat* source, cudamat* target, cudamat* indices);
int setSelectedRows(cudamat* target, cudamat* source, cudamat* indices);
int where(cudamat* condition_mat, cudamat* if_mat, cudamat* else_mat, cudamat* target);


#ifdef __cplusplus
}
#endif

#endif // _CUDAMAT_CUH_

