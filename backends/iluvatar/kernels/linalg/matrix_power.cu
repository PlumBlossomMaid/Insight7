// backends/cuda/kernels/linalg/matrix_power.cu
/**
 * @file matrix_power.cu
 * @brief CUDA kernel for integer matrix power using exponentiation by squaring.
 */

#include "common.cuh"

static C_Status matrix_power_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int n_exp = *(int *)inputs[2];

  if (!out || !x) {
    gpu_set_last_error("matrix_power: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int size = (int)x->dims[0];
  int32_t dtype = x->dtype;
  int64_t matrix_elems = (int64_t)size * size;
  size_t nbytes = matrix_elems * insight_dtype_size(dtype);

  dim3 threads(16, 16);
  dim3 blocks((size + 15) / 16, (size + 15) / 16);

  // Allocate work buffers: base (copy of x), work (temp)
  void *base_data = nullptr;
  void *work_data = nullptr;
  cudaMalloc(&base_data, nbytes);
  cudaMalloc(&work_data, nbytes);

  // Initialize: result = I, base = x
  if (dtype == INSIGHT_DTYPE_F64) {
    linalg_identity_kernel<double>
        <<<blocks, threads>>>((double *)out->data, size);
    linalg_copy_matrix_kernel<double>
        <<<linalg_blocks(matrix_elems), linalg_threads()>>>(
            (double *)base_data, (const double *)x->data, matrix_elems);

    const double alpha = 1.0, beta = 0.0;
    int exp = n_exp;
    while (exp > 0) {
      if (exp & 1) {
        // result = result * base → work = result, result = work * base
        cudaMemcpy(work_data, out->data, nbytes, cudaMemcpyDeviceToDevice);
        cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                    &alpha, (const double *)base_data, size,
                    (const double *)work_data, size, &beta, (double *)out->data,
                    size);
      }
      // base = base * base
      cudaMemcpy(work_data, base_data, nbytes, cudaMemcpyDeviceToDevice);
      cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                  &alpha, (const double *)work_data, size,
                  (const double *)work_data, size, &beta, (double *)base_data,
                  size);
      exp >>= 1;
    }
  } else {
    linalg_identity_kernel<float>
        <<<blocks, threads>>>((float *)out->data, size);
    linalg_copy_matrix_kernel<float>
        <<<linalg_blocks(matrix_elems), linalg_threads()>>>(
            (float *)base_data, (const float *)x->data, matrix_elems);

    const float alpha = 1.0f, beta = 0.0f;
    int exp = n_exp;
    while (exp > 0) {
      if (exp & 1) {
        cudaMemcpy(work_data, out->data, nbytes, cudaMemcpyDeviceToDevice);
        cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                    &alpha, (const float *)base_data, size,
                    (const float *)work_data, size, &beta, (float *)out->data,
                    size);
      }
      cudaMemcpy(work_data, base_data, nbytes, cudaMemcpyDeviceToDevice);
      cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                  &alpha, (const float *)work_data, size,
                  (const float *)work_data, size, &beta, (float *)base_data,
                  size);
      exp >>= 1;
    }
  }

  cudaFree(base_data);
  cudaFree(work_data);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(matrix_power, INSIGHT_DTYPE_F32,
                         matrix_power_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(matrix_power, INSIGHT_DTYPE_F64,
                         matrix_power_kernel_gpu);
