#include <ctype.h>
#include <hipblas.h>
#include <hip/hip_profile.h>
#include <hip/hip_runtime.h>
#include "gpuError.h"
#include "gpuMacro.h"
#include <time.h>

#include "gpuInterface.h"

/*******************************************************************************
**                              Global Variables                              **
*******************************************************************************/

int nDevices = -1;

ssize_t maxThreads = -1;
ssize_t maxThreadsX = -1;
ssize_t maxThreadsY = -1;

// device specifications to perform parallel calculations
dim3 nBlocks;
dim3 nThreads;

// cublas handle
hipblasHandle_t cublasHandle;

/*******************************************************************************
**                                CUDA CPU Code                               **
*******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                              Memory Management                             //
////////////////////////////////////////////////////////////////////////////////

hipblasStatus_t aimsSetVector(
      const ssize_t dim,
      const int elemSize,
      const void* CPU,
      const int ld1,
      void* GPU,
      const int ld2,
      const char* info)
{
   hipblasStatus_t error;
   CHECK_INFO(dim > 0, info);
   CHECK_INFO(elemSize > 0, info);
   CHECK_INFO(CPU != NULL, info);
   CHECK_INFO(GPU != NULL, info);
   CHECK_INFO(ld1 > 0, info);
   CHECK_INFO(ld2 > 0, info);

   error = hipblasSetVector(dim, elemSize, CPU, ld1, GPU, ld2);
   return error;
}

hipblasStatus_t aimsSetMatrix(
      const ssize_t rows,
      const ssize_t cols,
      const int elemSize,
      const void* CPU,
      const int ld1,
      void* GPU,
      const int ld2,
      const char* info)
{
   hipblasStatus_t error;
   ssize_t dim = rows * cols;
   CHECK_INFO(dim > 0, info);
   CHECK_INFO(elemSize > 0, info);
   CHECK_INFO(CPU != NULL, info);
   CHECK_INFO(GPU != NULL, info);
   CHECK_INFO(ld1 > 0, info);
   CHECK_INFO(ld2 > 0, info);

   error = hipblasSetMatrixAsync(rows, cols, elemSize, CPU, ld1, GPU, ld2, 0);
   return error;
}

hipblasStatus_t aimsGetVector(
      const ssize_t dim,
      const int elemSize,
      const void* GPU,
      const int ld1,
      void* CPU,
      const int ld2,
      const char* info)
{
   hipblasStatus_t error;
   CHECK_INFO(dim > 0, info);
   CHECK_INFO(elemSize > 0, info);
   CHECK_INFO(GPU != NULL, info);
   CHECK_INFO(CPU != NULL, info);
   CHECK_INFO(ld1 > 0, info);
   CHECK_INFO(ld2 > 0, info);

   error = hipblasGetVector(dim, elemSize, GPU, ld1, CPU, ld2);
   return error;
}

hipblasStatus_t aimsGetMatrix(
      const ssize_t rows,
      const ssize_t cols,
      const int elemSize,
      const void* GPU,
      const int ld1,
      void* CPU,
      const int ld2,
      const char* info)
{
   hipblasStatus_t error;
   ssize_t dim = rows * cols;
   CHECK_INFO(dim > 0,info);
   CHECK_INFO(elemSize > 0, info);
   CHECK_INFO(GPU != NULL, info);
   CHECK_INFO(CPU != NULL, info);
   CHECK_INFO(ld1 > 0, info);
   CHECK_INFO(ld2 > 0, info);

   error = hipblasGetMatrixAsync(rows, cols, elemSize, GPU, ld1, CPU, ld2, 0);
   return error;
}

////////////////////////////////////////////////////////////////////////////////
//                             Device Management                              //
////////////////////////////////////////////////////////////////////////////////

void FORTRAN(initialize_cuda_and_cublas)(
      int *dev_id)
{
   CHECK_VAR(*dev_id > -1, *dev_id);
   HANDLE_CUDA(hipSetDevice(*dev_id));

   // explicitly clean up GPU
   HANDLE_CUDA(hipDeviceReset());

   HANDLE_CUBLAS(hipblasCreate(&cublasHandle));

   hipDeviceProp_t *propPtr =
         (hipDeviceProp_t*) malloc (sizeof(hipDeviceProp_t));

   // get Properties
   HANDLE_CUDA(hipGetDeviceProperties(propPtr,*dev_id));

   /*
   maxThreads = propPtr->maxThreadsDim[0];
   maxThreadsX = sqrt(propPtr->maxThreadsDim[0]);
   maxThreadsY = sqrt(propPtr->maxThreadsDim[0]);
   */
   maxThreads = 32;
   maxThreadsX = 32;
   maxThreadsY = 32;

   free (propPtr);
}

void FORTRAN(set_gpu)(
      int *dev_id)
{
   HANDLE_CUDA(hipSetDevice (*dev_id));
}

void FORTRAN(finalize_cuda_and_cublas)()
{
   HANDLE_CUBLAS(hipblasDestroy(cublasHandle));

   HANDLE_CUDA(hipDeviceReset());
}

////////////////////////////////////////////////////////////////////////////////
//                          Device Information                                //
////////////////////////////////////////////////////////////////////////////////

void FORTRAN(get_num_gpus)(
      int *result)
{
   HANDLE_CUDA(hipGetDeviceCount(result));
   nDevices = *result;
}

void FORTRAN(get_gpu_specs)(
      int *dev_id,
      int *major,
      int *minor,
      int *total_global_mem,
      int *total_constant_mem,
      int *shared_mem_per_block,
      int *clock_rate,
      int *multiprocessor_count,
      int *max_threads_per_multiprocessor,
      char* name)
{
   CHECK (dev_id != NULL);

   hipDeviceProp_t *propPtr
         = (hipDeviceProp_t*) malloc (sizeof(hipDeviceProp_t));

   // get Properties
   HANDLE_CUDA(hipGetDeviceProperties(propPtr,*dev_id));

   // map CUDA device properties to own struct
   *major = propPtr->major;
   *minor = propPtr->minor;
   *total_global_mem = static_cast<int>(propPtr->totalGlobalMem);
   *total_constant_mem = static_cast<int>(propPtr->totalConstMem);
   *shared_mem_per_block = static_cast<int>(propPtr->sharedMemPerBlock);
   *shared_mem_per_block = static_cast<int>(propPtr->sharedMemPerBlock);
   *multiprocessor_count = propPtr->multiProcessorCount;
   *max_threads_per_multiprocessor = propPtr->maxThreadsPerMultiProcessor;
   *clock_rate = static_cast<int>(propPtr->clockRate);
   for (int i = 0; i < 256; i++) name[i] = propPtr->name[i];

   free(propPtr);
}
