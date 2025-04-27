#include "hip/hip_runtime.h"
#ifndef CUDA_INTEGRATE_FIRSTORDER_H
#define CUDA_INTEGRATE_FIRSTORDER_H

#include "gpuMacro.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" void FORTRAN(H_create_gpu)(int batchCount, int total_size_A_dev, int total_size_B_dev, int total_size_C_dev, int n_max_compute_ham);

extern "C" void FORTRAN(prepare_for_vbatched)(
    int *rank,                                                                      // mpi rank for set device
    int *batchId, int *batchCount,                                                  // for malloc and control
    int *n_compute_c, int *n_points, int *n_max_compute_ham, int *n_max_batchsize); // get matrix mnk info

extern "C" void FORTRAN(get_vbatched_array)(
    int *batchId, // for malloc and control
    int *n_max_compute_ham,
    double *contract, // matrix A
    double *wave_t);  // matrix B

extern "C" void FORTRAN(get_map_array)(int *map, int *n_compute_c, int *n_max_compute_ham, int *batchId, int *batchCount);

extern "C" void FORTRAN(perform_vbatched)(int *batchCount, int *n_max_compute_ham, int *first_scf);

extern "C" void FORTRAN(index_back_nbp)(int *n_max_compute_ham, int *batchCount, double *first_order_H, int *mpi_id);

extern "C" void FORTRAN(free_spaces)();

#endif /*CUDA_INTEGRATE_FIRSTORDER_H*/
