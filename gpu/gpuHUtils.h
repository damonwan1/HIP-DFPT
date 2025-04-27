#pragma once
#include <stdlib.h>

extern "C" void hip_device_init_();

void hip_common_buffer_init_();

void hip_common_buffer_free_();

void H_first_begin();

extern "C" void h_begin_();

void h_begin_0_();

extern "C" void h_pass_vars_(
    int *j_coord_,
    int *n_spin_,
    int *l_ylm_max_,
    int *n_basis_local_,
    int *n_matrix_size_,
    int *basis_l_max,
    int *n_points_all_batches,
    int *n_batch_centers_all_batches,
    int *batch_center_all_batches,
    int *ins_idx_all_batches,
    int *batches_batch_i_basis_h,
    double *partition_all_batches,
    double *first_order_H,
    double *local_potential_parts_all_points,
    double *local_first_order_rho_all_batches,
    double *local_first_order_potential_all_batches,
    double *local_dVxc_drho_all_batches,
    double *local_rho_gradient,
    double *first_order_gradient_rho);