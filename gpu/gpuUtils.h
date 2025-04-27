#pragma once
#include <stdlib.h>

extern "C" void hip_init_();

extern "C" void hip_device_init_();

extern "C" void hip_device_finish_();

void hip_common_buffer_init_();

void hip_common_buffer_free_();

void sum_up_first_begin();

extern "C" void sum_up_begin_();

void sum_up_final_end();

void rho_first_begin();

void H_first_begin();

extern "C" void rho_begin_();

extern "C" void h_begin_();

extern "C" void pre_run_();

void h_begin_0_();

extern "C" void rho_pass_vars_(
    int *l_ylm_max,
    int *n_local_matrix_size,
    int *n_basis_local,
    int *perm_n_full_points,
    int *first_order_density_matrix_size,
    int *basis_l_max,
    int *n_points_all_batches,
    int *n_batch_centers_all_batches,
    int *batch_center_all_batches,
    int *batch_point_to_i_full_point,
    int *ins_idx_all_batches,
    double *first_order_rho,
    double *first_order_density_matrix,
    double *partition_tab);

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

void set_sum_up_param(int forces_on, double *partition_tab_std, double *delta_v_hartree, double *rho_multipole,
                      // double *centers_rho_multipole_spl, double *centers_delta_v_hart_part_spl,
                      double *adap_outer_radius_sq, double *multipole_radius_sq, int *l_hartree_max_far_distance,
                      double *outer_potential_radius, double *multipole_c);

extern "C" void init_sum_up_c_(int *forces_on, double *partition_tab_std, double *delta_v_hartree, double *rho_multipole,
                               double *adap_outer_radius_sq, double *multipole_radius_sq, int *l_hartree_max_far_distance,
                               double *outer_potential_radius, double *multipole_c);

extern "C" void read_csv_to_map_(const char *filename);

extern "C" void get_gpu_time_(double *batch_times, int *i_my_batch, double *centers);
extern "C" void get_my_id_map_(int *procid, int *n_batches, double *centers);
extern "C" void get_info_(double *centers);
extern "C" void get_merged_batch_weight_(double *batch_desc, int *batch_start, int *batch_nums);
extern "C" void sort_batch_desc_mod_(double *batch_desc_mod, int *len);
extern "C" void output_times_fortran_sumup_(double *time_up, double *time_sum, double *time_all);
extern "C" void output_times_fortran_rho_(double *time_h_all, double *time_hf, double *time_comm, double *time_pre, double *time_others);
extern "C" void output_times_fortran_h_(double *time_h_all, double *time_hf, double *time_comm, double *time_pre, double *time_others);
extern "C" void output_times_fortran_(double *CPU_time,double*H_time);