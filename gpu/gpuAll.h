#pragma once
extern __global__ void integrate_first_order_rho_sub_tmp2_(
    int l_ylm_max_,
    int n_local_matrix_size_, // 仅适用于使用了 local_index 且实际使用 ins_idx 转换矩阵的版本
    int n_basis_local_,       // 仅适用于使用了 local_index 且实际使用 ins_idx 转换矩阵的版本
    int first_order_density_matrix_size_, int *basis_l_max, int *n_points_all_batches,
    int *n_batch_centers_all_batches, int *batch_center_all_batches,
    int *batch_point_to_i_full_point,
    int *ins_idx_all_batches, // 仅适用于使用了 local_index 且实际使用 ins_idx 转换矩阵的版本
    double *first_order_rho,
    double *first_order_density_matrix_sparse, // first_order_density_matrix 等价于 first_order_density_matrix_sparse
    double *partition_tab,
    // outer nums
    // dimensions num 13
    int n_centers, int n_centers_integrals, int n_max_compute_fns_ham, int n_basis_fns, int n_centers_basis_I,
    int n_max_grid, int n_max_compute_atoms, int n_max_compute_ham, int n_max_compute_dens, int n_max_batch_size,
    // pbc_lists num
    int index_hamiltonian_dim2, int position_in_hamiltonian_dim1, int position_in_hamiltonian_dim2,
    int column_index_hamiltonian_size,
    // rho batch num 27
    int n_my_batches_work_rho, int n_full_points_work_rho, // !!!!!! 记得给这几个值赋值
    // outer arrays 29
    // pbc_lists
    int *center_to_atom, int *species_center, int *center_to_cell, int *cbasis_to_basis,
    int *cbasis_to_center, int *centers_basis_integrals, int *index_hamiltonian,
    int *position_in_hamiltonian, int *column_index_hamiltonian, double *pbc_lists_coords_center,
    // grids
    int *n_grid, double *r_grid_min, double *log_r_grid_inc,
    // basis
    int *perm_basis_fns_spl, double *outer_radius_sq, int *basis_fn, int *basis_l,
    double *atom_radius_sq, int *basis_fn_start_spl, int *basis_fn_atom,
    double *basis_wave_ordered,
    // rho batch 50
    int *batches_size_rho, int *batches_batch_n_compute_rho, int *batches_batch_i_basis_rho,
    double *batches_points_coords_rho,
    // tmp 54
    double *dist_tab_sq__, double *dist_tab__, double *dir_tab__, int *atom_index__, int *atom_index_inv__,
    int *i_basis_fns__, int *i_basis_fns_inv__, int *i_atom_fns__, int *spline_array_start__, int *spline_array_end__,
    double *one_over_dist_tab__, int *rad_index__, int *wave_index__, int *l_index__, int *l_count__, int *fn_atom__,
    int *zero_index_point__, double *wave__, double *first_order_density_matrix_con__, double *i_r__,
    double *trigonom_tab__, double *radial_wave__,
    double *spline_array_aux__, double *aux_radial__,
    double *ylm_tab__, double *dylm_dtheta_tab__, double *scaled_dylm_dphi_tab__, int max_n_batch_centers);

extern __global__ void integrate_first_order_h_sub_tmp2_(
    int j_coord,
    int n_spin,
    int l_ylm_max,
    int n_basis_local,
    int n_matrix_size,
    int *basis_l_max,
    int *n_points_all_batches,
    int *n_batch_centers_all_batches,
    int *batch_center_all_batches,
    int *ins_idx_all_batches,
    int *batches_batch_i_basis_h_not_use__,
    double *partition_all_batches,
    double *first_order_H,
    double *local_potential_parts_all_points,
    double *local_first_order_rho_all_batches,
    double *local_first_order_potential_all_batches,
    double *local_dVxc_drho_all_batches,
    double *local_rho_gradient,
    double *first_order_gradient_rho,
    // outer nums
    // dimensions num 19
    int n_centers, int n_centers_integrals, int n_max_compute_fns_ham, int n_basis_fns, int n_centers_basis_I,
    int n_max_grid, int n_max_compute_atoms, int n_max_compute_ham, int n_max_compute_dens, int n_max_batch_size,
    // pbc_lists num
    int index_hamiltonian_dim2, int position_in_hamiltonian_dim1, int position_in_hamiltonian_dim2,
    int column_index_hamiltonian_size,
    // H batch num
    int n_my_batches_work_h, int n_full_points_work_h,
    // outer arrays 35
    // pbc_lists
    const int *center_to_atom, const int *species_center, const int *center_to_cell, const int *cbasis_to_basis,
    const int *cbasis_to_center, int *centers_basis_integrals, int *index_hamiltonian,
    int *position_in_hamiltonian, int *column_index_hamiltonian, double *pbc_lists_coords_center,
    // grids
    int *n_grid, double *r_grid_min, double *log_r_grid_inc,
    // basis
    const int *perm_basis_fns_spl, const double *outer_radius_sq, const int *basis_fn, const int *basis_l,
    const double *atom_radius_sq, const int *basis_fn_start_spl, const int *basis_fn_atom,
    double *basis_wave_ordered,
    double *basis_kinetic_ordered, // new !!!!
                                   // H batch
    int *batches_batch_n_compute_h, const int *batches_batch_i_basis_h,
    double *batches_points_coords_h,
    // tmp 60
    double *dist_tab_sq__, double *dist_tab__, double *dir_tab__, int *atom_index__, int *atom_index_inv__,
    int *i_basis_fns__, int *i_basis_fns_inv__, int *i_atom_fns__, int *spline_array_start__, int *spline_array_end__,
    double *one_over_dist_tab__, int *rad_index__, int *wave_index__, int *l_index__, int *l_count__, int *fn_atom__,
    int *zero_index_point__, double *wave__, double *first_order_density_matrix_con__, double *i_r__,
    double *trigonom_tab__, double *radial_wave__,
    double *spline_array_aux__, double *aux_radial__,
    double *ylm_tab__, double *dylm_dtheta_tab__, double *scaled_dylm_dphi_tab__,
    // tmp more
    double *kinetic_wave__, double *grid_coord__, double *H_times_psi__, double *T_plus_V__,
    double *contract__, double *wave_t__, double *first_order_H_dense__, int max_n_batch_centers,
    int *new_batch_count, int *new_batch_i_start, int n_new_batch_nums);

    __global__ void integrate_first_order_h_sub_tmp2_pre_(
    int j_coord, int n_spin, int l_ylm_max, int n_basis_local,
    int n_matrix_size, int *basis_l_max,
    int *n_points_all_batches, int *n_batch_centers_all_batches,
    int *batch_center_all_batches, int *ins_idx_all_batches,
    int *batches_batch_i_basis_h_not_use__,
    double *partition_all_batches, double *first_order_H,
    double *local_potential_parts_all_points,
    double *local_first_order_rho_all_batches,
    double *local_first_order_potential_all_batches,
    double *local_dVxc_drho_all_batches,
    double *local_rho_gradient, double *first_order_gradient_rho,
    // outer nums
    // dimensions num 19
    int n_centers, int n_centers_integrals, int n_max_compute_fns_ham,
    int n_basis_fns, int n_centers_basis_I, int n_max_grid,
    int n_max_compute_atoms, int n_max_compute_ham, int n_max_compute_dens,
    int n_max_batch_size,
    // pbc_lists num
    int index_hamiltonian_dim2, int position_in_hamiltonian_dim1,
    int position_in_hamiltonian_dim2, int column_index_hamiltonian_size,
    // H batch num
    int n_my_batches_work_h, int n_full_points_work_h,
    // outer arrays 35
    // pbc_lists
    const int *center_to_atom, const int *species_center,
    const int *center_to_cell, const int *cbasis_to_basis,
    const int *cbasis_to_center, int *centers_basis_integrals,
    int *index_hamiltonian, int *position_in_hamiltonian,
    int *column_index_hamiltonian,
    double *pbc_lists_coords_center,
    // grids
    int *n_grid, double *r_grid_min,
    double *log_r_grid_inc,
    // basis
    const int *perm_basis_fns_spl, const double *outer_radius_sq,
    const int *basis_fn, const int *basis_l,
    const double *atom_radius_sq, const int *basis_fn_start_spl,
    const int *basis_fn_atom, double *basis_wave_ordered,
    double *basis_kinetic_ordered, // new !!!!
                                   // H batch
    int *batches_batch_n_compute_h,
    const int *batches_batch_i_basis_h,
    double *batches_points_coords_h,
    // tmp 60
    double *dist_tab_sq__, double *dist_tab__,
    double *dir_tab__, int *atom_index__,
    int *atom_index_inv__, int *i_basis_fns__,
    int *i_basis_fns_inv__, int *i_atom_fns__,
    int *spline_array_start__, int *spline_array_end__,
    double *one_over_dist_tab__, int *rad_index__,
    int *wave_index__, int *l_index__, int *l_count__,
    int *fn_atom__, int *zero_index_point__,
    double *wave__, double *first_order_density_matrix_con__,
    double *i_r__, double *trigonom_tab__,
    double *radial_wave__, double *spline_array_aux__,
    double *aux_radial__, double *ylm_tab__,
    double *dylm_dtheta_tab__, double *scaled_dylm_dphi_tab__,
    // tmp more
    double *kinetic_wave__, double *grid_coord__,
    double *H_times_psi__, double *T_plus_V__,
    double *contract__, double *wave_t__,
    double *first_order_H_dense__, int max_n_batch_centers, int *new_batch_count, int *new_batch_i_start, int n_new_batch_nums, int *diverge_matrix,
    int i_batch_run);
