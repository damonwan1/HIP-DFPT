#pragma once

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
    double *contract__, double *wave_t__, double *first_order_H_dense__, int max_n_batch_centers
    // test
    // int *i_my_batch_                         // test
);
