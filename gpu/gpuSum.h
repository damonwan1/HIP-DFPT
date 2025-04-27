#pragma once

extern __global__ void sum_up_whole_potential_shanghui_sub_t_(
    int forces_on, double *partition_tab_std, double *delta_v_hartree, double *rho_multipole,
    double *centers_rho_multipole_spl, double *centers_delta_v_hart_part_spl,
    double *adap_outer_radius_sq, double *multipole_radius_sq, int *l_hartree_max_far_distance,
    double *outer_potential_radius, const double *multipole_c,
    // outer
    // dimensions
    int n_centers_hartree_potential, int n_periodic, int n_max_radial, int l_pot_max, int n_max_spline,
    int n_hartree_grid, int n_species, int n_atoms, int n_centers, int n_max_batch_size, int n_my_batches,
    int n_full_points,
    // runtime_choices
    int use_hartree_non_periodic_ewald, int hartree_fp_function_splines, int fast_ylm, int new_ylm,
    // analytic_multipole_coefficients
    int l_max_analytic_multipole,
    // hartree_potential_real_p0
    int n_hartree_atoms, int hartree_force_l_add,
    // hartree_f_p_functions
    int Fp_max_grid, int lmax_Fp, double Fp_grid_min, double Fp_grid_inc, double Fp_grid_max,
    // outer arrays
    // geometry
    int *species, // 从0开始数，第35个
    // pbc_lists
    int *centers_hartree_potential, int *center_to_atom, int *species_center,
    double *coords_center,
    // species_data
    int *l_hartree,
    // grids
    int *n_grid, int *n_radial, int *batches_size_s, double *batches_points_coords_s,
    double *r_grid_min, double *log_r_grid_inc, double *scale_radial,
    // analytic_multipole_coefficients
    int *n_cc_lm_ijk, int *index_cc, int *index_ijk_max_cc,
    // hartree_potential_real_p0
    double *b0, double *b2, double *b4, double *b6, double *a_save,
    // hartree_f_p_functions
    double *Fp_function_spline_slice, double *Fpc_function_spline_slice,
    // ------ loop helper ------
    int valid_max_point, int *point_to_i_batch, int *point_to_i_index,
    int *valid_point_to_i_full_point,
    const int *index_cc_aos,
    // ------ intermediate ------
    double *Fp_all, double *coord_c_all, double *coord_mat_all, double *rest_mat_all,
    double *vector_all, double *delta_v_hartree_multipole_component_all,
    double *rho_multipole_component_all, double *ylm_tab_all,
    int i_center_begin, int i_center_end, int *i_center_to_centers_index

);

extern __global__ void sum_up_whole_potential_shanghui_pre_proc_(
    // outer parameters
    int n_max_radial, int l_pot_max, int n_max_spline,
    int n_hartree_grid, int n_atoms, int n_max_grid,
    int Adams_Moulton_integrator, int compensate_multipole_errors,
    // outer arrays
    int *species, int *l_hartree, double *multipole_radius_free,
    int *n_grid, int *n_radial, double *r_grid_inc,
    double *scale_radial, double *r_grid, double *r_radial,
    double *rho_multipole, int *rho_multipole_index,
    int *compensation_norm, int *compensation_radius,
    int *centers_hartree_potential, int *center_to_atom,
    // tmp
    double *angular_integral_log_,
    // func spec
    double *rho_multipole_spl, double *delta_v_hartree, int n_coeff_hartree,
    int i_center_begin, int i_center_end, int *i_center_to_centers_index);

extern __global__ void empty_kernel();