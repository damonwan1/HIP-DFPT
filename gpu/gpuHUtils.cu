#include "gpuVars.h"
#include "gpuHUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include "gpuHAll.h"

extern int MV(mpi_tasks, myid);
extern int MV(opencl_util, mpi_platform_relative_id);
#define myid MV(mpi_tasks, myid)
#define mpi_platform_relative_id MV(opencl_util, mpi_platform_relative_id)
#define number_of_files 2
#define number_of_kernels 4
#define DEVICE_ID 0
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

size_t platformId = 0; // INFO choose a platform

int numOfGpus;
const char *kernel_names[] = {"sum_up_whole_potential_shanghui_sub_t_", "integrate_first_order_rho_sub_tmp2_", "integrate_first_order_h_sub_tmp2_", "sum_up_whole_potential_shanghui_pre_proc_"};
// "c_integration_points2"};
hipFunction_t kernels[number_of_kernels];
// char *buffer[number_of_files];

size_t localSize[] = {128};        // 可能被重新设置 !!!
size_t globalSize[] = {128 * 128}; // 可能被重新设置 !!!

size_t globalSize_sum_up_pre_proc[1];
size_t localSize_sum_up_pre_proc[1] = {64};
const int i_center_tile_size_default = 256;

extern double *Fp_function_spline_slice;
extern double *Fpc_function_spline_slice;
extern double *Fp;

H_PARAM H_param;

static int hip_init_finished = 0;
static int hip_common_buffer_init_finished = 0;
static int remember_arg_index_1 = -10;
static int remember_arg_Fp_function_spline = -10;
static int remember_arg_n_my_batches_work_sumup = -10;
static int remember_arg_Fp_max_grid = -10;
static int sum_up_first_begin_finished = 0;
static int sum_up_begin_0_finished = 0;
static int rho_first_begin_finished = 0;
static int H_first_begin_finished = 0;
static int h_begin_0_finished = 0;

struct HIP_BUF_COM_T
{
    // hipDeviceptr_t species;
    // hipDeviceptr_t empty;
    // hipDeviceptr_t centers_hartree_potential;
    hipDeviceptr_t center_to_atom;
    hipDeviceptr_t species_center;
    hipDeviceptr_t coords_center;
    // hipDeviceptr_t l_hartree;
    hipDeviceptr_t n_grid;
    // hipDeviceptr_t n_radial;
    hipDeviceptr_t r_grid_min;
    // hipDeviceptr_t r_grid_inc;
    hipDeviceptr_t log_r_grid_inc;
    // hipDeviceptr_t scale_radial;
    hipDeviceptr_t r_radial;
    // hipDeviceptr_t r_grid;
    // hipDeviceptr_t n_cc_lm_ijk;
    // hipDeviceptr_t index_cc;
    // hipDeviceptr_t index_ijk_max_cc;
    // hipDeviceptr_t b0;
    // hipDeviceptr_t b2;
    // hipDeviceptr_t b4;
    // hipDeviceptr_t b6;
    // hipDeviceptr_t a_save;
    hipDeviceptr_t Fp_function_spline_slice;
    hipDeviceptr_t Fpc_function_spline_slice;
    // ---
    // hipDeviceptr_t rho_multipole_index;
    // hipDeviceptr_t compensation_norm;
    // hipDeviceptr_t compensation_radius;
    // hipDeviceptr_t rho_multipole_h_p_s;
    // hipDeviceptr_t multipole_radius_free;
    // rho global
    hipDeviceptr_t perm_basis_fns_spl;
    hipDeviceptr_t outer_radius_sq;
    hipDeviceptr_t basis_fn;
    hipDeviceptr_t basis_l;
    hipDeviceptr_t atom_radius_sq;
    hipDeviceptr_t basis_fn_start_spl;
    hipDeviceptr_t basis_fn_atom;
    hipDeviceptr_t basis_wave_ordered;
    hipDeviceptr_t basis_kinetic_ordered;
    hipDeviceptr_t Cbasis_to_basis;
    hipDeviceptr_t Cbasis_to_center;
    hipDeviceptr_t centers_basis_integrals; // 可能因为宏展开出问题
    hipDeviceptr_t index_hamiltonian;
    hipDeviceptr_t position_in_hamiltonian;
    hipDeviceptr_t column_index_hamiltonian;
    // pbc_lists_coords_center 即 coords_center，只是为了规避一点点重名
    hipDeviceptr_t center_to_cell;
    // // loop helper
    // hipDeviceptr_t point_to_i_batch;
    // hipDeviceptr_t point_to_i_index;
    // hipDeviceptr_t valid_point_to_i_full_point;
    // hipDeviceptr_t index_cc_aos;
    // hipDeviceptr_t i_center_to_centers_index;
    // // sum up
    // // hipDeviceptr_t partition_tab_std;
    // // hipDeviceptr_t delta_v_hartree;               // (n_full_points_work)
    // // hipDeviceptr_t rho_multipole;                 // (n_full_points_work)
    // hipDeviceptr_t centers_rho_multipole_spl;     // (l_pot_max+1)**2, n_max_spline, n_max_radial+2, n_atoms)
    // hipDeviceptr_t centers_delta_v_hart_part_spl; // (l_pot_max+1)**2, n_coeff_hartree, n_hartree_grid, n_atoms)
    // // hipDeviceptr_t adap_outer_radius_sq;          // (n_atoms)
    // // hipDeviceptr_t multipole_radius_sq;           // (n_atoms)
    // // hipDeviceptr_t l_hartree_max_far_distance;    // (n_atoms)
    // // hipDeviceptr_t outer_potential_radius;        // (0:l_pot_max, n_atoms)
    // // hipDeviceptr_t multipole_c;                   // (n_cc_lm_ijk(l_pot_max), n_atoms)
    // // sum up tmp
    // hipDeviceptr_t angular_integral_log; // per block (l_pot_max + 1) * (l_pot_max + 1) * n_max_grid
    // hipDeviceptr_t Fp;                   // global_size * (l_pot_max + 2) * n_centers_hartree_potential)
    // hipDeviceptr_t coord_c;
    // hipDeviceptr_t coord_mat;
    // hipDeviceptr_t rest_mat;
    // hipDeviceptr_t vector;
    // hipDeviceptr_t delta_v_hartree_multipole_component;
    // hipDeviceptr_t rho_multipole_component;
    // hipDeviceptr_t ylm_tab;
    // sum_up batches
    // hipDeviceptr_t batches_size_sumup;
    // hipDeviceptr_t batches_points_coords_sumup;
    // rho tmp
    hipDeviceptr_t dist_tab_sq__;
    hipDeviceptr_t dist_tab__;
    hipDeviceptr_t dir_tab__;
    hipDeviceptr_t atom_index__;
    hipDeviceptr_t atom_index_inv__;
    hipDeviceptr_t i_basis_fns__;
    hipDeviceptr_t i_basis_fns_inv__;
    hipDeviceptr_t i_atom_fns__;
    hipDeviceptr_t spline_array_start__;
    hipDeviceptr_t spline_array_end__;
    hipDeviceptr_t one_over_dist_tab__;
    hipDeviceptr_t rad_index__;
    hipDeviceptr_t wave_index__;
    hipDeviceptr_t l_index__;
    hipDeviceptr_t l_count__;
    hipDeviceptr_t fn_atom__;
    hipDeviceptr_t zero_index_point__;
    hipDeviceptr_t wave__;
    hipDeviceptr_t first_order_density_matrix_con__;
    hipDeviceptr_t i_r__;
    hipDeviceptr_t trigonom_tab__;
    hipDeviceptr_t radial_wave__;
    hipDeviceptr_t spline_array_aux__;
    hipDeviceptr_t aux_radial__;
    hipDeviceptr_t ylm_tab__;
    hipDeviceptr_t dylm_dtheta_tab__;
    hipDeviceptr_t scaled_dylm_dphi_tab__;
    hipDeviceptr_t kinetic_wave__;
    hipDeviceptr_t grid_coord__;
    hipDeviceptr_t H_times_psi__;
    hipDeviceptr_t T_plus_V__;
    hipDeviceptr_t contract__;
    hipDeviceptr_t wave_t__;
    hipDeviceptr_t first_order_H_dense__;
    // // rho batches
    // hipDeviceptr_t batches_size_rho;
    // hipDeviceptr_t batches_batch_n_compute_rho;
    // hipDeviceptr_t batches_batch_i_basis_rho;
    // hipDeviceptr_t batches_points_coords_rho;
    // H batches
    hipDeviceptr_t batches_size_H;
    hipDeviceptr_t batches_batch_n_compute_H;
    hipDeviceptr_t batches_batch_i_basis_H;
    hipDeviceptr_t batches_points_coords_H;
    // rho
    hipDeviceptr_t basis_l_max__;
    hipDeviceptr_t n_points_all_batches__;
    hipDeviceptr_t n_batch_centers_all_batches__;
    hipDeviceptr_t batch_center_all_batches__;
    hipDeviceptr_t batch_point_to_i_full_point__;
    hipDeviceptr_t ins_idx_all_batches__;
    hipDeviceptr_t first_order_rho__;
    hipDeviceptr_t first_order_density_matrix__;
    hipDeviceptr_t partition_tab__;
    // hipDeviceptr_t tmp_rho__; // only for swcl
    // H
    hipDeviceptr_t batches_batch_i_basis_h__;
    hipDeviceptr_t partition_all_batches__;
    hipDeviceptr_t first_order_H__;
    hipDeviceptr_t local_potential_parts_all_points__;
    hipDeviceptr_t local_first_order_rho_all_batches__;
    hipDeviceptr_t local_first_order_potential_all_batches__;
    hipDeviceptr_t local_dVxc_drho_all_batches__;
    hipDeviceptr_t local_rho_gradient__;
    hipDeviceptr_t first_order_gradient_rho__;
} hip_buf_com;

static int H_pass_vars_count = -1;
void h_pass_vars_(
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
    double *first_order_gradient_rho)
{
    H_pass_vars_count++;

    H_param.j_coord = *j_coord_;
    H_param.n_spin = *n_spin_;
    H_param.l_ylm_max = *l_ylm_max_;
    H_param.n_basis_local = *n_basis_local_;
    H_param.n_matrix_size = *n_matrix_size_;
    H_param.basis_l_max = basis_l_max;
    H_param.n_points_all_batches = n_points_all_batches;
    H_param.n_batch_centers_all_batches = n_batch_centers_all_batches;
    H_param.batch_center_all_batches = batch_center_all_batches;
    H_param.ins_idx_all_batches = ins_idx_all_batches;
    H_param.batches_batch_i_basis_h = batches_batch_i_basis_h;
    H_param.partition_all_batches = partition_all_batches;
    H_param.first_order_H = first_order_H;
    H_param.local_potential_parts_all_points = local_potential_parts_all_points;
    H_param.local_first_order_rho_all_batches = local_first_order_rho_all_batches;
    H_param.local_first_order_potential_all_batches = local_first_order_potential_all_batches;
    H_param.local_dVxc_drho_all_batches = local_dVxc_drho_all_batches;
    H_param.local_rho_gradient = local_rho_gradient;
    H_param.first_order_gradient_rho = first_order_gradient_rho;
    // char save_file_name[64];
    // if ((myid == 0 || myid == 3) && H_pass_vars_count <= 4)
    // {
    //     sprintf(save_file_name, "mdata_outer_rank%d_%d.bin", myid, H_pass_vars_count);
    //     m_save_load_H(save_file_name, 0, 0);
    // }
}

#define IF_ERROR_EXIT(cond, err_code, str)                                                                              \
    if (cond)                                                                                                           \
    {                                                                                                                   \
        printf("Error! rank%d, %s:%d, %s\nError_code=%d\n%s\n", myid, __FILE__, __LINE__, __FUNCTION__, err_code, str); \
        fflush(stdout);                                                                                                 \
        exit(-1);                                                                                                       \
    }

#define _CHK_(size1, size2)                                                           \
    if ((size1) != (size2))                                                           \
    {                                                                                 \
        printf("Error! rank%d, %s:%d, %s\n", myid, __FILE__, __LINE__, __FUNCTION__); \
        fflush(stdout);                                                               \
        exit(-1);                                                                     \
    }
#define _NL_ status = func(&endl, sizeof(char), 1, file_p);
#define _FWD_(type, var)                              \
    status = func(&var, sizeof(type), 1, file_p);     \
    if (print)                                        \
    {                                                 \
        printf("rank%d, %s = %d\n", myid, #var, var); \
    }

#define _FW_(type, var, size, var_out, hip_mem_flag)                                                 \
    if (hip_mem_flag == hipMemcpyHostToDevice)                                                       \
    {                                                                                                \
        hipError_t hip_err = hipMalloc(&hip_buf_com.var_out, sizeof(type) * (size));                 \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMalloc failed");                           \
        hip_err = hipMemcpy(hip_buf_com.var_out, var, sizeof(type) * (size), hipMemcpyHostToDevice); \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMemcpy failed");                           \
    }                                                                                                \
    else if (hip_mem_flag == 0)                                                                      \
    {                                                                                                \
        hipError_t hip_err = hipMalloc(&hip_buf_com.var_out, sizeof(type) * (size));                 \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipMalloc failed");                           \
    }                                                                                                \
    else if (hip_mem_flag == 1)                                                                      \
    {                                                                                                \
        hipError_t hip_err = hipHostRegister(var, sizeof(type) * (size), hipHostRegisterDefault);    \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipHostRegister failed");                     \
        hip_err = hipHostGetDevicePointer((void **)&hip_buf_com.var_out, var, 0);                    \
        IF_ERROR_EXIT(hip_err != hipSuccess, hip_err, "hipHostGetDevicePointer failed");             \
    }                                                                                                \
    else                                                                                             \
    {                                                                                                \
        fprintf(stderr, "Invalid hip memory flag");                                                  \
        exit(EXIT_FAILURE);                                                                          \
    }

#define _FWV_(type, var, size, var_out, cl_mem_flag)                                    \
    var_out = clCreateBuffer(context, cl_mem_flag, sizeof(type) * (size), var, &error); \
    IF_ERROR_EXIT(error != CL_SUCCESS, error, "clCreateBuffer failed");

#define hipSetupArgumentWithCheck(kernel_name, arg_index, off_set, arg_size, arg_value)                      \
    {                                                                                                        \
        hipError_t error = hipSetupArgument(arg_value, arg_size, off_set);                                   \
        off_set = off_set + arg_size;                                                                        \
        if (error != hipSuccess)                                                                             \
        {                                                                                                    \
            fprintf(stderr, "hipSetupArgument failed at arg %d: %s\n", arg_index, hipGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    }

#define setKernelArgs(args, arg_index, arg_size, arg_value) \
    {                                                       \
        args[arg_index] = arg_value;                        \
    }

void hip_device_init_()
{
    if (hip_init_finished)
        return;
    hip_init_finished = 1;
    hipError_t error;
    // 得到平台数量
    error = hipGetDeviceCount(&numOfGpus);
    IF_ERROR_EXIT(error != hipSuccess, error, "Unable to find any GPUs");
    printf("Number of GPUs: %d\n", numOfGpus);
    // int device_id = DEVICE_ID;
    // 选择指定平台上的指定设备
    int device_id = mpi_platform_relative_id / 8;
    // int device_id = myid % 4;
    IF_ERROR_EXIT(numOfGpus <= device_id, numOfGpus, "The selected platformId is out of range");
    error = hipSetDevice(device_id);
    IF_ERROR_EXIT(error != hipSuccess, error, "Unable to set GPU");
}

// void opencl_finish()
// {
//     if (!opencl_init_finished)
//         return;
//     opencl_init_finished = 0;
//     clReleaseCommandQueue(cQ);
//     for (int i = 0; i < number_of_kernels; i++)
//         clReleaseKernel(kernels[i]);
//     for (int i = 0; i < number_of_files; i++)
//         free(buffer[i]);
//     clReleaseProgram(program);
//     clReleaseContext(context);
// }

void hip_common_buffer_init_()
{
    if (hip_common_buffer_init_finished)
        return;
    hip_common_buffer_init_finished = 1;

    // _FW_(int, MV(geometry, species), n_atoms, species, hipMemcpyHostToDevice);
    // _FW_(int, MV(pbc_lists, centers_hartree_potential), n_centers_hartree_potential, centers_hartree_potential,
    //      hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, center_to_atom), n_centers, center_to_atom, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, species_center), n_centers, species_center, hipMemcpyHostToDevice);
    _FW_(double, MV(pbc_lists, coords_center), 3 * n_centers, coords_center, hipMemcpyHostToDevice);
    // _FW_(int, MV(species_data, l_hartree), n_species, l_hartree, hipMemcpyHostToDevice);
    _FW_(int, MV(grids, n_grid), n_species, n_grid, hipMemcpyHostToDevice);
    // _FW_(int, MV(grids, n_radial), n_species, n_radial, hipMemcpyHostToDevice);

    _FW_(double, MV(grids, r_grid_min), n_species, r_grid_min, hipMemcpyHostToDevice);
    // _FW_(double, MV(grids, r_grid_inc), n_species, r_grid_inc, hipMemcpyHostToDevice);
    _FW_(double, MV(grids, log_r_grid_inc), n_species, log_r_grid_inc, hipMemcpyHostToDevice);
    // _FW_(double, MV(grids, scale_radial), n_species, scale_radial, hipMemcpyHostToDevice);
    // _FW_(int, MV(analytic_multipole_coefficients, n_cc_lm_ijk), (l_max_analytic_multipole + 1), n_cc_lm_ijk,
    //      hipMemcpyHostToDevice);
    // _FW_(int, MV(analytic_multipole_coefficients, index_cc), n_cc_lm_ijk(l_max_analytic_multipole) * 6, index_cc,
    //      hipMemcpyHostToDevice);
    // _FW_(int, MV(analytic_multipole_coefficients, index_ijk_max_cc), 3 * (l_max_analytic_multipole + 1), index_ijk_max_cc,
    //      hipMemcpyHostToDevice);
    // _FW_(double, MV(hartree_potential_real_p0, b0), pmaxab + 1, b0, hipMemcpyHostToDevice);
    // _FW_(double, MV(hartree_potential_real_p0, b2), pmaxab + 1, b2, hipMemcpyHostToDevice);
    // _FW_(double, MV(hartree_potential_real_p0, b4), pmaxab + 1, b4, hipMemcpyHostToDevice);
    // _FW_(double, MV(hartree_potential_real_p0, b6), pmaxab + 1, b6, hipMemcpyHostToDevice);
    // _FW_(double, MV(hartree_potential_real_p0, a_save), pmaxab + 1, a_save, hipMemcpyHostToDevice);
    // // _FW_(double, Fp_function_spline_slice, (lmax_Fp + 1) * 4 * (Fp_max_grid+1), Fp_function_spline_slice,
    // //      hipMemcpyHostToDevice);
    // // _FW_(double, Fpc_function_spline_slice, (lmax_Fp + 1) * 4 * (Fp_max_grid+1), Fpc_function_spline_slice,
    // //      hipMemcpyHostToDevice);
    // rho global
    _FW_(int, MV(basis, perm_basis_fns_spl), n_basis_fns, perm_basis_fns_spl, hipMemcpyHostToDevice);
    _FW_(double, MV(basis, outer_radius_sq), n_basis_fns, outer_radius_sq, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_fn), n_basis, basis_fn, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_l), n_basis, basis_l, hipMemcpyHostToDevice);
    _FW_(double, MV(basis, atom_radius_sq), n_species, atom_radius_sq, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_fn_start_spl), n_species, basis_fn_start_spl, hipMemcpyHostToDevice);
    _FW_(int, MV(basis, basis_fn_atom), n_basis_fns *n_atoms, basis_fn_atom, hipMemcpyHostToDevice);
    _FW_(double, MV(basis, basis_wave_ordered), n_basis_fns *n_max_spline *n_max_grid, basis_wave_ordered, hipMemcpyHostToDevice);       // 进程间可能不同
    _FW_(double, MV(basis, basis_kinetic_ordered), n_basis_fns *n_max_spline *n_max_grid, basis_kinetic_ordered, hipMemcpyHostToDevice); // 进程间可能不同

    _FW_(int, MV(pbc_lists, cbasis_to_basis), n_centers_basis_T, Cbasis_to_basis, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, cbasis_to_center), n_centers_basis_T, Cbasis_to_center, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, centers_basis_integrals), n_centers_basis_integrals, centers_basis_integrals, hipMemcpyHostToDevice);
    _FW_(int, MV(pbc_lists, index_hamiltonian), 2 * index_hamiltonian_dim2 * n_basis, index_hamiltonian, hipMemcpyHostToDevice);                                   // 进程间可能不同
    _FW_(int, MV(pbc_lists, position_in_hamiltonian), position_in_hamiltonian_dim1 *position_in_hamiltonian_dim2, position_in_hamiltonian, hipMemcpyHostToDevice); // 进程间可能不同
    _FW_(int, MV(pbc_lists, column_index_hamiltonian), column_index_hamiltonian_size, column_index_hamiltonian, hipMemcpyHostToDevice);                            // 进程间可能不同
    _FW_(int, MV(pbc_lists, center_to_cell), n_centers, center_to_cell, hipMemcpyHostToDevice);

    // if(ctrl_use_sumup_pre_c_cl_version){
    _FW_(double, MV(grids, r_radial), n_max_radial *n_species, r_radial, hipMemcpyHostToDevice);
    // _FW_(double, MV(grids, r_grid), n_max_grid *n_species, r_grid, hipMemcpyHostToDevice);
    // _FW_(int, NULL, n_atoms, rho_multipole_index, CL_MEM_READ_ONLY);
    // if (compensate_multipole_errors)
    // {
    //     _FW_(int, MV(hartree_potential_storage, compensation_norm), n_atoms, compensation_norm, hipMemcpyHostToDevice);
    //     _FW_(int, MV(hartree_potential_storage, compensation_radius), n_atoms, compensation_radius, hipMemcpyHostToDevice);
    // }
    // else
    // {
    //     _FW_(int, NULL, 1, compensation_norm, 0);
    //     _FW_(int, NULL, 1, compensation_radius, 0);
    // }

    // _FW_(double, MV(species_data, multipole_radius_free), n_species, multipole_radius_free, hipMemcpyHostToDevice);
    // }
    // ------
}

void hip_common_buffer_free_()
{
    if (!hip_common_buffer_init_finished)
        return;
    hip_common_buffer_init_finished = 0;

    unsigned int arg_index = 0;

    // hipFree(hip_buf_com.species);
    // hipFree(hip_buf_com.centers_hartree_potential);
    hipFree(hip_buf_com.center_to_atom);
    hipFree(hip_buf_com.species_center);
    hipFree(hip_buf_com.coords_center);
    // hipFree(hip_buf_com.l_hartree);
    hipFree(hip_buf_com.n_grid);
    // hipFree(hip_buf_com.n_radial);

    hipFree(hip_buf_com.r_grid_min);
    // hipFree(hip_buf_com.r_grid_inc);
    hipFree(hip_buf_com.log_r_grid_inc);
    // hipFree(hip_buf_com.scale_radial);
    // hipFree(hip_buf_com.n_cc_lm_ijk);
    // hipFree(hip_buf_com.index_cc);
    // hipFree(hip_buf_com.index_ijk_max_cc);
    // hipFree(hip_buf_com.b0);
    // hipFree(hip_buf_com.b2);
    // hipFree(hip_buf_com.b4);
    // hipFree(hip_buf_com.b6);
    // hipFree(hip_buf_com.a_save);

    // rho global
    hipFree(hip_buf_com.perm_basis_fns_spl);
    hipFree(hip_buf_com.outer_radius_sq);
    hipFree(hip_buf_com.basis_fn);
    hipFree(hip_buf_com.basis_l);
    hipFree(hip_buf_com.atom_radius_sq);
    hipFree(hip_buf_com.basis_fn_start_spl);
    hipFree(hip_buf_com.basis_fn_atom);
    hipFree(hip_buf_com.basis_wave_ordered);
    hipFree(hip_buf_com.basis_kinetic_ordered);

    hipFree(hip_buf_com.Cbasis_to_basis);
    hipFree(hip_buf_com.Cbasis_to_center);
    hipFree(hip_buf_com.centers_basis_integrals);
    hipFree(hip_buf_com.index_hamiltonian);
    hipFree(hip_buf_com.position_in_hamiltonian);
    hipFree(hip_buf_com.column_index_hamiltonian);

    hipFree(hip_buf_com.center_to_cell);

    // if(ctrl_use_sumup_pre_c_cl_version){
    // hipFree(hip_buf_com.r_radial);
    // hipFree(hip_buf_com.r_grid);
    // hipFree(hip_buf_com.rho_multipole_index);
    // hipFree(hip_buf_com.compensation_norm);
    // hipFree(hip_buf_com.compensation_radius);
    // // hipFree(hip_buf_com.rho_multipole_h_p_s);
    // hipFree(hip_buf_com.multipole_radius_free);
    // }
    // ------
}

// ok
void H_first_begin()
{
    if (H_first_begin_finished)
        return;
    H_first_begin_finished = 1;
    hip_device_init_();        // ok may be wrong !
    hip_common_buffer_init_(); // ok
    // size_t localSize[] = {256};        // 覆盖前面的设置
    // size_t globalSize[] = {256 * 128}; // 覆盖前面的设置
    // dim3 blockDim(localSize[0], 1, 1);
    // dim3 gridDim(globalSize[0] / blockDim.x, 1, 1);
    // error = hipConfigureCall(gridDim, blockDim, 0, 0);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipConfigureCall failed");
    // int fast_ylm = 1;
    // int new_ylm = 0;
}

// ok
void h_begin_0_()
{
    if (h_begin_0_finished)
        return;
    h_begin_0_finished = 1;

    H_first_begin();
    H_first_begin_finished = 0;

    hipError_t error;

    // H param
    _FW_(int, H_param.basis_l_max, n_species, basis_l_max__, hipMemcpyHostToDevice);
    _FW_(int, H_param.n_points_all_batches, n_my_batches_work_h, n_points_all_batches__, hipMemcpyHostToDevice);
    _FW_(int, H_param.n_batch_centers_all_batches, n_my_batches_work_h, n_batch_centers_all_batches__, hipMemcpyHostToDevice);
    _FW_(int, H_param.batch_center_all_batches, max_n_batch_centers *n_my_batches_work_h, batch_center_all_batches__, hipMemcpyHostToDevice);
    // _FW_(int, H_param.batch_point_to_i_full_point, n_max_batch_size * n_my_batches_work_h, batch_point_to_i_full_point__, hipMemcpyHostToDevice);
    if (H_param.n_basis_local > 0)
    {
        _FW_(int, H_param.ins_idx_all_batches, H_param.n_basis_local *n_my_batches_work_h, ins_idx_all_batches__, hipMemcpyHostToDevice);
    }
    else
    {
        _FW_(int, NULL, 1, ins_idx_all_batches__, 0); // 废弃
    }
    // _FW_(int, H_param.batches_batch_i_basis_h, (n_centers_basis_I * n_my_batches_work_h), batches_batch_i_basis_h__, 0 | CL_MEM_COPY_HOST_PTR);  // 输出
    // _FW_(int, H_param.batches_batch_i_basis_h, 1, batches_batch_i_basis_h__, 0 | CL_MEM_COPY_HOST_PTR);  // 输出
    _FW_(int, NULL, 1, batches_batch_i_basis_h__, 0);
    _FW_(double, H_param.partition_all_batches, (n_max_batch_size * n_my_batches_work_h), partition_all_batches__, hipMemcpyHostToDevice);
    _FW_(double, H_param.first_order_H, (H_param.n_matrix_size * H_param.n_spin), first_order_H__, hipMemcpyHostToDevice);
    // _FW_(double, H_param.local_potential_parts_all_points, (H_param.n_spin * n_full_points_work_h), local_potential_parts_all_points__, hipMemcpyHostToDevice);
    _FW_(double, H_param.local_potential_parts_all_points, 1, local_potential_parts_all_points__, hipMemcpyHostToDevice);
    _FW_(double, H_param.local_first_order_rho_all_batches, (H_param.n_spin * n_max_batch_size * n_my_batches_work_h), local_first_order_rho_all_batches__, 1);
    _FW_(double, H_param.local_first_order_potential_all_batches, (n_max_batch_size * n_my_batches_work_h), local_first_order_potential_all_batches__, 1);
    _FW_(double, H_param.local_dVxc_drho_all_batches, (3 * n_max_batch_size * n_my_batches_work_h), local_dVxc_drho_all_batches__, 1);
    _FW_(double, H_param.local_rho_gradient, (3 * H_param.n_spin * n_max_batch_size), local_rho_gradient__, hipMemcpyHostToDevice);
    _FW_(double, H_param.first_order_gradient_rho, (3 * H_param.n_spin * n_max_batch_size), first_order_gradient_rho__, hipMemcpyHostToDevice);

    // H batches
    _FW_(int, MV(opencl_util, batches_size_h), n_my_batches_work_h, batches_size_H, 1);
    _FW_(int, MV(opencl_util, batches_batch_n_compute_h), n_my_batches_work_h, batches_batch_n_compute_H, 1);
    _FW_(int, MV(opencl_util, batches_batch_i_basis_h), n_max_compute_dens *n_my_batches_work_h, batches_batch_i_basis_H, 1);
    _FW_(double, MV(opencl_util, batches_points_coords_h), 3 * n_max_batch_size * n_my_batches_work_h, batches_points_coords_H, 1);
}

// ok
void h_begin_()
{
    h_begin_0_();
    h_begin_0_finished = 0;

    // long time_uses[32];
    // char *time_infos[32];
    // size_t time_index = 0;
    long time_uses[2];
    struct timeval start, end, start1, end1;
    gettimeofday(&start, NULL);

    hipError_t error;
    int arg_index;
    size_t localSize[] = {256};        // 覆盖前面的设置
    size_t globalSize[] = {256 * 128}; // 覆盖前面的设置
    dim3 blockDim(localSize[0], 1, 1);
    dim3 gridDim(globalSize[0] / blockDim.x, 1, 1);
    // printf("n_max_batch_size=%d\n", n_max_batch_size);
    // H tmp
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), dist_tab_sq__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), dist_tab__, 0);
    _FW_(double, NULL, globalSize[0] * (3 * n_max_compute_atoms), dir_tab__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), atom_index__, 0);
    // _FW_(int, NULL, globalSize[0] * (n_centers_integrals), atom_index_inv__, 0);
    _FW_(int, NULL, 1, atom_index_inv__, 0);
    _FW_(int, NULL, 1, i_basis_fns__, 0);
    _FW_(int, NULL, globalSize[0] * (n_basis_fns * (n_max_compute_atoms + 1)), i_basis_fns_inv__, 0);
    _FW_(int, NULL, 1, i_atom_fns__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), spline_array_start__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_atoms), spline_array_end__, 0);
    _FW_(double, NULL, globalSize[0] * n_max_compute_atoms, one_over_dist_tab__, 0);
    _FW_(int, NULL, globalSize[0] * n_max_compute_atoms, rad_index__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), wave_index__, 0);
    _FW_(int, NULL, 1, l_index__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), l_count__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_fns_ham), fn_atom__, 0);
    _FW_(int, NULL, globalSize[0] * (n_max_compute_ham), zero_index_point__, 0);
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) * ((n_max_compute_ham + 127) / 128 * 128) + 256 + 16 * n_max_compute_ham,
         wave__, 0); // 多加 128 为了避免 TILE 后越界, 长宽按 128 对齐
    _FW_(double, NULL, 1, first_order_density_matrix_con__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms), i_r__, 0);
    _FW_(double, NULL, globalSize[0] * (4 * n_max_compute_atoms), trigonom_tab__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_fns_ham), radial_wave__, 0);
    _FW_(double, NULL, globalSize[0] * (n_basis_fns), spline_array_aux__, 0);
    _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms * n_basis_fns), aux_radial__, 0);
    _FW_(double, NULL, globalSize[0] * ((H_param.l_ylm_max + 1) * (H_param.l_ylm_max + 1) * n_max_compute_atoms), ylm_tab__, 0);
    _FW_(double, NULL, globalSize[0] * ((H_param.l_ylm_max + 1) * (H_param.l_ylm_max + 1) * n_max_compute_atoms), dylm_dtheta_tab__, 0);
    _FW_(double, NULL, 100, scaled_dylm_dphi_tab__, 0); // 反正这个和上面那个暂时无用
    _FW_(double, NULL, 1, kinetic_wave__, 0);
    _FW_(double, NULL, (globalSize[0] / localSize[0]) * ((n_max_batch_size + 127) / 128 * 128) + 256, grid_coord__, 0); // 长宽按 128 对齐
    // _FW_(double, NULL, globalSize[0] * n_max_compute_ham * H_param.n_spin, H_times_psi__, 0);
    _FW_(double, NULL, 1, H_times_psi__, 0);
    _FW_(double, NULL, 1, T_plus_V__, 0);
    // _FW_(double, NULL, globalSize[0] * (n_max_compute_atoms * n_basis_fns), T_plus_V__, 0);
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * ((n_max_batch_size+127)/128*128) * n_max_compute_ham, contract__, 0);
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * ((n_max_batch_size+127)/128*128) * n_max_compute_ham, wave_t__, 0);
    // _FW_(double, NULL, (globalSize[0]/localSize[0]) * n_max_compute_ham * n_max_compute_ham * H_param.n_spin, first_order_H_dense__, 0);
    _FW_(double, NULL, 1, contract__, 0);
    _FW_(double, NULL, 1, wave_t__, 0);
    _FW_(double, NULL, (globalSize[0] / localSize[0] + 1) * n_max_compute_ham * n_max_compute_ham + 128 * H_param.n_spin, first_order_H_dense__, 0);

    // H param
    arg_index = 0;
    void *args[95];
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.j_coord);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.n_spin);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.l_ylm_max);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.n_basis_local);
    setKernelArgs(args, arg_index++, sizeof(int), &H_param.n_matrix_size);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_l_max__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_points_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_batch_centers_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batch_center_all_batches__);
    // setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batch_point_to_i_full_point__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ins_idx_all_batches__);

    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_i_basis_h__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.partition_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_H__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_potential_parts_all_points__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_first_order_rho_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_first_order_potential_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_dVxc_drho_all_batches__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.local_rho_gradient__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_gradient_rho__);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    arg_index = 19;
    setKernelArgs(args, arg_index++, sizeof(int), &n_centers);
    setKernelArgs(args, arg_index++, sizeof(int), &n_centers_integrals);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_fns_ham);
    setKernelArgs(args, arg_index++, sizeof(int), &n_basis_fns);
    setKernelArgs(args, arg_index++, sizeof(int), &n_centers_basis_I);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_grid);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_atoms);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_ham);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_compute_dens);
    setKernelArgs(args, arg_index++, sizeof(int), &n_max_batch_size);
    setKernelArgs(args, arg_index++, sizeof(int), &index_hamiltonian_dim2);
    setKernelArgs(args, arg_index++, sizeof(int), &position_in_hamiltonian_dim1);
    setKernelArgs(args, arg_index++, sizeof(int), &position_in_hamiltonian_dim2);
    setKernelArgs(args, arg_index++, sizeof(int), &column_index_hamiltonian_size);
    // _CHK_(error, hipSuccess);

    arg_index = 33;
    // int test_batch = 1;
    // setKernelArgs(args, arg_index++, sizeof(int), &test_batch);
    setKernelArgs(args, arg_index++, sizeof(int), &n_my_batches_work_h);
    setKernelArgs(args, arg_index++, sizeof(int), &n_full_points_work_h);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    arg_index = 35;
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_atom);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.species_center);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.center_to_cell);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Cbasis_to_basis);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.Cbasis_to_center);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.centers_basis_integrals);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.index_hamiltonian);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.position_in_hamiltonian);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.column_index_hamiltonian);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.coords_center);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.n_grid);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.r_grid_min);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.log_r_grid_inc);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.perm_basis_fns_spl);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.outer_radius_sq);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_l);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_radius_sq);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn_start_spl);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_fn_atom);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_wave_ordered);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.basis_kinetic_ordered); // new
    // _CHK_(error, hipSuccess);
    // H batches
    arg_index = 57;
    // setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_size_H);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_n_compute_H);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_batch_i_basis_H);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.batches_points_coords_H);
    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    // H tmp
    arg_index = 60;
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dist_tab_sq__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dist_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dir_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.atom_index_inv__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_basis_fns__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_basis_fns_inv__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_atom_fns__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_start__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_end__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.one_over_dist_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.rad_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_index__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.l_count__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.fn_atom__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.zero_index_point__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_density_matrix_con__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.i_r__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.trigonom_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.radial_wave__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.spline_array_aux__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.aux_radial__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.ylm_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.dylm_dtheta_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.scaled_dylm_dphi_tab__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.kinetic_wave__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.grid_coord__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.H_times_psi__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.T_plus_V__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.contract__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.wave_t__);
    setKernelArgs(args, arg_index++, sizeof(hipDeviceptr_t), &hip_buf_com.first_order_H_dense__);
    setKernelArgs(args, arg_index++, sizeof(int), &max_n_batch_centers);

    // setKernelArgs(args, arg_index++, sizeof(double) * 1024, NULL); // local mem

    // IF_ERROR_EXIT(error != hipSuccess, error, "hipSetKernelArg failed");

    // gettimeofday(&end, NULL);
    // time_uses[time_index] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // // time_infos[time_index++] = "H writebuf and setarg";
    // if (myid < 4)
    // {
    //     printf("rank%d, %s: %lf seconds\n", myid, "H writebuf and setarg", time_uses[time_index - 1] / 1000000.0);
    //     fflush(stdout);
    // }
    // gettimeofday(&start, NULL);

    // printf("start H kernel\n");
    // fflush(stdout);

    // size_t localSize[] = {256};        // 覆盖前面的设置
    // size_t globalSize[] = {256 * 128}; // 覆盖前面的设置
    // dim3 blockDim(localSize[0], 1, 1);
    // dim3 gridDim(globalSize[0] / blockDim.x, 1, 1);
    gettimeofday(&start1, NULL);
    error = hipLaunchKernel(reinterpret_cast<const void *>(&integrate_first_order_h_sub_tmp2_), gridDim, blockDim, args, 0, 0);
    IF_ERROR_EXIT(error != hipSuccess, error, "hipLaunchKernel failed");
    hipMemcpy(H_param.first_order_H, hip_buf_com.first_order_H__, sizeof(double) * (H_param.n_matrix_size * H_param.n_spin), hipMemcpyDeviceToHost);

    gettimeofday(&end1, NULL);
    time_uses[1] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // time_infos[time_index++] = "H kernel and readbuf";

    // 可能有问题
    hipHostUnregister(MV(opencl_util, batches_size_h));
    hipFree(hip_buf_com.batches_size_H);
    // 可能有问题
    hipHostUnregister(MV(opencl_util, batches_batch_n_compute_h));
    hipFree(hip_buf_com.batches_batch_n_compute_H);
    // 可能有问题
    hipHostUnregister(MV(opencl_util, batches_batch_i_basis_h));
    hipFree(hip_buf_com.batches_batch_i_basis_H);
    // 可能有问题
    hipHostUnregister(MV(opencl_util, batches_points_coords_h));
    hipFree(hip_buf_com.batches_points_coords_H);

    hipFree(hip_buf_com.basis_l_max__);
    hipFree(hip_buf_com.n_points_all_batches__);
    hipFree(hip_buf_com.n_batch_centers_all_batches__);
    hipFree(hip_buf_com.batch_center_all_batches__);
    // hipFree(hip_buf_com.batch_point_to_i_full_point__);
    hipFree(hip_buf_com.ins_idx_all_batches__);

    hipFree(hip_buf_com.batches_batch_i_basis_h__);
    hipFree(hip_buf_com.partition_all_batches__);
    hipFree(hip_buf_com.first_order_H__);
    hipFree(hip_buf_com.local_potential_parts_all_points__);

    // 可能有问题
    hipHostUnregister(H_param.local_first_order_rho_all_batches);
    hipFree(hip_buf_com.local_first_order_rho_all_batches__);

    // 可能有问题
    hipHostUnregister(H_param.local_first_order_potential_all_batches);
    hipFree(hip_buf_com.local_first_order_potential_all_batches__);

    // 可能有问题
    hipHostUnregister(H_param.local_dVxc_drho_all_batches);
    hipFree(hip_buf_com.local_dVxc_drho_all_batches__);

    hipFree(hip_buf_com.local_rho_gradient__);
    hipFree(hip_buf_com.first_order_gradient_rho__);

    hipFree(hip_buf_com.dist_tab_sq__);
    hipFree(hip_buf_com.dist_tab__);
    hipFree(hip_buf_com.dir_tab__);
    hipFree(hip_buf_com.atom_index__);
    hipFree(hip_buf_com.atom_index_inv__);
    hipFree(hip_buf_com.i_basis_fns__);
    hipFree(hip_buf_com.i_basis_fns_inv__);
    hipFree(hip_buf_com.i_atom_fns__);
    hipFree(hip_buf_com.spline_array_start__);
    hipFree(hip_buf_com.spline_array_end__);
    hipFree(hip_buf_com.one_over_dist_tab__);
    hipFree(hip_buf_com.rad_index__);
    hipFree(hip_buf_com.wave_index__);
    hipFree(hip_buf_com.l_index__);
    hipFree(hip_buf_com.l_count__);
    hipFree(hip_buf_com.fn_atom__);
    hipFree(hip_buf_com.zero_index_point__);
    hipFree(hip_buf_com.wave__);
    hipFree(hip_buf_com.first_order_density_matrix_con__);
    hipFree(hip_buf_com.i_r__);
    hipFree(hip_buf_com.trigonom_tab__);
    hipFree(hip_buf_com.radial_wave__);
    hipFree(hip_buf_com.spline_array_aux__);
    hipFree(hip_buf_com.aux_radial__);
    hipFree(hip_buf_com.ylm_tab__);
    hipFree(hip_buf_com.dylm_dtheta_tab__);
    hipFree(hip_buf_com.scaled_dylm_dphi_tab__);
    hipFree(hip_buf_com.kinetic_wave__);
    hipFree(hip_buf_com.grid_coord__);
    hipFree(hip_buf_com.H_times_psi__);
    hipFree(hip_buf_com.T_plus_V__);
    hipFree(hip_buf_com.contract__);
    hipFree(hip_buf_com.wave_t__);
    hipFree(hip_buf_com.first_order_H_dense__);

    // if (myid == 0)
    //     m_save_check_h_(H_param.first_order_H, &(H_param.n_spin), &(H_param.n_matrix_size));

    // printf("End\n");

    // hip_common_buffer_free_();

    gettimeofday(&end, NULL);
    time_uses[0] = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    // time_infos[time_index++] = "H write check file(for debug)";

    // printf("rank %d, %s: %lf seconds, %s: %lf seconds\n", myid, " H kernel time is ", time_uses[1] / 1000000.0, " H time all is  ", time_uses[0] / 1000000.0);
    // fflush(stdout);

    // for(size_t i=0; i<time_index; i++){
    //   if(myid < 8)
    //     printf("rank%d, %s: %lf seconds\n", myid, time_infos[i], time_uses[i]/1000000.0);
    // }
}