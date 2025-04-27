#include <hip/hip_runtime.h>
#include "gpuAll.h"


#define ZERONULL 0
// #define atomic_cmpxchg atom_cmpxchg
//gpu main clocks 1225Mhz
#define H_IC_TILE 16
#define H_JC_TILE 16
#define H_PT_TILE 16
#define H_IC_WORK 3
#define H_JC_WORK 3
typedef double m_float_type;
#define LOCAL_SIZE 256
#define TILE_N 16 // blockDim.x 必须为 16 的倍数
#define TILE_K 16 // TILE_K == TILE_N
#define WORK_M 4
#define WORK_N 8
#define TILE_M (LOCAL_SIZE / TILE_N)
#define MERGE_BATCH 3
// #define LOCAL_SIZE 64
// #define TILE_N 8 // blockDim.x 必须为 16 的倍数
// #define TILE_K 8 // TILE_K == TILE_N
// #define WORK_M 4
// #define WORK_N 4
// #define TILE_M (LOCAL_SIZE / TILE_N)

__device__ void prune_radial_basis_p2_c_h(
    int *dim_atoms_, int *dim_fns_, double *dist_tab_sq,
    double *dist_tab,
    double *dir_tab, // (3, n_atom_list)
    int *n_compute_atoms_, int *atom_index, int *atom_index_inv,
    int *n_compute_fns_, int *i_basis_fns,
    int *i_basis_fns_inv, // (n_basis_fns,n_centers)
    int *i_atom_fns, int *spline_array_start,
    int *spline_array_end, int *n_atom_list_, int *atom_list,
    int *n_compute_, const int *i_basis, int *n_batch_centers_,
    int *batch_center, double *one_over_dist_tab,
    int *rad_index, int *wave_index, int *l_index,
    int *l_count, int *fn_atom, int *n_zero_compute_,
    int *zero_index_point,
    // outer
    int n_basis_fns, // const
    const int *center_to_atom, const int *species_center,
    const int *Cbasis_to_basis, const int *Cbasis_to_center,
    const int *perm_basis_fns_spl, const double *outer_radius_sq,
    const int *basis_fn, const int *basis_l,
    const double *atom_radius_sq, const int *basis_fn_start_spl,
    const int *basis_fn_atom, double *pbc_lists_coords_center,
    double coords_center0, double coords_center1, double coords_center2, int gid_trans, int gsize_trans, int bid_trans)
{

    

    int bid = bid_trans;
    int gid = gid_trans;
    int gsize = gsize_trans;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
// #define dir_tab(i, j) dir_tab[(i)-1 + 3 * ((j)-1)]
#define dir_tab(i, j) dir_tab[lid + lsize * ((i)-1 + 3 * ((j)-1))]
// #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]
#define i_basis_fns_inv(i, j) \
    i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * gsize + gid]
// #define i_basis_fns_inv(i, j) \
//     i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * bid * gridDim.x * lsize + gid]
// #define i_basis_fns_inv(i, j) \
//     i_basis_fns_inv[gid * n_basis_fns * (n_max_compute_atoms + 1) + ((i)-1 + n_basis_fns * ((j)-1))]
// outer
#define center_to_atom(i) center_to_atom[(i)-1]
#define species_center(i) species_center[(i)-1]
#define Cbasis_to_basis(i) Cbasis_to_basis[(i)-1]
#define Cbasis_to_center(i) Cbasis_to_center[(i)-1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i)-1]
#define outer_radius_sq(i) outer_radius_sq[(i)-1]
#define basis_fn(i) basis_fn[(i)-1]
#define basis_l(i) basis_l[(i)-1]
#define atom_radius_sq(i) atom_radius_sq[(i)-1]
#define basis_fn_start_spl(i) basis_fn_start_spl[(i)-1]
#define basis_fn_atom(i, j) basis_fn_atom[(i)-1 + ((j)-1) * n_basis_fns]
#define pbc_lists_coords_center(i, j) \
    pbc_lists_coords_center[((j)-1) * 3 + (i)-1]
#define atom_index_inv(i) atom_index_inv[((i)-1) * lsize + lid]

    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;
    // int lid = threadIdx.x;
    // int lsize = blockDim.x;

    // if (gid == 0)
    //     printf("start prune_radial_basis_p2_c_ : gid is : %d\n", gid);
    int dim_atoms = *dim_atoms_;
    int dim_fns = *dim_fns_;
    int n_atom_list = *n_atom_list_;
    int n_batch_centers = *n_batch_centers_;
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int n_zero_compute = *n_zero_compute_;

    // __shared__ variables
    int i_offset_spl;
    int l_aux;

    // counters
    int i_batch_center;
    int i_basis_1;
    int i_compute;
    int i_fn;
    int i_lm;
    int i_center, i_center_L;
    int i_atom_compute;
    int i_spline;

    // n_batch_centers = 50000;
    double dir_tab_local[3];
    for (i_batch_center = 1; i_batch_center <= n_batch_centers;
         i_batch_center++)
    {
        i_center_L = batch_center[i_batch_center - 1];
        i_center = atom_list[i_center_L - 1];

        dir_tab_local[0] =
            coords_center0 - pbc_lists_coords_center(1, atom_list[i_center_L - 1]);
        dir_tab_local[1] =
            coords_center1 - pbc_lists_coords_center(2, atom_list[i_center_L - 1]);
        dir_tab_local[2] =
            coords_center2 - pbc_lists_coords_center(3, atom_list[i_center_L - 1]);

        double dist_tab_sq_now = dir_tab_local[0] * dir_tab_local[0] +
                                 dir_tab_local[1] * dir_tab_local[1] +
                                 dir_tab_local[2] * dir_tab_local[2];

        // ???time
        if (dist_tab_sq_now <= atom_radius_sq(species_center(i_center)) &&
            dist_tab_sq_now > 0.0)
        {

            n_compute_atoms = n_compute_atoms + 1;

            atom_index[(n_compute_atoms - 1)] = i_center;
            // atom_index_inv(i_center) = n_compute_atoms;

            dist_tab_sq[n_compute_atoms - 1] = dist_tab_sq_now;

            dist_tab[n_compute_atoms - 1] = sqrt(dist_tab_sq[n_compute_atoms - 1]);

            double tmp = 1.0 / dist_tab[n_compute_atoms - 1];

            one_over_dist_tab[n_compute_atoms - 1] =
                1.0 / dist_tab[n_compute_atoms - 1];

            for (int i = 1; i <= 3; i++)
            {
                // dir_tab(i, n_compute_atoms) = dir_tab(i, i_center_L) * tmp;
                dir_tab(i, n_compute_atoms) = dir_tab_local[i - 1] * tmp;
            }
        }
    }
    // __private__int private_center_to_atom[MACRO_n_centers+1];
    // // 确保非法的部分将被映射到从不会被使用的 MACRO_n_max_compute_atoms + 1
    // 上，由此保证其值为 0 for (int i=0; i<MACRO_n_centers+1; i++){
    //   private_center_to_atom[i] = MACRO_n_max_compute_atoms + 1;
    // }
    // next, check for radial basis functions
    // __shared__ double i_basis_fns_inv_split[256][];
    // __shared__ int tmp_shared_arr[1280];
    // for (int i = 0; i < 5; i++)
    // {
    //     tmp_shared_arr[5 * lid + i] = 0;
    // }
    for (i_atom_compute = 1; i_atom_compute <= n_compute_atoms;
         i_atom_compute++)
    {
        i_center = atom_index[(i_atom_compute - 1)];
        // private_center_to_atom[i_center] = i_atom_compute;
        i_offset_spl = basis_fn_start_spl(species_center(i_center));
        double dist_tab_sq_reg = dist_tab_sq[i_atom_compute - 1];
        for (i_spline = 1; i_spline <= n_basis_fns; i_spline++)
        {
            i_basis_1 = perm_basis_fns_spl(i_spline);
            if (basis_fn_atom(i_basis_1, center_to_atom(i_center)))
            {
                if (dist_tab_sq_reg <= outer_radius_sq(i_basis_1))
                {
                    // tmp_shared_arr[5 * lid] = i_spline;
                    // spline_array_start[i_atom_compute - 1] = tmp_shared_arr[5 * lid];
                    spline_array_start[i_atom_compute - 1] = i_spline;
                    break;
                }
            }
        }

        for (i_basis_1 = 1; i_basis_1 <= n_basis_fns; i_basis_1++)
        {
            if (basis_fn_atom(i_basis_1, center_to_atom(i_center)))
            {
                if (dist_tab_sq_reg <= outer_radius_sq(i_basis_1))
                {
                    n_compute_fns = n_compute_fns + 1;
                    // i_basis_fns[n_compute_fns - 1] = i_basis_1;
                    // i_atom_fns[n_compute_fns - 1] = i_center;
                    // i_basis_fns_inv(i_basis_1, i_center) = n_compute_fns;

                    // tmp_shared_arr[5 * lid + 1] = n_compute_fns;
                    // tmp_shared_arr[5 * lid + 2] = i_atom_compute;
                    // i_basis_fns_inv(i_basis_1, i_atom_compute) = tmp_shared_arr[5 * lid + 1];
                    // fn_atom[n_compute_fns - 1] = tmp_shared_arr[5 * lid + 2];

                    i_basis_fns_inv(i_basis_1, i_atom_compute) = n_compute_fns;
                    fn_atom[n_compute_fns - 1] = i_atom_compute;
                }
                i_offset_spl = i_offset_spl + 1;
            }
        }
        // tmp_shared_arr[5 * lid + 3] = n_compute_fns;
        // tmp_shared_arr[5 * lid + 4] = i_offset_spl - 1;
        // rad_index[i_atom_compute - 1] = tmp_shared_arr[5 * lid + 3];
        // spline_array_end[i_atom_compute - 1] = tmp_shared_arr[5 * lid + 4];
        rad_index[i_atom_compute - 1] = n_compute_fns;
        spline_array_end[i_atom_compute - 1] = i_offset_spl - 1;
    }

    // athread 版本不考虑下面这种情况，因此保持一致 WARNING
    // // 担忧出现 i_center = atom_index[(i_atom_compute - 1)]; 将不同的
    // i_atom_compute 映射到相同的 i_center
    // // 为了确保后文其他函数调用是正常，额外传一个 i_basis_fns_inv_remapping
    // 出去
    // // 暂时不选用重新给 i_basis_fns_inv 赋值的方式
    // for (i_atom_compute = 1; i_atom_compute <= n_compute_atoms;
    // i_atom_compute++) {
    //   i_basis_fns_inv_remapping[i_atom_compute-1] =
    //   private_center_to_atom[atom_index[(i_atom_compute - 1)]];
    // }

    n_zero_compute = 0;
    for (int i = 1; i <= n_compute_fns; i++)
    {
        wave_index[i - 1] = 0;
    }

    i_compute = 1;

    while (i_compute <= n_compute)
    {
        i_basis_1 = i_basis[i_compute - 1];
        int cc = dim_atoms + 1;
        int i_center = Cbasis_to_center(i_basis_1);
        for (int i = 0; i < dim_atoms; i++)
        {
            if (atom_index[i] == i_center)
            {
                cc = i + 1;
                break;
            }
        }
        i_fn = i_basis_fns_inv(basis_fn(Cbasis_to_basis(i_basis_1)), cc);
        l_aux = basis_l(Cbasis_to_basis(i_basis[i_compute - 1]));
        if (i_fn == 0)
        {
            for (i_lm = 0; i_lm <= 2 * l_aux; i_lm++)
            {
                n_zero_compute = n_zero_compute + 1;
                zero_index_point[n_zero_compute - 1] = i_compute + i_lm;
            }
        }
        else if (wave_index[i_fn - 1] == 0)
        {

            wave_index[i_fn - 1] = i_compute;
            l_count[i_fn - 1] = l_aux;
        }
        i_compute = i_compute + 2 * l_aux + 1;
    }
    *n_compute_atoms_ = n_compute_atoms;
    *n_compute_fns_ = n_compute_fns;
    *n_zero_compute_ = n_zero_compute;

#undef center_to_atom
#undef species_center
#undef Cbasis_to_basis
#undef Cbasis_to_center
#undef perm_basis_fns_spl
#undef outer_radius_sq
#undef basis_fn
#undef basis_l
#undef atom_radius_sq
#undef basis_fn_start_spl
#undef basis_fn_atom
#undef pbc_lists_coords_center
#undef atom_index_inv

#undef dir_tab
#undef i_basis_fns_inv
}

__device__ void tab_local_geometry_p2_c_(int *n_compute_atoms_, int *atom_index,
                                         double *dist_tab, double *i_r,
                                         // outer
                                         const int *species_center,
                                         double *r_grid_min,
                                         double *log_r_grid_inc)
{
#define species_center(i) species_center[(i)-1]
#define r_grid_min(i) r_grid_min[(i)-1]
#define log_r_grid_inc(i) log_r_grid_inc[(i)-1]
    int gsize = gridDim.x * blockDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gid == 0)
    //     printf("start tab_local_geometry_p2_c_: gid is : %d\n", gid);
    int n_compute_atoms = *n_compute_atoms_;
    //  counters
    int i_compute_atom;
    for (i_compute_atom = 1; i_compute_atom <= n_compute_atoms;
         i_compute_atom++)
    {
        double r_current = dist_tab[i_compute_atom - 1];
        int i_species = species_center(atom_index[(i_compute_atom - 1)]);
        i_r[i_compute_atom - 1] = 1.0 + log(r_current / r_grid_min(i_species)) /
                                            log_r_grid_inc(i_species);
    }
    // if (gid == 0)
    //     printf("end tab_local_geometry_p2_c_: gid is : %d\n", gid);

#undef species_center
#undef r_grid_min
#undef log_r_grid_inc
}

__device__ void tab_gradient_ylm_p0_c_2(
    double *trigonom_tab, // ( 4, n_compute_atoms )
    int *basis_l_max, int *l_ylm_max_, int *n_compute_atoms_,
    int *atom_index,
    double *ylm_tab,             // ( (l_ylm_max+1)**2, n_compute_atoms )
    double *dylm_dtheta_tab,     // ( (l_ylm_max+1)**2, n_compute_atoms )
    double *scaled_dylm_dphi_tab // ( (l_ylm_max+1)**2, n_compute_atoms )
    // outer
    ,
    double *dir_tab, const int *species_center, int gid_trans, int gsize_trans)
{
    int gid = gid_trans;
    int gsize = gsize_trans;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
#define species_center(i) species_center[(i)-1]
// #define dir_tab(i, j) dir_tab[(i)-1 + ((j)-1) * 3]
#define dir_tab(i, j) dir_tab[lid + lsize * ((i)-1 + ((j)-1) * 3)]
    int n_compute_atoms = *n_compute_atoms_;
    int l_ylm_max = *l_ylm_max_;
    int l_ylm_max_1pow2 = (l_ylm_max + 1) * (l_ylm_max + 1);
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;
    // int lid = threadIdx.x;
    // int lsize = blockDim.x;

    // if (gid == 0)
    //     printf("start tab_gradient_ylm_p0_c_2: gid is : %d\n", gid);
#define trigonom_tab(i, j) trigonom_tab[(i)-1 + ((j)-1) * 4]
#define ylm_tab(i, j) ylm_tab[(i - 1 + l_ylm_max_1pow2 * (j - 1)) * gsize + gid]
#define dylm_dtheta_tab(i, j) dylm_dtheta_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
#define scaled_dylm_dphi_tab(i, j) \
    scaled_dylm_dphi_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
    for (int i_atom = 1; i_atom <= n_compute_atoms; i_atom++)
    {
        double trigonom_tab_reg[4];
        {
            //  __shared__ variables
            double abmax, abcmax, ab, abc;
            abmax = fmax(fabs(dir_tab(1, i_atom)), fabs(dir_tab(2, i_atom)));
            if (abmax > 0.0)
            {
                ab = sqrt(pow(dir_tab(1, i_atom), 2.0) + pow(dir_tab(2, i_atom), 2.0));
                trigonom_tab_reg[3] = dir_tab(1, i_atom) / ab;
                trigonom_tab_reg[2] = dir_tab(2, i_atom) / ab;
            }
            else
            {
                trigonom_tab_reg[3] = 1.0;
                trigonom_tab_reg[2] = 0.0;
                ab = 0.0;
            }
            abcmax = fmax(abmax, fabs(dir_tab(3, i_atom)));
            if (abcmax > 0.0)
            {
                abc = sqrt(pow(ab, 2.0) + pow(dir_tab(3, i_atom), 2.0));
                trigonom_tab_reg[1] = dir_tab(3, i_atom) / abc;
                trigonom_tab_reg[0] = ab / abc;
            }
            else
            {
                trigonom_tab_reg[1] = 1.0;
                trigonom_tab_reg[0] = 0.0;
            }
        }
        //     increment_ylm_deriv(trigonom_tab(1, i_atom), trigonom_tab(2, i_atom),
        //     trigonom_tab(3, i_atom),
        //                         trigonom_tab(4, i_atom), 0,
        //                         basis_l_max(species_center(atom_index(i_atom))),
        //                         &ylm_tab(1, i_atom), &dylm_dtheta_tab(1, i_atom),
        //                         &scaled_dylm_dphi_tab(1, i_atom));
        if (1)
        {
            // SHEvalderiv_c_(basis_l_max[species_center(atom_index[i_atom - 1]) - 1],
            // trigonom_tab(1, i_atom),
            //                trigonom_tab(2, i_atom), trigonom_tab(3, i_atom),
            //                trigonom_tab(4, i_atom), &ylm_tab(1, i_atom),
            //                &dylm_dtheta_tab(1, i_atom), &scaled_dylm_dphi_tab(1,
            //                i_atom));
            {
                int l_atom_max =
                    basis_l_max[species_center(atom_index[(i_atom - 1)]) - 1];

                /* variables for tabulate ylm */
                double YLLI, YLL1I, YL1L1I, YLMI;
                double YLLR, YLL1R, YL1L1R, YLMR;
                int I2L, I4L2, INDEX, INDEX2, L, M, MSIGN;
                /* VB */
                int I22L, I24L2;
                double TEMP1, TEMP2, TEMP3;

                double D4LL1C, D2L13;
                const double PI = 3.14159265358979323846;

#define trigonom_tab_(i1) trigonom_tab_reg[i1 - 1]
#define ylm_tab_(i) ylm_tab(i, i_atom)
                if (0 <= 0)
                {
                    YLLR = 1.0 / sqrt(4.0 * PI);
                    YLLI = 0.0;
                    ylm_tab_(1) = YLLR;
                }

                if ((0 <= 1) && (l_atom_max >= 1))
                {
                    ylm_tab_(3) = sqrt(3.00) * YLLR * trigonom_tab_(2);
                    TEMP1 = -sqrt(3.00) * YLLR * trigonom_tab_(1);
                    ylm_tab_(4) = TEMP1 * trigonom_tab_(4);
                    ylm_tab_(2) = -TEMP1 * trigonom_tab_(3);
                }

                // L = max(2,0)
                for (L = 2; L <= l_atom_max; L++)
                {
                    INDEX = L * L + 1;
                    INDEX2 = INDEX + 2 * L;
                    MSIGN = 1 - 2 * (L % 2);

                    YL1L1R = ylm_tab_(INDEX - 1);
                    YL1L1I = -MSIGN * ylm_tab_(INDEX - 2 * L + 1);
                    TEMP1 =
                        -sqrt((double)(2 * L + 1) / (double)(2 * L)) * trigonom_tab_(1);
                    YLLR =
                        TEMP1 * (trigonom_tab_(4) * YL1L1R - trigonom_tab_(3) * YL1L1I);
                    YLLI =
                        TEMP1 * (trigonom_tab_(4) * YL1L1I + trigonom_tab_(3) * YL1L1R);
                    ylm_tab_(INDEX2) = YLLR;
                    ylm_tab_(INDEX) = MSIGN * YLLI;
                    INDEX2 = INDEX2 - 1;
                    INDEX = INDEX + 1;

                    TEMP2 = sqrt((double)(2 * L + 1)) * trigonom_tab_(2);
                    YLL1R = TEMP2 * YL1L1R;
                    YLL1I = TEMP2 * YL1L1I;
                    ylm_tab_(INDEX2) = YLL1R;
                    ylm_tab_(INDEX) = -MSIGN * YLL1I;
                    INDEX2 = INDEX2 - 1;
                    INDEX = INDEX + 1;

                    I4L2 = INDEX - 4 * L + 2;
                    I2L = INDEX - 2 * L;
                    I24L2 = INDEX2 - 4 * L + 2;
                    I22L = INDEX2 - 2 * L;
                    D4LL1C = trigonom_tab_(2) * sqrt((double)(4 * L * L - 1));
                    D2L13 = -sqrt((double)(2 * L + 1) / (double)(2 * L - 3));

                    for (M = L - 2; M >= 0; M--)
                    {
                        TEMP1 = 1.00 / sqrt((double)((L + M) * (L - M)));
                        TEMP2 = D4LL1C * TEMP1;
                        TEMP3 = D2L13 * sqrt((double)((L + M - 1) * (L - M - 1))) * TEMP1;
                        YLMR = TEMP2 * ylm_tab_(I22L) + TEMP3 * ylm_tab_(I24L2);
                        YLMI = TEMP2 * ylm_tab_(I2L) + TEMP3 * ylm_tab_(I4L2);
                        ylm_tab_(INDEX2) = YLMR;
                        ylm_tab_(INDEX) = YLMI;

                        INDEX2 = INDEX2 - 1;
                        INDEX = INDEX + 1;
                        I24L2 = I24L2 - 1;
                        I22L = I22L - 1;
                        I4L2 = I4L2 + 1;
                        I2L = I2L + 1;
                    }
                }
#undef trigonom_tab_
#undef ylm_tab_
            }
        }
        else
        {
            // if(gid == 0) printf("%s, not finished\n", __func__); // TODO
            // exit(-19);
        }
    }
    const int REL_x2c = 7;
    const int REL_q4c = 8;
    // if (flag_rel == REL_q4c || flag_rel == REL_x2c) {
    if (0)
    {
        for (int i_atom = 1; i_atom <= n_compute_atoms; i_atom++)
        {
            if (l_ylm_max >= 1)
            {
                ylm_tab(4, i_atom) = -ylm_tab(4, i_atom);
                dylm_dtheta_tab(4, i_atom) = -dylm_dtheta_tab(4, i_atom);
                scaled_dylm_dphi_tab(4, i_atom) = -scaled_dylm_dphi_tab(4, i_atom);
            }
            if (l_ylm_max >= 2)
            {
                ylm_tab(8, i_atom) = -ylm_tab(8, i_atom);
                dylm_dtheta_tab(8, i_atom) = -dylm_dtheta_tab(8, i_atom);
                scaled_dylm_dphi_tab(8, i_atom) = -scaled_dylm_dphi_tab(8, i_atom);
            }
            if (l_ylm_max >= 3)
            {
                ylm_tab(14, i_atom) = -ylm_tab(14, i_atom);
                dylm_dtheta_tab(14, i_atom) = -dylm_dtheta_tab(14, i_atom);
                scaled_dylm_dphi_tab(14, i_atom) = -scaled_dylm_dphi_tab(14, i_atom);
                ylm_tab(16, i_atom) = -ylm_tab(16, i_atom);
                dylm_dtheta_tab(16, i_atom) = -dylm_dtheta_tab(16, i_atom);
                scaled_dylm_dphi_tab(16, i_atom) = -scaled_dylm_dphi_tab(16, i_atom);
            }
            if (l_ylm_max >= 4)
            {
                ylm_tab(22, i_atom) = -ylm_tab(22, i_atom);
                dylm_dtheta_tab(22, i_atom) = -dylm_dtheta_tab(22, i_atom);
                scaled_dylm_dphi_tab(22, i_atom) = -scaled_dylm_dphi_tab(22, i_atom);
                ylm_tab(24, i_atom) = -ylm_tab(24, i_atom);
                dylm_dtheta_tab(24, i_atom) = -dylm_dtheta_tab(24, i_atom);
                scaled_dylm_dphi_tab(24, i_atom) = -scaled_dylm_dphi_tab(24, i_atom);
            }
            if (l_ylm_max >= 5)
            {
                ylm_tab(32, i_atom) = -ylm_tab(32, i_atom);
                dylm_dtheta_tab(32, i_atom) = -dylm_dtheta_tab(32, i_atom);
                scaled_dylm_dphi_tab(32, i_atom) = -scaled_dylm_dphi_tab(32, i_atom);
                ylm_tab(34, i_atom) = -ylm_tab(34, i_atom);
                dylm_dtheta_tab(34, i_atom) = -dylm_dtheta_tab(34, i_atom);
                scaled_dylm_dphi_tab(34, i_atom) = -scaled_dylm_dphi_tab(34, i_atom);
                ylm_tab(36, i_atom) = -ylm_tab(36, i_atom);
                dylm_dtheta_tab(36, i_atom) = -dylm_dtheta_tab(36, i_atom);
                scaled_dylm_dphi_tab(36, i_atom) = -scaled_dylm_dphi_tab(36, i_atom);
            }
            if (l_ylm_max >= 6)
            {
                ylm_tab(44, i_atom) = -ylm_tab(44, i_atom);
                dylm_dtheta_tab(44, i_atom) = -dylm_dtheta_tab(44, i_atom);
                scaled_dylm_dphi_tab(44, i_atom) = -scaled_dylm_dphi_tab(44, i_atom);
                ylm_tab(46, i_atom) = -ylm_tab(46, i_atom);
                dylm_dtheta_tab(46, i_atom) = -dylm_dtheta_tab(46, i_atom);
                scaled_dylm_dphi_tab(46, i_atom) = -scaled_dylm_dphi_tab(46, i_atom);
                ylm_tab(48, i_atom) = -ylm_tab(48, i_atom);
                dylm_dtheta_tab(48, i_atom) = -dylm_dtheta_tab(48, i_atom);
                scaled_dylm_dphi_tab(48, i_atom) = -scaled_dylm_dphi_tab(48, i_atom);
            }
        }
    }
    // if (gid == 0)
    //     printf("end tab_gradient_ylm_p0_c_2: gid is : %d\n", gid);
#undef trigonom_tab
#undef ylm_tab
#undef dylm_dtheta_tab
#undef scaled_dylm_dphi_tab

#undef species_center
#undef dir_tab
}

__device__ double spline_vector_waves_c_(double r_output, double *spl_param,
                                         int n_grid_dim, int n_compute_fns,
                                         int spline_start, int spline_end,
                                         int n_spl_points, int n_spline,
                                         double *out_wave, int index)
{
    // if (gid == 0)
    //     printf("start spline_vector_waves_c_: gid is : %d\n", gid);
#define spl_param(i, j, k) \
    spl_param[i - 1 + n_compute_fns * (j - 1 + 4 * (k - 1))]
#define out_wave(i) out_wave[i - 1]
    int i_spl;
    double t, term;
    int i_term;
    i_spl = (int)(r_output);
    i_spl = i_spl > 1 ? i_spl : 1;
    i_spl = i_spl < (n_spl_points - 1) ? i_spl : (n_spl_points - 1);
    // i_spl = fmax(1, i_spl);
    // i_spl = fmin(n_spl_points - 1, i_spl);
    t = r_output - (double)(i_spl);
    double ans = spl_param(index + 1 + spline_start - 1, 1, i_spl);
    // for (int i = 1; i <= n_spline; i++) {
    //   out_wave(i) = spl_param(i + spline_start - 1, 1, i_spl);
    // }
    term = 1.0;
    for (i_term = 2; i_term <= 4; i_term++)
    {
        term = term * t;
        // for (int i = 1; i <= n_spline; i++) {
        //   out_wave(i) = out_wave(i) + term * spl_param(i + spline_start - 1,
        //   i_term, i_spl);
        // }
        ans += term * spl_param(index + 1 + spline_start - 1, i_term, i_spl);
    }
    return ans;
    // if (gid == 0)
    //     printf("end spline_vector_waves_c_: gid is : %d\n", gid);
#undef spl_param
#undef out_wave
}

__device__ void evaluate_radial_functions_p0_c_(
    int *spline_array_start, int *spline_array_end,
    int *n_compute_atoms_, int *n_compute_fns_, double *dist_tab,
    double *i_r, int *atom_index,
    int *i_basis_fns_inv, // (n_basis_fns,n_centers)
    double *spline_data,  // (n_basis_fns,n_max_spline, n_max_grid)
    double *wave_aux, int *derivative_, int *n_compute_,
    int *n_basis_list_
    // outer
    ,
    int n_basis_fns, int n_max_grid, const int *species_center,
    int *n_grid,
    const int *perm_basis_fns_spl
    // tmp
    ,
    double *spline_array_aux, int gid_trans, int gsize_trans, int bid_trans)
{
    int bid = bid_trans;
    int gid = gid_trans;
    int gsize = gsize_trans;
#define species_center(i) species_center[(i)-1]
#define n_grid(i) n_grid[(i)-1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i)-1]

// #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]
#define i_basis_fns_inv(i, j) \
    i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * gsize + gid]
    // #define i_basis_fns_inv(i, j) \
//     i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * bid * gridDim.x * blockDim.x + gid]
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;

    // if (gid == 0)
    //     printf("before evaluate_radial_functions_p0_c_: gid is : %d\n", gid);
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int n_basis_list = *n_basis_list_;
    int derivative = *derivative_;
    // double spline_array_aux[n_basis_fns];
    for (int i_atom_1 = 1; i_atom_1 <= n_compute_atoms; i_atom_1++)
    {
        int current_atom = atom_index[(i_atom_1 - 1)];
        int current_species = species_center(current_atom);
        int spline_start = spline_array_start[i_atom_1 - 1];
        int spline_end = spline_array_end[i_atom_1 - 1];
        int n_spline = spline_end - spline_start + 1;
        double r_point = i_r[i_atom_1 - 1];
        double distance_from_atom = dist_tab[i_atom_1 - 1];
        // spline_vector_waves
        int i_rad = (spline_start - 1);
        for (int i_spline = spline_start; i_spline <= spline_end; i_spline++)
        {
            i_rad = i_rad + 1;
            int spline_counter = i_spline - (spline_start - 1);
            int current_basis_fn = perm_basis_fns_spl(i_rad);
            // int current_basis_fn_comp = i_basis_fns_inv(current_basis_fn,
            // current_atom);
            int current_basis_fn_comp = i_basis_fns_inv(current_basis_fn, i_atom_1);
            if (current_basis_fn_comp == 0)
                continue;
            // TODO 注意 spline_data 切片
            double tmp = spline_vector_waves_c_(r_point, spline_data, n_max_grid,
                                                n_basis_fns, spline_start, spline_end,
                                                n_grid(current_species), n_spline,
                                                spline_array_aux, spline_counter - 1);
            // if (derivative) {
            //   if (distance_from_atom > 0)
            //     wave_aux[current_basis_fn_comp - 1] =
            //     spline_array_aux[spline_counter - 1];
            //   else
            //     wave_aux[current_basis_fn_comp - 1] = 0.0;
            // } else {
            //   wave_aux[current_basis_fn_comp - 1] = spline_array_aux[spline_counter
            //   - 1];
            // }
            // if (derivative) {
            //   if (distance_from_atom > 0)
            //     wave_aux[current_basis_fn_comp - 1] = tmp;
            //   else
            //     wave_aux[current_basis_fn_comp - 1] = 0.0;
            // } else {
            //   wave_aux[current_basis_fn_comp - 1] = tmp;
            // }
            wave_aux[current_basis_fn_comp - 1] =
                (derivative && (distance_from_atom <= 0)) ? 0.0 : tmp;
        }
    }
    // if (gid == 0)
    //     printf("end evaluate_radial_functions_p0_c_: gid is : %d\n", gid);
#undef i_basis_fns_inv

#undef species_center
#undef n_grid
#undef perm_basis_fns_spl
}

__device__ void mul_vec_c_2(double *wave, int n_mul, double *ylm,
                            double factor, int array_factor, int ylm_factor)
{
    for (int i = 0; i < n_mul; i++)
        wave[i * array_factor] = ylm[i * ylm_factor] * factor;
}

__device__ void evaluate_waves_p2_c_2(
    int *n_compute_, int *n_compute_atoms_, int *n_compute_fns_,
    int *l_ylm_max_,
    double *ylm_tab, // ((l_ylm_max+1)**2, n_compute_atoms )
    double *one_over_dist_tab, double *radial_wave,
    double *wave, int *rad_index, int *wave_index,
    int *l_index, int *l_count, int *fn_atom,
    int *n_zero_compute_,
    int *zero_index_point
    // tmp
    ,
    double *aux_radial, int array_factor, int gid_trans, int gsize_trans)
{
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;
    int gid = gid_trans;
    int gsize = gsize_trans;
    // if (gid == 0)
    //     printf("before evaluate_waves_p2_c_2: gid is : %d\n", gid);
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int l_ylm_max = *l_ylm_max_;
    int n_zero_compute = *n_zero_compute_;
    // double aux_radial[n_compute_fns];
    int index_start = 1;
    int index_end;
    int ylm_tab_dim1 = (l_ylm_max + 1) * (l_ylm_max + 1);
    for (int i_compute_atom = 1; i_compute_atom <= n_compute_atoms;
         i_compute_atom++)
    {
        index_end = rad_index[i_compute_atom - 1];
        for (int i = index_start; i <= index_end; i++)
            aux_radial[i - 1] =
                radial_wave[i - 1] * one_over_dist_tab[i_compute_atom - 1];
        index_start = index_end + 1;
    }

    for (int i_compute_fn = 1; i_compute_fn <= n_compute_fns; i_compute_fn++)
    {
        int l_aux = l_count[i_compute_fn - 1];
        int l_index_val = l_aux * l_aux + 1;
        int l_count_val = 2 * l_aux;
        mul_vec_c_2(&wave[(wave_index[i_compute_fn - 1] - 1) * array_factor],
                    l_count_val + 1,
                    &ylm_tab[(l_index_val - 1 +
                              (fn_atom[i_compute_fn - 1] - 1) * ylm_tab_dim1) *
                                 gsize +
                             gid],
                    aux_radial[i_compute_fn - 1], array_factor, gsize);
    }
    for (int i_compute_point = 1; i_compute_point <= n_zero_compute;
         i_compute_point++)
    {
        int i_compute = zero_index_point[i_compute_point - 1];
        wave[(i_compute - 1) * array_factor] = 0.0;
    }
    // if (gid == 0)
    //     printf("end evaluate_waves_p2_c_2: gid is : %d\n", gid);
}

__device__ void evaluate_first_order_h_polar_reduce_memory_c_(
    double *first_order_H, int *n_points_,
    double *partition_tab, // (n_points)
    double *grid_coord,    // (n_points)
    double *H_times_psi,   // (n_max_compute_ham,n_points,n_spin)
    int *n_compute_c_,
    const int *i_basis_index,    // (n_compute_c)
    const double *wave,          // (n_max_compute_ham, n_points)
    double *gradient_basis_wave, // (n_max_compute_ham, 3, n_points)
    double *first_order_rho,     // (n_spin, n_points)
    double *v_hartree_gradient,  // (n_points)
                                 // LDA
    double *dVxc_drho,           // (3,n_points)
                                 // GGA
    double *vsigma, double *v2rho2, double *v2rhosigma,
    double *v2sigma2, double *gradient_rho,
    double *first_order_gradient_rho,
    // ---
    int *n_matrix_size_,
    // use
    int *ins_idx,
    int *n_basis_local_, // 仅适用于使用了 local_index 且实际使用 ins_idx
                         // 转换矩阵的版本
    // outer
    int *n_spin_, int *n_max_compute_ham_,
    // tmp
    double *contract, double *wave_t,
    double *first_order_H_dense,
    // __shared__ double local_contract[H_PT_TILE][H_JC_TILE * H_JC_WORK],
    // __shared__ double local_wave[H_PT_TILE][H_IC_TILE * H_IC_WORK]
    // __shared__ double A_local[TILE_M * WORK_M][TILE_K],
    // __shared__ double B_local[TILE_K][TILE_N * WORK_N]
    double A_local[TILE_M * WORK_M][TILE_K],
    double B_local[TILE_K][TILE_N * WORK_N],
    int total_points, int i_my_batch, int bid_local, int n_max_batch_size,
    int batch_nums, int *n_compute_c_all, int *n_points_all, int batch_start)
{
    int n_compute_c = *n_compute_c_;
    int n_points = *n_points_;
    int n_basis_local = *n_basis_local_;
    int n_spin = *n_spin_;
    int n_max_compute_ham = *n_max_compute_ham_;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
    // if (gid == 0)
    //     printf("before evaluate_first_order_h_polar_reduce_memory_c_: gid is : %d\n", gid);
    // double contract[n_points * n_compute_c];
    // double wave_t[n_points * n_max_compute_ham];
    // double first_order_H_dense[n_compute_c * n_compute_c * n_spin];

    // __shared__ double local_contract[H_PT_TILE][H_JC_TILE * H_JC_WORK];
    // __shared__ double local_wave[H_PT_TILE][H_IC_TILE * H_IC_WORK];

// #define wave(i,j) wave[(i) - 1 + n_max_compute_ham * ((j) - 1)]
#define wave(i, j) wave[((i)-1) * n_points + ((j)-1)]

#define dVxc_drho(i, j) dVxc_drho[(i)-1 + 3 * ((j)-1)]
#define first_order_rho(i, j) first_order_rho[(i)-1 + n_spin * ((j)-1)]

#define contract(i, j) contract[(i)-1 + n_points * ((j)-1)]
#define wave_t(i, j) wave_t[(i)-1 + n_points * ((j)-1)]
#define first_order_H_dense(i, j, k) \
    first_order_H_dense[(i)-1 + n_compute_c * ((j)-1 + ((k)-1) * n_compute_c)]
    __syncthreads();
    // not use_gga
    double *first_order_H_dense_local = first_order_H_dense;
    const double *wave_local = wave;
    double *grid_coord_local = grid_coord;
    int *ins_idx_local = ins_idx;
    first_order_H_dense = first_order_H_dense_local + bid_local * n_max_compute_ham * n_max_compute_ham * n_spin;
    wave = wave_local + bid_local * n_max_batch_size * n_max_compute_ham;
    grid_coord = grid_coord_local + bid_local * n_max_batch_size;
    if (n_spin == 1)
    {
        int i_spin = 1;
        // for(int i=lid; i<n_compute_c * n_compute_c * n_spin; i+=lsize)
        //   first_order_H_dense[i] = 0;
        // for (int i_point = lid + 1; i_point <= n_points; i_point += lsize)
        // {
        // for (int i_point = lid - (total_points - n_points) + 1; i_point <= n_points; i_point += lsize)
        // {
        int i_point = lid - (total_points - n_points) + 1;
        if (i_point <= n_points)
        {
            grid_coord[i_point - 1] =
                partition_tab[i_point - 1] *
                (-grid_coord[i_point - 1] + v_hartree_gradient[i_point - 1] +
                 dVxc_drho(i_spin, i_point) * first_order_rho(i_spin, i_point));
        }
        // }
        __syncthreads();

        for (int i_batch = 0; i_batch < batch_nums; i_batch++)
        // for (int i_batch = 0; i_batch < 2; i_batch++)
        {
            int pos_id = blockIdx.x * MERGE_BATCH + i_batch;
            first_order_H_dense = first_order_H_dense_local + pos_id * n_max_compute_ham * n_max_compute_ham * n_spin;
            wave = wave_local + pos_id * n_max_batch_size * n_max_compute_ham;
            grid_coord = grid_coord_local + pos_id * n_max_batch_size;
            // int index = threadIdx.x;
            // int totalThreads = 256;
            n_compute_c = n_compute_c_all[batch_start + i_batch - 1];
            n_points = n_points_all[batch_start + i_batch - 1];

            // 使用循环使每个线程处理多个元素
            // for (int idx = index; idx < n_compute_c * n_compute_c; idx += totalThreads)
            // {

            //     int row = idx / n_compute_c;
            //     int col = idx % n_compute_c;

            //     if (row < n_compute_c && col < n_compute_c)
            //     {
            //         double sum = 0.0;
            //         for (int k = 0; k < n_points; k++)
            //         {9999gg
            //             sum += grid_coord[k] * wave(row + 1, k + 1) * wave(col + 1, k + 1);
            //             // sum += contract(k + 1, row + 1) * wave(col + 1, k + 1);
            //         }
            //         first_order_H_dense(col + 1, row + 1, i_spin) = sum;
            //     }
            // }

            const int M = n_compute_c;
            const int N = n_compute_c;
            const int K = n_points;
            int lid = threadIdx.x;
            int lsize = blockDim.x;

            // __shared__ m_float_type A_local[TILE_M * WORK_M][TILE_K];
            // __shared__ m_float_type B_local[TILE_K][TILE_N * WORK_N];

            for (int m_out = 0; m_out < M; m_out += TILE_M * WORK_M)
            {
                for (int n_out = 0; n_out < N; n_out += TILE_N * WORK_N)
                {
                    int m_in = lid / TILE_M;
                    int n_in = lid % TILE_N;
                    m_float_type sum[WORK_M][WORK_N];
                    for (int i = 0; i < WORK_M; i++)
                    {
                        for (int j = 0; j < WORK_N; j++)
                        {
                            sum[i][j] = 0;
                        }
                    }
                    m_float_type B_regs[WORK_N];
                    for (int k_out = 0; k_out < K; k_out += TILE_K)
                    {
                        {
                            {
                                int m_in = lid / TILE_K;
                                int k_in = lid % TILE_K;
#pragma unroll WORK_M
                                for (int i = 0; i < WORK_M; i++)
                                {
                                    m_float_type val =
                                        wave((m_out + m_in + TILE_M * i) + 1, (k_out + k_in) + 1);
                                    bool cond =
                                        (m_out + m_in + TILE_M * i) >= M || (k_out + k_in) >= K;
                                    val = cond ? 0.0 : val;
                                    A_local[m_in + TILE_M * i][k_in] = val;
                                }
                            }

                            {
                                int k_in = lid / TILE_K;
                                int n_in = lid % TILE_K;
#pragma unroll WORK_N
                                for (int i = 0; i < WORK_N; i++)
                                {
                                    m_float_type val =
                                        grid_coord[(k_out + k_in)] *
                                        wave((n_out + n_in + TILE_N * i) + 1, (k_out + k_in) + 1);
                                    bool cond =
                                        (n_out + n_in + TILE_N * i) >= N || (k_out + k_in) >= K;
                                    val = cond ? 0.0 : val;
                                    B_local[k_in][n_in + TILE_N * i] = val;
                                }
                            }
                        }

                        __syncthreads();

                        for (int k = 0; k < TILE_K; k++)
                        {
                            for (int j = 0; j < WORK_N; j++)
                            {
                                B_regs[j] = B_local[k][n_in + TILE_N * j];
                            }
                            for (int i = 0; i < WORK_M; i++)
                            {
                                m_float_type A_reg = A_local[m_in + TILE_M * i][k];
                                for (int j = 0; j < WORK_N; j++)
                                {
                                    sum[i][j] += A_reg * B_regs[j];
                                }
                            }
                        }

                        __syncthreads();
                    }

                    for (int i = 0; i < WORK_M; i++)
                    {
                        for (int j = 0; j < WORK_N; j++)
                        {
                            if ((m_out + m_in + TILE_M * i) < M &&
                                (n_out + n_in + TILE_N * j) < N)
                                // C_group[(m_out + m_in + TILE_M * i) * N + (n_out + n_in +
                                // TILE_N * j)] = sum[i][j];
                                first_order_H_dense((n_out + n_in + TILE_N * j) + 1,
                                                    (m_out + m_in + TILE_M * i) + 1, i_spin) =
                                    sum[i][j];
                        }
                    }
                }
            }

            __syncthreads();
            ins_idx = ins_idx_local + n_basis_local * (batch_start + i_batch - 1);
            for (int i = 0; i < n_compute_c; i++)
            {
                int i_off = (ins_idx[i] * (ins_idx[i] - 1)) / 2;
                for (int j = lid; j <= i; j += lsize)
                {
                    atomicAdd(&first_order_H[ins_idx[j] + i_off - 1],
                              first_order_H_dense(j + 1, i + 1, i_spin));
                }
            }
            __syncthreads();
        }
    }
    // if (gid == 0)
    //     printf("end evaluate_first_order_h_polar_reduce_memory_c_: gid is : %d\n", gid);

#undef wave
#undef dVxc_drho
#undef first_order_rho

#undef contract
#undef wave_t
#undef first_order_H_dense
}

#define centers_hartree_potential(i) centers_hartree_potential[(i)-1]
#define center_to_atom(i) center_to_atom[(i)-1]
#define species_center(i) species_center[(i)-1]
#define center_to_cell(i) center_to_cell[(i)-1]
#define centers_basis_integrals centers_basis_integrals
#define Cbasis_to_basis(i) cbasis_to_basis[(i)-1]
#define Cbasis_to_center(i) cbasis_to_center[(i)-1]
#define pbc_lists_coords_center(i, j) pbc_lists_coords_center[((j)-1) * 3 + (i)-1]
#define column_index_hamiltonian(i) column_index_hamiltonian[(i)-1]
#define index_hamiltonian(i, j, k) index_hamiltonian[(((k)-1) * index_hamiltonian_dim2 + (j)-1) * 2 + (i)-1]
#define position_in_hamiltonian(i, j) position_in_hamiltonian[((i)-1) + ((j)-1) * position_in_hamiltonian_dim1]

#define n_grid(i) n_grid[(i)-1]
#define r_grid_min(i) r_grid_min[(i)-1]
#define log_r_grid_inc(i) log_r_grid_inc[(i)-1]

#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i)-1]
#define outer_radius_sq(i) outer_radius_sq[(i)-1]
#define basis_fn(i) basis_fn[(i)-1]
#define basis_l(i) basis_l[(i)-1]
#define atom_radius_sq(i) atom_radius_sq[(i)-1]
#define basis_fn_start_spl(i) basis_fn_start_spl[(i)-1]
#define basis_fn_atom(i, j) basis_fn_atom[(i)-1 + ((j)-1) * n_basis_fns]

#define batches_size_rho(i) batches_size_rho[(i)-1]
#define batches_batch_n_compute_rho(i) batches_batch_n_compute_rho[(i)-1]
// #define batches_batch_i_basis_rho(i, j) batches_batch_i_basis_rho[(i)-1 + n_centers_basis_I * ((j)-1)]
#define batches_batch_i_basis_rho(i, j) batches_batch_i_basis_rho[(i)-1 + n_max_compute_dens * ((j)-1)]
#define batches_points_coords_rho(i, j, k) batches_points_coords_rho[(((k)-1) * n_max_batch_size + (j)-1) * 3 + (i)-1]

#define batches_size_h(i) batches_size_h[(i)-1]
#define batches_batch_n_compute_h(i) batches_batch_n_compute_h[(i)-1]
#define batches_batch_i_basis_h(i, j) batches_batch_i_basis_h[(i)-1 + n_max_compute_dens * ((j)-1)]
#define batches_points_coords_h(i, j, k) batches_points_coords_h[(((k)-1) * n_max_batch_size + (j)-1) * 3 + (i)-1]

#define center_to_atom(i) center_to_atom[(i)-1]
#define species_center(i) species_center[(i)-1]
#define Cbasis_to_basis(i) cbasis_to_basis[(i)-1]
#define Cbasis_to_center(i) cbasis_to_center[(i)-1]
#define n_grid(i) n_grid[(i)-1]
#define r_grid_min(i) r_grid_min[(i)-1]
#define log_r_grid_inc(i) log_r_grid_inc[(i)-1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i)-1]
#define outer_radius_sq(i) outer_radius_sq[(i)-1]
#define basis_fn(i) basis_fn[(i)-1]
#define basis_l(i) basis_l[(i)-1]
#define atom_radius_sq(i) atom_radius_sq[(i)-1]
#define basis_fn_start_spl(i) basis_fn_start_spl[(i)-1]
#define basis_fn_atom(i, j) basis_fn_atom[(i)-1 + ((j)-1) * n_basis_fns]

__global__ void integrate_first_order_h_sub_tmp2_(
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
    double *first_order_H_dense__, int max_n_batch_centers, int *new_batch_count, int *new_batch_i_start, int n_new_batch_nums)
{

    unsigned long long H_thread_start,H_thread_end,H_thread_time;
    unsigned long long H_matrix_cal_start,H_matrix_cal_end,H_matrix_cal_time; 

    


    //--------------------------------------------------
    // if(gid == 0) printf("at integrate_first_order_h_sub_tmp2_ top ");
    int lid = threadIdx.x;
    int lsize = blockDim.x;
    // if (threadIdx.x == 0)
    // if(gid == 0) printf("at integrate_first_order_h_sub_tmp2_ : gid is : %d, lid is : %d, bid is : %d\n", gid, lid, bid);
    __shared__ double A_local[TILE_M * WORK_M][TILE_K];
    __shared__ double B_local[TILE_K][TILE_N * WORK_N];
    // __shared__ int tmp_shared_arr[1280];
#define dist_tab_sq(i) dist_tab_sq[(i)-1]
#define dist_tab(i) dist_tab[(i)-1]
// #define dir_tab(i, j) dir_tab[(i)-1 + 3 * ((j)-1)]
// #define dir_tab(i, j) dir_tab[lid + lsize * ((i)-1 + 3 * ((j)-1))]
#define wave(i, j) wave[(i)-1 + n_max_compute_ham * ((j)-1)]
#define batch_center_all_batches(i, j) batch_center_all_batches[(i)-1 + max_n_batch_centers * ((j)-1)]
#define batch_point_to_i_full_point(i, j) batch_point_to_i_full_point[(i)-1 + n_max_batch_size * ((j)-1)]

    //-----------------------------------
    double *wave_group;
    double *i_r;
    double *trigonom_tab;
    double *radial_wave;
    double *aux_radial;
    double *dylm_dtheta_tab;
    double *scaled_dylm_dphi_tab;
    double *grid_coord_group;
    double *first_order_H_dense_group;
    for (int i_batch_run = blockIdx.x + 1; i_batch_run <= n_new_batch_nums; i_batch_run += gridDim.x)
    {
        H_thread_start = clock64();

        int total_points = 0;
        int count_batch = 0;//逻辑批次中的第几个实际批次
        int i_my_batch = new_batch_i_start[i_batch_run - 1];
        while (lid >= total_points && i_my_batch < new_batch_i_start[i_batch_run - 1] + new_batch_count[i_batch_run - 1])
        {
            total_points += n_points_all_batches[i_my_batch - 1];
            i_my_batch++;
            count_batch++;
        }
        i_my_batch--; // 调整batch_id以反映正确的batch
        count_batch--;
        int i_point = lid - (total_points - n_points_all_batches[i_my_batch - 1]) + 1;
        int bid = MERGE_BATCH * blockIdx.x + count_batch;
        int gid = bid * lsize + i_point - 1;
        int gsize = MERGE_BATCH * gridDim.x * blockDim.x;

        wave_group = wave__ + bid * n_max_batch_size * n_max_compute_ham;
        i_r = i_r__ + gid * n_max_compute_atoms;
        trigonom_tab = trigonom_tab__ + gid * 4 * n_max_compute_atoms;
        radial_wave = radial_wave__ + gid * n_max_compute_fns_ham;
        aux_radial = aux_radial__ + gid * n_max_compute_atoms * n_basis_fns;
        dylm_dtheta_tab = dylm_dtheta_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);
        scaled_dylm_dphi_tab = dylm_dtheta_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);
        grid_coord_group = grid_coord__ + bid * n_max_batch_size;
        first_order_H_dense_group = first_order_H_dense__ + bid * n_max_compute_ham * n_max_compute_ham * n_spin;

        int n_compute_c = batches_batch_n_compute_h(i_my_batch);
        // if (lid == 0)
        //     printf("bid is : %d, i_my_batch is : %d, n_compute_c is : %d\n", bid, i_my_batch, n_compute_c);
        //     // if (gid == 0)

        //     //     printf("loop 1 outer: gid is : %d, lid is : %d, bid is : %d, n_compute_c is : %d, i_my_batch is : %d \n", gid, lid, bid, n_compute_c, i_my_batch);
        if (n_compute_c > 0)
        {
            // if (gid == 0)
            //     printf("n_compute_c > 0: gid is : %d, lid is : %d, bid is : %d, n_compute_c is : %d, i_my_batch is : %d\n", gid, lid, bid, n_compute_c, i_my_batch);

            if (i_point <= n_points_all_batches[i_my_batch - 1])
            {
                // if (gid == 0)
                //     printf("i_point <= n_points_all_batches[i_my_batch - 1]: gid is : %d, lid is : %d, bid is : %d, i_point is : %d\n", gid, lid, bid, i_point);
                double *dist_tab_sq = dist_tab_sq__ + gid * n_max_compute_atoms;
                double *dist_tab = dist_tab__ + gid * n_max_compute_atoms;
                double *dir_tab = dir_tab__ + bid * lsize * 3 * n_max_compute_atoms;
                int *atom_index = atom_index__ + gid * n_max_compute_atoms; // use private instead

                int *spline_array_start = spline_array_start__ + gid * n_max_compute_atoms; // use private instead
                int *spline_array_end = spline_array_end__ + gid * n_max_compute_atoms;
                // int *rad_index = rad_index__ + gid * n_max_compute_atoms;                    // use private instead
                // double *one_over_dist_tab = one_over_dist_tab__ + gid * n_max_compute_atoms; // use private instead

                int *wave_index = wave_index__ + gid * n_max_compute_fns_ham;

                int *l_count = l_count__ + gid * n_max_compute_fns_ham;
                // for (int i = 0; i < n_max_compute_fns_ham; i++)
                //     l_count[i] = 0;

                int *fn_atom = fn_atom__ + gid * n_max_compute_fns_ham;
                int *zero_index_point = zero_index_point__ + gid * n_max_compute_ham;

                int *rad_index = rad_index__ + gid * n_max_compute_atoms;
                double *one_over_dist_tab = one_over_dist_tab__ + gid * n_max_compute_atoms;

                for (int i = 0; i < n_basis_fns * (n_max_compute_atoms + 1); i++)
                    i_basis_fns_inv__[i * gsize + gid] = 0.0;

                double zora_operator[2];

                int n_compute_atoms = 0;
                int n_compute_fns = 0;
                int n_zero_compute;

                // grid_coord__[gid] = batches_points_coords_h(j_coord, i_point, i_my_batch);
                grid_coord_group[i_point - 1] = batches_points_coords_h(j_coord, i_point, i_my_batch);
                // if (threadIdx.x == 0)

                prune_radial_basis_p2_c_h(&n_max_compute_atoms, &n_max_compute_fns_ham, &dist_tab_sq(1), &dist_tab(1),
                                          dir_tab, // (3, n_atom_list)
                                          &n_compute_atoms, atom_index, ZERONULL, &n_compute_fns, ZERONULL,
                                          //  &n_compute_atoms, atom_index, atom_index_inv, &n_compute_fns, ZERONULL,
                                          i_basis_fns_inv__, // (n_basis_fns,n_centers)
                                          ZERONULL, spline_array_start, spline_array_end, &n_centers_integrals,
                                          centers_basis_integrals, &n_compute_c, &batches_batch_i_basis_h(1, i_my_batch),
                                          &n_batch_centers_all_batches[i_my_batch - 1], &batch_center_all_batches(1, i_my_batch),
                                          one_over_dist_tab, rad_index, wave_index, ZERONULL, l_count, fn_atom, &n_zero_compute,
                                          zero_index_point
                                          // outer
                                          ,
                                          n_basis_fns, &center_to_atom(1), &species_center(1), &Cbasis_to_basis(1),
                                          &Cbasis_to_center(1), &perm_basis_fns_spl(1), &outer_radius_sq(1), &basis_fn(1),
                                          &basis_l(1), &atom_radius_sq(1), &basis_fn_start_spl(1), &basis_fn_atom(1, 1),
                                          pbc_lists_coords_center,
                                          batches_points_coords_h(1, i_point, i_my_batch),
                                          batches_points_coords_h(2, i_point, i_my_batch),
                                          batches_points_coords_h(3, i_point, i_my_batch), gid, gsize, bid);

                tab_local_geometry_p2_c_(&n_compute_atoms, atom_index, &dist_tab(1),
                                         i_r
                                         // outer
                                         ,
                                         &species_center(1), &r_grid_min(1), &log_r_grid_inc(1));

                tab_gradient_ylm_p0_c_2(ZERONULL, basis_l_max, &l_ylm_max, &n_compute_atoms, atom_index, ylm_tab__,
                                        dylm_dtheta_tab, scaled_dylm_dphi_tab, dir_tab, &species_center(1), gid, gsize);
                int mfalse = 0;

                evaluate_radial_functions_p0_c_(
                    spline_array_start, spline_array_end, &n_compute_atoms, &n_compute_fns, &dist_tab(1), i_r, atom_index,
                    i_basis_fns_inv__, basis_wave_ordered, radial_wave, &mfalse, &n_compute_c,
                    &n_max_compute_fns_ham
                    // outer
                    ,
                    n_basis_fns, n_max_grid, &species_center(1), &n_grid(1), &perm_basis_fns_spl(1),
                    ZERONULL, gid, gsize, bid);

                evaluate_waves_p2_c_2(&n_compute_c, &n_compute_atoms, &n_compute_fns, &l_ylm_max, ylm_tab__, one_over_dist_tab,
                                      radial_wave, &wave_group[(i_point - 1)], rad_index, wave_index, ZERONULL, l_count, fn_atom,
                                      &n_zero_compute, zero_index_point, aux_radial, n_points_all_batches[i_my_batch - 1], gid, gsize);
            }
            // __syncthreads();

            H_thread_end = clock64();
            H_matrix_cal_start = clock64();
            double *H_times_psi_group = ZERONULL;
            first_order_H_dense_group = first_order_H_dense__;
            wave_group = wave__;
            grid_coord_group = grid_coord__;
            evaluate_first_order_h_polar_reduce_memory_c_(first_order_H, &n_points_all_batches[i_my_batch - 1],
                                                          &partition_all_batches[(i_my_batch - 1) * n_max_batch_size], grid_coord_group,
                                                          H_times_psi_group, &n_compute_c, &batches_batch_i_basis_h[n_centers_basis_I * (i_my_batch - 1)],
                                                          wave_group, ZERONULL,
                                                          &local_first_order_rho_all_batches[n_spin * n_max_batch_size * (i_my_batch - 1)],
                                                          &local_first_order_potential_all_batches[n_max_batch_size * (i_my_batch - 1)],
                                                          &local_dVxc_drho_all_batches[3 * n_max_batch_size * (i_my_batch - 1)],
                                                          ZERONULL, ZERONULL, ZERONULL, ZERONULL,
                                                          local_rho_gradient,
                                                          first_order_gradient_rho, &n_matrix_size,
                                                          ins_idx_all_batches, &n_basis_local, &n_spin, &n_max_compute_ham,
                                                          ZERONULL, ZERONULL, first_order_H_dense_group, A_local, B_local,
                                                          total_points, i_my_batch, bid, n_max_batch_size, new_batch_count[i_batch_run - 1], batches_batch_n_compute_h, n_points_all_batches, new_batch_i_start[i_batch_run - 1]);
            H_matrix_cal_end = clock64();

            H_thread_time = H_thread_end - H_thread_start;
            H_matrix_cal_time = H_matrix_cal_end - H_matrix_cal_start;

            float H_thread_time_ms = float(H_thread_time)/1225e3;
            float H_matrix_cal_time_ms = float(H_matrix_cal_time)/1225e3;
            float martix_cal_rate = float(H_matrix_cal_time)/float(H_thread_time + H_matrix_cal_time);
            if(gid <= 2)
            printf("thread cal time is:%f matrix cal time is:%f,matrix rate is:%f\n",H_thread_time_ms,H_matrix_cal_time_ms,martix_cal_rate);

        }



    }

    

    

#undef i_basis_fns_inv

    // m_save_check_h_(first_order_h);
#undef dist_tab_sq
#undef dist_tab
#undef dir_tab
#undef wave
#undef batch_center_all_batches
#undef batch_point_to_i_full_point
}
__device__ void prune_radial_basis_p2_c_h_pre(
    int *dim_atoms_, int *dim_fns_, double *dist_tab_sq,
    double *dist_tab,
    double *dir_tab, // (3, n_atom_list)
    int *n_compute_atoms_, int *atom_index, int *atom_index_inv,
    int *n_compute_fns_, int *i_basis_fns,
    int *i_basis_fns_inv, // (n_basis_fns,n_centers)
    int *i_atom_fns, int *spline_array_start,
    int *spline_array_end, int *n_atom_list_, int *atom_list,
    int *n_compute_, const int *i_basis, int *n_batch_centers_,
    int *batch_center, double *one_over_dist_tab,
    int *rad_index, int *wave_index, int *l_index,
    int *l_count, int *fn_atom, int *n_zero_compute_,
    int *zero_index_point,
    // outer
    int n_basis_fns, // const
    const int *center_to_atom, const int *species_center,
    const int *Cbasis_to_basis, const int *Cbasis_to_center,
    const int *perm_basis_fns_spl, const double *outer_radius_sq,
    const int *basis_fn, const int *basis_l,
    const double *atom_radius_sq, const int *basis_fn_start_spl,
    const int *basis_fn_atom, double *pbc_lists_coords_center,
    double coords_center0, double coords_center1, double coords_center2, int *diverge_matrix_local, int *counter_, int n_max_compute_ham, int max_n_batch_centers, int n_max_compute_atoms, int gid_trans, int gsize_trans)
{

    int gid = gid_trans;
    int gsize = gsize_trans;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
// #define dir_tab(i, j) dir_tab[(i)-1 + 3 * ((j)-1)]
#define dir_tab(i, j) dir_tab[lid + lsize * ((i) - 1 + 3 * ((j) - 1))]
// #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]
#define i_basis_fns_inv(i, j) \
    i_basis_fns_inv[((i) - 1 + n_basis_fns * ((j) - 1)) * gsize + gid]
// outer
#define center_to_atom(i) center_to_atom[(i) - 1]
#define species_center(i) species_center[(i) - 1]
#define Cbasis_to_basis(i) Cbasis_to_basis[(i) - 1]
#define Cbasis_to_center(i) Cbasis_to_center[(i) - 1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i) - 1]
#define outer_radius_sq(i) outer_radius_sq[(i) - 1]
#define basis_fn(i) basis_fn[(i) - 1]
#define basis_l(i) basis_l[(i) - 1]
#define atom_radius_sq(i) atom_radius_sq[(i) - 1]
#define basis_fn_start_spl(i) basis_fn_start_spl[(i) - 1]
#define basis_fn_atom(i, j) basis_fn_atom[(i) - 1 + ((j) - 1) * n_basis_fns]
#define pbc_lists_coords_center(i, j) \
    pbc_lists_coords_center[((j) - 1) * 3 + (i) - 1]
#define atom_index_inv(i) atom_index_inv[((i) - 1) * lsize + lid]

    // if (gid == 0)
    //     printf("start prune_radial_basis_p2_c_ : gid is : %d\n", gid);
    int dim_atoms = *dim_atoms_;
    int dim_fns = *dim_fns_;
    int n_atom_list = *n_atom_list_;
    int n_batch_centers = *n_batch_centers_;
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int n_zero_compute = *n_zero_compute_;

    // __shared__ variables
    int i_offset_spl;
    int l_aux;

    // counters
    int i_batch_center;
    int i_basis_1;
    int i_compute;
    int i_fn;
    int i_lm;
    int i_center, i_center_L;
    int i_atom_compute;
    int i_spline;

    // n_batch_centers = 50000;
    double dir_tab_local[3];
    for (i_batch_center = 1; i_batch_center <= n_batch_centers;
         i_batch_center++)
    {
        i_center_L = batch_center[i_batch_center - 1];
        i_center = atom_list[i_center_L - 1];

        dir_tab_local[0] =
            coords_center0 - pbc_lists_coords_center(1, atom_list[i_center_L - 1]);
        dir_tab_local[1] =
            coords_center1 - pbc_lists_coords_center(2, atom_list[i_center_L - 1]);
        dir_tab_local[2] =
            coords_center2 - pbc_lists_coords_center(3, atom_list[i_center_L - 1]);

        double dist_tab_sq_now = dir_tab_local[0] * dir_tab_local[0] +
                                 dir_tab_local[1] * dir_tab_local[1] +
                                 dir_tab_local[2] * dir_tab_local[2];

        // ???time
        if (dist_tab_sq_now <= atom_radius_sq(species_center(i_center)) &&
            dist_tab_sq_now > 0.0)
        {
            diverge_matrix_local[i_batch_center - 1] = 1;
            n_compute_atoms = n_compute_atoms + 1;

            atom_index[(n_compute_atoms - 1)] = i_center;
            // atom_index_inv(i_center) = n_compute_atoms;

            dist_tab_sq[n_compute_atoms - 1] = dist_tab_sq_now;

            dist_tab[n_compute_atoms - 1] = sqrt(dist_tab_sq[n_compute_atoms - 1]);

            double tmp = 1.0 / dist_tab[n_compute_atoms - 1];

            one_over_dist_tab[n_compute_atoms - 1] =
                1.0 / dist_tab[n_compute_atoms - 1];

            for (int i = 1; i <= 3; i++)
            {
                // dir_tab(i, n_compute_atoms) = dir_tab(i, i_center_L) * tmp;
                dir_tab(i, n_compute_atoms) = dir_tab_local[i - 1] * tmp;
            }
        }
    }
    int counter = *counter_;
    counter += n_max_compute_ham;
    int counter_local = counter;
    // __private__int private_center_to_atom[MACRO_n_centers+1];
    // // 确保非法的部分将被映射到从不会被使用的 MACRO_n_max_compute_atoms + 1
    // 上，由此保证其值为 0 for (int i=0; i<MACRO_n_centers+1; i++){
    //   private_center_to_atom[i] = MACRO_n_max_compute_atoms + 1;
    // }
    // next, check for radial basis functions

    for (i_atom_compute = 1; i_atom_compute <= n_compute_atoms;
         i_atom_compute++)
    {
        i_center = atom_index[(i_atom_compute - 1)];
        // private_center_to_atom[i_center] = i_atom_compute;
        i_offset_spl = basis_fn_start_spl(species_center(i_center));
        double dist_tab_sq_reg = dist_tab_sq[i_atom_compute - 1];
        for (i_spline = 1; i_spline <= n_basis_fns; i_spline++)
        {
            i_basis_1 = perm_basis_fns_spl(i_spline);
            if (basis_fn_atom(i_basis_1, center_to_atom(i_center)))
            {
                if (dist_tab_sq_reg <= outer_radius_sq(i_basis_1))
                {
                    diverge_matrix_local[counter + (i_atom_compute - 1) * n_basis_fns + i_spline - 1] = 1;
                    spline_array_start[i_atom_compute - 1] = i_spline;
                    break;
                }
            }
        }

        for (i_basis_1 = 1; i_basis_1 <= n_basis_fns; i_basis_1++)
        {
            if (basis_fn_atom(i_basis_1, center_to_atom(i_center)))
            {
                diverge_matrix_local[counter + n_max_compute_atoms * n_basis_fns + (i_atom_compute - 1) * n_basis_fns + i_basis_1 - 1] = 1;

                if (dist_tab_sq_reg <= outer_radius_sq(i_basis_1))
                {
                    diverge_matrix_local[counter + 2 * n_max_compute_atoms * n_basis_fns + (i_atom_compute - 1) * n_basis_fns + i_basis_1 - 1] = 1;
                    n_compute_fns = n_compute_fns + 1;
                    i_basis_fns_inv(i_basis_1, i_atom_compute) = n_compute_fns;
                    fn_atom[n_compute_fns - 1] = i_atom_compute;
                }
                i_offset_spl = i_offset_spl + 1;
            }
        }
        rad_index[i_atom_compute - 1] = n_compute_fns;
        spline_array_end[i_atom_compute - 1] = i_offset_spl - 1;
    }
    counter = counter_local + 3 * n_max_compute_atoms * n_basis_fns;

    // athread 版本不考虑下面这种情况，因此保持一致 WARNING
    // // 担忧出现 i_center = atom_index[(i_atom_compute - 1)]; 将不同的
    // i_atom_compute 映射到相同的 i_center
    // // 为了确保后文其他函数调用是正常，额外传一个 i_basis_fns_inv_remapping
    // 出去
    // // 暂时不选用重新给 i_basis_fns_inv 赋值的方式
    // for (i_atom_compute = 1; i_atom_compute <= n_compute_atoms;
    // i_atom_compute++) {
    //   i_basis_fns_inv_remapping[i_atom_compute-1] =
    //   private_center_to_atom[atom_index[(i_atom_compute - 1)]];
    // }

    n_zero_compute = 0;
    for (int i = 1; i <= n_compute_fns; i++)
    {
        wave_index[i - 1] = 0;
    }

    i_compute = 1;

    while (i_compute <= n_compute)
    {
        i_basis_1 = i_basis[i_compute - 1];
        int cc = dim_atoms + 1;
        int i_center = Cbasis_to_center(i_basis_1);
        for (int i = 0; i < dim_atoms; i++)
        {
            if (atom_index[i] == i_center)
            {
                cc = i + 1;
                break;
            }
        }
        i_fn = i_basis_fns_inv(basis_fn(Cbasis_to_basis(i_basis_1)), cc);
        l_aux = basis_l(Cbasis_to_basis(i_basis[i_compute - 1]));
        if (i_fn == 0)
        {
            diverge_matrix_local[counter + i_compute - 1] = 1;
            for (i_lm = 0; i_lm <= 2 * l_aux; i_lm++)
            {
                n_zero_compute = n_zero_compute + 1;
                zero_index_point[n_zero_compute - 1] = i_compute + i_lm;
            }
        }
        else if (wave_index[i_fn - 1] == 0)
        {
            diverge_matrix_local[counter + n_max_compute_ham + i_compute - 1] = 1;

            wave_index[i_fn - 1] = i_compute;
            l_count[i_fn - 1] = l_aux;
        }
        i_compute = i_compute + 2 * l_aux + 1;
    }
    counter += 2 * n_max_compute_ham;
    *counter_ = counter;
    *n_compute_atoms_ = n_compute_atoms;
    *n_compute_fns_ = n_compute_fns;
    *n_zero_compute_ = n_zero_compute;

#undef center_to_atom
#undef species_center
#undef Cbasis_to_basis
#undef Cbasis_to_center
#undef perm_basis_fns_spl
#undef outer_radius_sq
#undef basis_fn
#undef basis_l
#undef atom_radius_sq
#undef basis_fn_start_spl
#undef basis_fn_atom
#undef pbc_lists_coords_center
#undef atom_index_inv

#undef dir_tab
#undef i_basis_fns_inv
}

__device__ void tab_local_geometry_p2_c_pre(int *n_compute_atoms_, int *atom_index,
                                         double *dist_tab, double *i_r,
                                         // outer
                                         const int *species_center,
                                         double *r_grid_min,
                                         double *log_r_grid_inc, int *diverge_matrix_local)
{
#define species_center(i) species_center[(i) - 1]
#define r_grid_min(i) r_grid_min[(i) - 1]
#define log_r_grid_inc(i) log_r_grid_inc[(i) - 1]
    int gsize = gridDim.x * blockDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gid == 0)
    //     printf("start tab_local_geometry_p2_c_pre: gid is : %d\n", gid);
    int n_compute_atoms = *n_compute_atoms_;
    //  counters
    int i_compute_atom;
    for (i_compute_atom = 1; i_compute_atom <= n_compute_atoms;
         i_compute_atom++)
    {
        double r_current = dist_tab[i_compute_atom - 1];
        int i_species = species_center(atom_index[(i_compute_atom - 1)]);
        i_r[i_compute_atom - 1] = 1.0 + log(r_current / r_grid_min(i_species)) /
                                            log_r_grid_inc(i_species);
    }
    // if (gid == 0)
    //     printf("end tab_local_geometry_p2_c_pre: gid is : %d\n", gid);

#undef species_center
#undef r_grid_min
#undef log_r_grid_inc
}

__device__ void tab_gradient_ylm_p0_c_2_pre(
    double *trigonom_tab, // ( 4, n_compute_atoms )
    int *basis_l_max, int *l_ylm_max_, int *n_compute_atoms_,
    int *atom_index,
    double *ylm_tab,             // ( (l_ylm_max+1)**2, n_compute_atoms )
    double *dylm_dtheta_tab,     // ( (l_ylm_max+1)**2, n_compute_atoms )
    double *scaled_dylm_dphi_tab // ( (l_ylm_max+1)**2, n_compute_atoms )
    // outer
    ,
    double *dir_tab, const int *species_center, int *diverge_matrix_local, int *counter_, int n_max_compute_atoms, int gid_trans, int gsize_trans)
{
    int gid = gid_trans;
    int gsize = gsize_trans;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
#define species_center(i) species_center[(i) - 1]
// #define dir_tab(i, j) dir_tab[(i)-1 + ((j)-1) * 3]
#define dir_tab(i, j) dir_tab[lid + lsize * ((i) - 1 + ((j) - 1) * 3)]
    int n_compute_atoms = *n_compute_atoms_;
    int l_ylm_max = *l_ylm_max_;
    int l_ylm_max_1pow2 = (l_ylm_max + 1) * (l_ylm_max + 1);
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;
    // int lid = threadIdx.x;
    // int lsize = blockDim.x;

    // if (gid == 0)
    //     printf("start tab_gradient_ylm_p0_c_2_pre: gid is : %d\n", gid);
#define trigonom_tab(i, j) trigonom_tab[(i) - 1 + ((j) - 1) * 4]
#define ylm_tab(i, j) ylm_tab[(i - 1 + l_ylm_max_1pow2 * (j - 1)) * gsize + gid]
#define dylm_dtheta_tab(i, j) dylm_dtheta_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
#define scaled_dylm_dphi_tab(i, j) \
    scaled_dylm_dphi_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
    int counter = *counter_;
    for (int i_atom = 1; i_atom <= n_compute_atoms; i_atom++)
    {
        double trigonom_tab_reg[4];
        {
            //  __shared__ variables
            double abmax, abcmax, ab, abc;
            abmax = fmax(fabs(dir_tab(1, i_atom)), fabs(dir_tab(2, i_atom)));
            if (abmax > 0.0)
            {
                diverge_matrix_local[counter + i_atom - 1] = 1;
                ab = sqrt(pow(dir_tab(1, i_atom), 2.0) + pow(dir_tab(2, i_atom), 2.0));
                trigonom_tab_reg[3] = dir_tab(1, i_atom) / ab;
                trigonom_tab_reg[2] = dir_tab(2, i_atom) / ab;
            }
            else
            {
                diverge_matrix_local[counter + n_max_compute_atoms + i_atom - 1] = 1;
                trigonom_tab_reg[3] = 1.0;
                trigonom_tab_reg[2] = 0.0;
                ab = 0.0;
            }
            abcmax = fmax(abmax, fabs(dir_tab(3, i_atom)));
            if (abcmax > 0.0)
            {
                diverge_matrix_local[counter + 2 * n_max_compute_atoms + i_atom - 1] = 1;
                abc = sqrt(pow(ab, 2.0) + pow(dir_tab(3, i_atom), 2.0));
                trigonom_tab_reg[1] = dir_tab(3, i_atom) / abc;
                trigonom_tab_reg[0] = ab / abc;
            }
            else
            {
                diverge_matrix_local[counter + 3 * n_max_compute_atoms + i_atom - 1] = 1;
                trigonom_tab_reg[1] = 1.0;
                trigonom_tab_reg[0] = 0.0;
            }
        }
        //     increment_ylm_deriv(trigonom_tab(1, i_atom), trigonom_tab(2, i_atom),
        //     trigonom_tab(3, i_atom),
        //                         trigonom_tab(4, i_atom), 0,
        //                         basis_l_max(species_center(atom_index(i_atom))),
        //                         &ylm_tab(1, i_atom), &dylm_dtheta_tab(1, i_atom),
        //                         &scaled_dylm_dphi_tab(1, i_atom));
        if (1)
        {
            // SHEvalderiv_c_(basis_l_max[species_center(atom_index[i_atom - 1]) - 1],
            // trigonom_tab(1, i_atom),
            //                trigonom_tab(2, i_atom), trigonom_tab(3, i_atom),
            //                trigonom_tab(4, i_atom), &ylm_tab(1, i_atom),
            //                &dylm_dtheta_tab(1, i_atom), &scaled_dylm_dphi_tab(1,
            //                i_atom));
            {
                int l_atom_max =
                    basis_l_max[species_center(atom_index[(i_atom - 1)]) - 1];

                /* variables for tabulate ylm */
                double YLLI, YLL1I, YL1L1I, YLMI;
                double YLLR, YLL1R, YL1L1R, YLMR;
                int I2L, I4L2, INDEX, INDEX2, L, M, MSIGN;
                /* VB */
                int I22L, I24L2;
                double TEMP1, TEMP2, TEMP3;

                double D4LL1C, D2L13;
                const double PI = 3.14159265358979323846;

#define trigonom_tab_(i1) trigonom_tab_reg[i1 - 1]
#define ylm_tab_(i) ylm_tab(i, i_atom)
                if (0 <= 0)
                {
                    YLLR = 1.0 / sqrt(4.0 * PI);
                    YLLI = 0.0;
                    ylm_tab_(1) = YLLR;
                }

                if ((0 <= 1) && (l_atom_max >= 1))
                {
                    diverge_matrix_local[counter + 4 * n_max_compute_atoms + i_atom - 1] = 1;
                    ylm_tab_(3) = sqrt(3.00) * YLLR * trigonom_tab_(2);
                    TEMP1 = -sqrt(3.00) * YLLR * trigonom_tab_(1);
                    ylm_tab_(4) = TEMP1 * trigonom_tab_(4);
                    ylm_tab_(2) = -TEMP1 * trigonom_tab_(3);
                }

                // L = max(2,0)
                for (L = 2; L <= l_atom_max; L++)
                {
                    INDEX = L * L + 1;
                    INDEX2 = INDEX + 2 * L;
                    MSIGN = 1 - 2 * (L % 2);

                    YL1L1R = ylm_tab_(INDEX - 1);
                    YL1L1I = -MSIGN * ylm_tab_(INDEX - 2 * L + 1);
                    TEMP1 =
                        -sqrt((double)(2 * L + 1) / (double)(2 * L)) * trigonom_tab_(1);
                    YLLR =
                        TEMP1 * (trigonom_tab_(4) * YL1L1R - trigonom_tab_(3) * YL1L1I);
                    YLLI =
                        TEMP1 * (trigonom_tab_(4) * YL1L1I + trigonom_tab_(3) * YL1L1R);
                    ylm_tab_(INDEX2) = YLLR;
                    ylm_tab_(INDEX) = MSIGN * YLLI;
                    INDEX2 = INDEX2 - 1;
                    INDEX = INDEX + 1;

                    TEMP2 = sqrt((double)(2 * L + 1)) * trigonom_tab_(2);
                    YLL1R = TEMP2 * YL1L1R;
                    YLL1I = TEMP2 * YL1L1I;
                    ylm_tab_(INDEX2) = YLL1R;
                    ylm_tab_(INDEX) = -MSIGN * YLL1I;
                    INDEX2 = INDEX2 - 1;
                    INDEX = INDEX + 1;

                    I4L2 = INDEX - 4 * L + 2;
                    I2L = INDEX - 2 * L;
                    I24L2 = INDEX2 - 4 * L + 2;
                    I22L = INDEX2 - 2 * L;
                    D4LL1C = trigonom_tab_(2) * sqrt((double)(4 * L * L - 1));
                    D2L13 = -sqrt((double)(2 * L + 1) / (double)(2 * L - 3));

                    for (M = L - 2; M >= 0; M--)
                    {
                        TEMP1 = 1.00 / sqrt((double)((L + M) * (L - M)));
                        TEMP2 = D4LL1C * TEMP1;
                        TEMP3 = D2L13 * sqrt((double)((L + M - 1) * (L - M - 1))) * TEMP1;
                        YLMR = TEMP2 * ylm_tab_(I22L) + TEMP3 * ylm_tab_(I24L2);
                        YLMI = TEMP2 * ylm_tab_(I2L) + TEMP3 * ylm_tab_(I4L2);
                        ylm_tab_(INDEX2) = YLMR;
                        ylm_tab_(INDEX) = YLMI;

                        INDEX2 = INDEX2 - 1;
                        INDEX = INDEX + 1;
                        I24L2 = I24L2 - 1;
                        I22L = I22L - 1;
                        I4L2 = I4L2 + 1;
                        I2L = I2L + 1;
                    }
                }
#undef trigonom_tab_
#undef ylm_tab_
            }
        }
        else
        {
            // if(gid == 0) printf("%s, not finished\n", __func__); // TODO
            // exit(-19);
        }
    }
    const int REL_x2c = 7;
    const int REL_q4c = 8;
    // if (flag_rel == REL_q4c || flag_rel == REL_x2c) {
    if (0)
    {
        for (int i_atom = 1; i_atom <= n_compute_atoms; i_atom++)
        {
            if (l_ylm_max >= 1)
            {
                ylm_tab(4, i_atom) = -ylm_tab(4, i_atom);
                dylm_dtheta_tab(4, i_atom) = -dylm_dtheta_tab(4, i_atom);
                scaled_dylm_dphi_tab(4, i_atom) = -scaled_dylm_dphi_tab(4, i_atom);
            }
            if (l_ylm_max >= 2)
            {
                ylm_tab(8, i_atom) = -ylm_tab(8, i_atom);
                dylm_dtheta_tab(8, i_atom) = -dylm_dtheta_tab(8, i_atom);
                scaled_dylm_dphi_tab(8, i_atom) = -scaled_dylm_dphi_tab(8, i_atom);
            }
            if (l_ylm_max >= 3)
            {
                ylm_tab(14, i_atom) = -ylm_tab(14, i_atom);
                dylm_dtheta_tab(14, i_atom) = -dylm_dtheta_tab(14, i_atom);
                scaled_dylm_dphi_tab(14, i_atom) = -scaled_dylm_dphi_tab(14, i_atom);
                ylm_tab(16, i_atom) = -ylm_tab(16, i_atom);
                dylm_dtheta_tab(16, i_atom) = -dylm_dtheta_tab(16, i_atom);
                scaled_dylm_dphi_tab(16, i_atom) = -scaled_dylm_dphi_tab(16, i_atom);
            }
            if (l_ylm_max >= 4)
            {
                ylm_tab(22, i_atom) = -ylm_tab(22, i_atom);
                dylm_dtheta_tab(22, i_atom) = -dylm_dtheta_tab(22, i_atom);
                scaled_dylm_dphi_tab(22, i_atom) = -scaled_dylm_dphi_tab(22, i_atom);
                ylm_tab(24, i_atom) = -ylm_tab(24, i_atom);
                dylm_dtheta_tab(24, i_atom) = -dylm_dtheta_tab(24, i_atom);
                scaled_dylm_dphi_tab(24, i_atom) = -scaled_dylm_dphi_tab(24, i_atom);
            }
            if (l_ylm_max >= 5)
            {
                ylm_tab(32, i_atom) = -ylm_tab(32, i_atom);
                dylm_dtheta_tab(32, i_atom) = -dylm_dtheta_tab(32, i_atom);
                scaled_dylm_dphi_tab(32, i_atom) = -scaled_dylm_dphi_tab(32, i_atom);
                ylm_tab(34, i_atom) = -ylm_tab(34, i_atom);
                dylm_dtheta_tab(34, i_atom) = -dylm_dtheta_tab(34, i_atom);
                scaled_dylm_dphi_tab(34, i_atom) = -scaled_dylm_dphi_tab(34, i_atom);
                ylm_tab(36, i_atom) = -ylm_tab(36, i_atom);
                dylm_dtheta_tab(36, i_atom) = -dylm_dtheta_tab(36, i_atom);
                scaled_dylm_dphi_tab(36, i_atom) = -scaled_dylm_dphi_tab(36, i_atom);
            }
            if (l_ylm_max >= 6)
            {
                ylm_tab(44, i_atom) = -ylm_tab(44, i_atom);
                dylm_dtheta_tab(44, i_atom) = -dylm_dtheta_tab(44, i_atom);
                scaled_dylm_dphi_tab(44, i_atom) = -scaled_dylm_dphi_tab(44, i_atom);
                ylm_tab(46, i_atom) = -ylm_tab(46, i_atom);
                dylm_dtheta_tab(46, i_atom) = -dylm_dtheta_tab(46, i_atom);
                scaled_dylm_dphi_tab(46, i_atom) = -scaled_dylm_dphi_tab(46, i_atom);
                ylm_tab(48, i_atom) = -ylm_tab(48, i_atom);
                dylm_dtheta_tab(48, i_atom) = -dylm_dtheta_tab(48, i_atom);
                scaled_dylm_dphi_tab(48, i_atom) = -scaled_dylm_dphi_tab(48, i_atom);
            }
        }
    }
    // if (gid == 0)
    //     printf("end tab_gradient_ylm_p0_c_2_pre: gid is : %d\n", gid);
#undef trigonom_tab
#undef ylm_tab
#undef dylm_dtheta_tab
#undef scaled_dylm_dphi_tab

#undef species_center
#undef dir_tab
}

__device__ double spline_vector_waves_c_pre(double r_output, double *spl_param,
                                         int n_grid_dim, int n_compute_fns,
                                         int spline_start, int spline_end,
                                         int n_spl_points, int n_spline,
                                         double *out_wave, int index)
{
    // if (gid == 0)
    //     printf("start spline_vector_waves_c_pre: gid is : %d\n", gid);
#define spl_param(i, j, k) \
    spl_param[i - 1 + n_compute_fns * (j - 1 + 4 * (k - 1))]
#define out_wave(i) out_wave[i - 1]
    int i_spl;
    double t, term;
    int i_term;
    i_spl = (int)(r_output);
    i_spl = i_spl > 1 ? i_spl : 1;
    i_spl = i_spl < (n_spl_points - 1) ? i_spl : (n_spl_points - 1);
    // i_spl = fmax(1, i_spl);
    // i_spl = fmin(n_spl_points - 1, i_spl);
    t = r_output - (double)(i_spl);
    double ans = spl_param(index + 1 + spline_start - 1, 1, i_spl);
    // for (int i = 1; i <= n_spline; i++) {
    //   out_wave(i) = spl_param(i + spline_start - 1, 1, i_spl);
    // }
    term = 1.0;
    for (i_term = 2; i_term <= 4; i_term++)
    {
        term = term * t;
        // for (int i = 1; i <= n_spline; i++) {
        //   out_wave(i) = out_wave(i) + term * spl_param(i + spline_start - 1,
        //   i_term, i_spl);
        // }
        ans += term * spl_param(index + 1 + spline_start - 1, i_term, i_spl);
    }
    return ans;
    // if (gid == 0)
    //     printf("end spline_vector_waves_c_pre: gid is : %d\n", gid);
#undef spl_param
#undef out_wave
}

__device__ void evaluate_radial_functions_p0_c_pre(
    int *spline_array_start, int *spline_array_end,
    int *n_compute_atoms_, int *n_compute_fns_, double *dist_tab,
    double *i_r, int *atom_index,
    int *i_basis_fns_inv, // (n_basis_fns,n_centers)
    double *spline_data,  // (n_basis_fns,n_max_spline, n_max_grid)
    double *wave_aux, int *derivative_, int *n_compute_,
    int *n_basis_list_
    // outer
    ,
    int n_basis_fns, int n_max_grid, const int *species_center,
    int *n_grid,
    const int *perm_basis_fns_spl
    // tmp
    ,
    double *spline_array_aux, int *diverge_matrix_local, int gid_trans, int gsize_trans)
{

    int gid = gid_trans;
    int gsize = gsize_trans;
#define species_center(i) species_center[(i) - 1]
#define n_grid(i) n_grid[(i) - 1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i) - 1]

// #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]
#define i_basis_fns_inv(i, j) \
    i_basis_fns_inv[((i) - 1 + n_basis_fns * ((j) - 1)) * gsize + gid]
    // #define i_basis_fns_inv(i, j) \
//     i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * bid * gridDim.x * blockDim.x + gid]
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;

    // if (gid == 0)
    //     printf("before evaluate_radial_functions_p0_c_pre: gid is : %d\n", gid);
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int n_basis_list = *n_basis_list_;
    int derivative = *derivative_;
    // double spline_array_aux[n_basis_fns];
    for (int i_atom_1 = 1; i_atom_1 <= n_compute_atoms; i_atom_1++)
    {
        int current_atom = atom_index[(i_atom_1 - 1)];
        int current_species = species_center(current_atom);
        int spline_start = spline_array_start[i_atom_1 - 1];
        int spline_end = spline_array_end[i_atom_1 - 1];
        int n_spline = spline_end - spline_start + 1;
        double r_point = i_r[i_atom_1 - 1];
        double distance_from_atom = dist_tab[i_atom_1 - 1];
        // spline_vector_waves
        int i_rad = (spline_start - 1);
        for (int i_spline = spline_start; i_spline <= spline_end; i_spline++)
        {
            i_rad = i_rad + 1;
            int spline_counter = i_spline - (spline_start - 1);
            int current_basis_fn = perm_basis_fns_spl(i_rad);
            // int current_basis_fn_comp = i_basis_fns_inv(current_basis_fn,
            // current_atom);
            int current_basis_fn_comp = i_basis_fns_inv(current_basis_fn, i_atom_1);
            if (current_basis_fn_comp == 0)
                continue;
            // TODO 注意 spline_data 切片
            double tmp = spline_vector_waves_c_pre(r_point, spline_data, n_max_grid,
                                                n_basis_fns, spline_start, spline_end,
                                                n_grid(current_species), n_spline,
                                                spline_array_aux, spline_counter - 1);
            // if (derivative) {
            //   if (distance_from_atom > 0)
            //     wave_aux[current_basis_fn_comp - 1] =
            //     spline_array_aux[spline_counter - 1];
            //   else
            //     wave_aux[current_basis_fn_comp - 1] = 0.0;
            // } else {
            //   wave_aux[current_basis_fn_comp - 1] = spline_array_aux[spline_counter
            //   - 1];
            // }
            // if (derivative) {
            //   if (distance_from_atom > 0)
            //     wave_aux[current_basis_fn_comp - 1] = tmp;
            //   else
            //     wave_aux[current_basis_fn_comp - 1] = 0.0;
            // } else {
            //   wave_aux[current_basis_fn_comp - 1] = tmp;
            // }
            wave_aux[current_basis_fn_comp - 1] =
                (derivative && (distance_from_atom <= 0)) ? 0.0 : tmp;
        }
    }
    // if (gid == 0)
    //     printf("end evaluate_radial_functions_p0_c_pre: gid is : %d\n", gid);
#undef i_basis_fns_inv

#undef species_center
#undef n_grid
#undef perm_basis_fns_spl
}

__device__ void mul_vec_c_2_pre(double *wave, int n_mul, double *ylm,
                            double factor, int array_factor, int ylm_factor)
{
    for (int i = 0; i < n_mul; i++)
        wave[i * array_factor] = ylm[i * ylm_factor] * factor;
}

__device__ void evaluate_waves_p2_c_2_pre(
    int *n_compute_, int *n_compute_atoms_, int *n_compute_fns_,
    int *l_ylm_max_,
    double *ylm_tab, // ((l_ylm_max+1)**2, n_compute_atoms )
    double *one_over_dist_tab, double *radial_wave,
    double *wave, int *rad_index, int *wave_index,
    int *l_index, int *l_count, int *fn_atom,
    int *n_zero_compute_,
    int *zero_index_point
    // tmp
    ,
    double *aux_radial, int array_factor, int *diverge_matrix_local, int gid_trans, int gsize_trans)
{
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int gsize = gridDim.x * blockDim.x;
    int gid = gid_trans;
    int gsize = gsize_trans;
    // if (gid == 0)
    //     printf("before evaluate_waves_p2_c_2_pre: gid is : %d\n", gid);
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int l_ylm_max = *l_ylm_max_;
    int n_zero_compute = *n_zero_compute_;
    // double aux_radial[n_compute_fns];
    int index_start = 1;
    int index_end;
    int ylm_tab_dim1 = (l_ylm_max + 1) * (l_ylm_max + 1);
    for (int i_compute_atom = 1; i_compute_atom <= n_compute_atoms;
         i_compute_atom++)
    {
        index_end = rad_index[i_compute_atom - 1];
        for (int i = index_start; i <= index_end; i++)
            aux_radial[i - 1] =
                radial_wave[i - 1] * one_over_dist_tab[i_compute_atom - 1];
        index_start = index_end + 1;
    }

    for (int i_compute_fn = 1; i_compute_fn <= n_compute_fns; i_compute_fn++)
    {
        int l_aux = l_count[i_compute_fn - 1];
        int l_index_val = l_aux * l_aux + 1;
        int l_count_val = 2 * l_aux;
        mul_vec_c_2_pre(&wave[(wave_index[i_compute_fn - 1] - 1) * array_factor],
                    l_count_val + 1,
                    &ylm_tab[(l_index_val - 1 + (fn_atom[i_compute_fn - 1] - 1) * ylm_tab_dim1) * gsize + gid],
                    aux_radial[i_compute_fn - 1], array_factor, gsize);
    }
    for (int i_compute_point = 1; i_compute_point <= n_zero_compute;
         i_compute_point++)
    {
        int i_compute = zero_index_point[i_compute_point - 1];
        wave[(i_compute - 1) * array_factor] = 0.0;
    }
    // if (gid == 0)
    //     printf("end evaluate_waves_p2_c_2_pre: gid is : %d\n", gid);
}

__device__ void evaluate_first_order_h_polar_reduce_memory_c_pre(
    double *first_order_H, int *n_points_,
    double *partition_tab, // (n_points)
    double *grid_coord,    // (n_points)
    double *H_times_psi,   // (n_max_compute_ham,n_points,n_spin)
    int *n_compute_c_,
    const int *i_basis_index,    // (n_compute_c)
    const double *wave,          // (n_max_compute_ham, n_points)
    double *gradient_basis_wave, // (n_max_compute_ham, 3, n_points)
    double *first_order_rho,     // (n_spin, n_points)
    double *v_hartree_gradient,  // (n_points)
                                 // LDA
    double *dVxc_drho,           // (3,n_points)
                                 // GGA
    double *vsigma, double *v2rho2, double *v2rhosigma,
    double *v2sigma2, double *gradient_rho,
    double *first_order_gradient_rho,
    // ---
    int *n_matrix_size_,
    // use
    int *ins_idx,
    int *n_basis_local_, // 仅适用于使用了 local_index 且实际使用 ins_idx
                         // 转换矩阵的版本
    // outer
    int *n_spin_, int *n_max_compute_ham_,
    // tmp
    double *contract, double *wave_t,
    double *first_order_H_dense,
    // __shared__ double local_contract[H_PT_TILE][H_JC_TILE * H_JC_WORK],
    // __shared__ double local_wave[H_PT_TILE][H_IC_TILE * H_IC_WORK]
    // __shared__ double A_local[TILE_M * WORK_M][TILE_K],
    // __shared__ double B_local[TILE_K][TILE_N * WORK_N]
    double A_local[TILE_M * WORK_M][TILE_K],
    double B_local[TILE_K][TILE_N * WORK_N],
    int total_points, int i_my_batch, int bid_local, int n_max_batch_size,
    int batch_nums, int *n_compute_c_all, int *n_points_all, int batch_start)
{
    int n_compute_c = *n_compute_c_;
    int n_points = *n_points_;
    int n_basis_local = *n_basis_local_;
    int n_spin = *n_spin_;
    int n_max_compute_ham = *n_max_compute_ham_;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
    // if (gid == 0)
    //     printf("before evaluate_first_order_h_polar_reduce_memory_c_pre: gid is : %d\n", gid);
    // double contract[n_points * n_compute_c];
    // double wave_t[n_points * n_max_compute_ham];
    // double first_order_H_dense[n_compute_c * n_compute_c * n_spin];

    // __shared__ double local_contract[H_PT_TILE][H_JC_TILE * H_JC_WORK];
    // __shared__ double local_wave[H_PT_TILE][H_IC_TILE * H_IC_WORK];

// #define wave(i,j) wave[(i) - 1 + n_max_compute_ham * ((j) - 1)]
#define wave(i, j) wave[((i) - 1) * n_points + ((j) - 1)]

#define dVxc_drho(i, j) dVxc_drho[(i) - 1 + 3 * ((j) - 1)]
#define first_order_rho(i, j) first_order_rho[(i) - 1 + n_spin * ((j) - 1)]

#define contract(i, j) contract[(i) - 1 + n_points * ((j) - 1)]
#define wave_t(i, j) wave_t[(i) - 1 + n_points * ((j) - 1)]
#define first_order_H_dense(i, j, k) \
    first_order_H_dense[(i) - 1 + n_compute_c * ((j) - 1 + ((k) - 1) * n_compute_c)]
    __syncthreads();
    // not use_gga
    double *first_order_H_dense_local = first_order_H_dense;
    const double *wave_local = wave;
    double *grid_coord_local = grid_coord;
    int *ins_idx_local = ins_idx;
    first_order_H_dense = first_order_H_dense_local + bid_local * n_max_compute_ham * n_max_compute_ham * n_spin;
    wave = wave_local + bid_local * n_max_batch_size * n_max_compute_ham;
    grid_coord = grid_coord_local + bid_local * n_max_batch_size;
    if (n_spin == 1)
    {
        int i_spin = 1;
        // for(int i=lid; i<n_compute_c * n_compute_c * n_spin; i+=lsize)
        //   first_order_H_dense[i] = 0;
        // for (int i_point = lid + 1; i_point <= n_points; i_point += lsize)
        // {
        // for (int i_point = lid - (total_points - n_points) + 1; i_point <= n_points; i_point += lsize)
        // {
        int i_point = lid - (total_points - n_points) + 1;
        if (i_point <= n_points)
        {
            grid_coord[i_point - 1] =
                partition_tab[i_point - 1] *
                (-grid_coord[i_point - 1] + v_hartree_gradient[i_point - 1] +
                 dVxc_drho(i_spin, i_point) * first_order_rho(i_spin, i_point));
        }
        // }
        __syncthreads();

        for (int i_batch = 0; i_batch < batch_nums; i_batch++)
        // for (int i_batch = 0; i_batch < 2; i_batch++)
        {
            int pos_id = blockIdx.x * MERGE_BATCH + i_batch;
            first_order_H_dense = first_order_H_dense_local + pos_id * n_max_compute_ham * n_max_compute_ham * n_spin;
            wave = wave_local + pos_id * n_max_batch_size * n_max_compute_ham;
            grid_coord = grid_coord_local + pos_id * n_max_batch_size;
            // int index = threadIdx.x;
            // int totalThreads = 256;
            n_compute_c = n_compute_c_all[batch_start + i_batch - 1];
            n_points = n_points_all[batch_start + i_batch - 1];

            // 使用循环使每个线程处理多个元素
            // for (int idx = index; idx < n_compute_c * n_compute_c; idx += totalThreads)
            // {

            //     int row = idx / n_compute_c;
            //     int col = idx % n_compute_c;

            //     if (row < n_compute_c && col < n_compute_c)
            //     {
            //         double sum = 0.0;
            //         for (int k = 0; k < n_points; k++)
            //         {9999gg
            //             sum += grid_coord[k] * wave(row + 1, k + 1) * wave(col + 1, k + 1);
            //             // sum += contract(k + 1, row + 1) * wave(col + 1, k + 1);
            //         }
            //         first_order_H_dense(col + 1, row + 1, i_spin) = sum;
            //     }
            // }

            const int M = n_compute_c;
            const int N = n_compute_c;
            const int K = n_points;
            int lid = threadIdx.x;
            int lsize = blockDim.x;

            // __shared__ m_float_type A_local[TILE_M * WORK_M][TILE_K];
            // __shared__ m_float_type B_local[TILE_K][TILE_N * WORK_N];

            for (int m_out = 0; m_out < M; m_out += TILE_M * WORK_M)
            {
                for (int n_out = 0; n_out < N; n_out += TILE_N * WORK_N)
                {
                    int m_in = lid / TILE_M;
                    int n_in = lid % TILE_N;
                    m_float_type sum[WORK_M][WORK_N];
                    for (int i = 0; i < WORK_M; i++)
                    {
                        for (int j = 0; j < WORK_N; j++)
                        {
                            sum[i][j] = 0;
                        }
                    }
                    m_float_type B_regs[WORK_N];
                    for (int k_out = 0; k_out < K; k_out += TILE_K)
                    {
                        {
                            {
                                int m_in = lid / TILE_K;
                                int k_in = lid % TILE_K;
#pragma unroll WORK_M
                                for (int i = 0; i < WORK_M; i++)
                                {
                                    m_float_type val =
                                        wave((m_out + m_in + TILE_M * i) + 1, (k_out + k_in) + 1);
                                    bool cond =
                                        (m_out + m_in + TILE_M * i) >= M || (k_out + k_in) >= K;
                                    val = cond ? 0.0 : val;
                                    A_local[m_in + TILE_M * i][k_in] = val;
                                }
                            }

                            {
                                int k_in = lid / TILE_K;
                                int n_in = lid % TILE_K;
#pragma unroll WORK_N
                                for (int i = 0; i < WORK_N; i++)
                                {
                                    m_float_type val =
                                        grid_coord[(k_out + k_in)] *
                                        wave((n_out + n_in + TILE_N * i) + 1, (k_out + k_in) + 1);
                                    bool cond =
                                        (n_out + n_in + TILE_N * i) >= N || (k_out + k_in) >= K;
                                    val = cond ? 0.0 : val;
                                    B_local[k_in][n_in + TILE_N * i] = val;
                                }
                            }
                        }

                        __syncthreads();

                        for (int k = 0; k < TILE_K; k++)
                        {
                            for (int j = 0; j < WORK_N; j++)
                            {
                                B_regs[j] = B_local[k][n_in + TILE_N * j];
                            }
                            for (int i = 0; i < WORK_M; i++)
                            {
                                m_float_type A_reg = A_local[m_in + TILE_M * i][k];
                                for (int j = 0; j < WORK_N; j++)
                                {
                                    sum[i][j] += A_reg * B_regs[j];
                                }
                            }
                        }

                        __syncthreads();
                    }

                    for (int i = 0; i < WORK_M; i++)
                    {
                        for (int j = 0; j < WORK_N; j++)
                        {
                            if ((m_out + m_in + TILE_M * i) < M &&
                                (n_out + n_in + TILE_N * j) < N)
                                // C_group[(m_out + m_in + TILE_M * i) * N + (n_out + n_in +
                                // TILE_N * j)] = sum[i][j];
                                first_order_H_dense((n_out + n_in + TILE_N * j) + 1,
                                                    (m_out + m_in + TILE_M * i) + 1, i_spin) =
                                    sum[i][j];
                        }
                    }
                }
            }

            __syncthreads();
            ins_idx = ins_idx_local + n_basis_local * (batch_start + i_batch - 1);
            for (int i = 0; i < n_compute_c; i++)
            {
                int i_off = (ins_idx[i] * (ins_idx[i] - 1)) / 2;
                for (int j = lid; j <= i; j += lsize)
                {
                    atomicAdd(&first_order_H[ins_idx[j] + i_off - 1],
                              first_order_H_dense(j + 1, i + 1, i_spin));
                }
            }
            __syncthreads();
        }
    }
    // if (gid == 0)
    //     printf("end evaluate_first_order_h_polar_reduce_memory_c_pre: gid is : %d\n", gid);

#undef wave
#undef dVxc_drho
#undef first_order_rho

#undef contract
#undef wave_t
#undef first_order_H_dense
}

#define centers_hartree_potential(i) centers_hartree_potential[(i) - 1]
#define center_to_atom(i) center_to_atom[(i) - 1]
#define species_center(i) species_center[(i) - 1]
#define center_to_cell(i) center_to_cell[(i) - 1]
#define centers_basis_integrals centers_basis_integrals
#define Cbasis_to_basis(i) cbasis_to_basis[(i) - 1]
#define Cbasis_to_center(i) cbasis_to_center[(i) - 1]
#define pbc_lists_coords_center(i, j) pbc_lists_coords_center[((j) - 1) * 3 + (i) - 1]
#define column_index_hamiltonian(i) column_index_hamiltonian[(i) - 1]
#define index_hamiltonian(i, j, k) index_hamiltonian[(((k) - 1) * index_hamiltonian_dim2 + (j) - 1) * 2 + (i) - 1]
#define position_in_hamiltonian(i, j) position_in_hamiltonian[((i) - 1) + ((j) - 1) * position_in_hamiltonian_dim1]

#define n_grid(i) n_grid[(i) - 1]
#define r_grid_min(i) r_grid_min[(i) - 1]
#define log_r_grid_inc(i) log_r_grid_inc[(i) - 1]

#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i) - 1]
#define outer_radius_sq(i) outer_radius_sq[(i) - 1]
#define basis_fn(i) basis_fn[(i) - 1]
#define basis_l(i) basis_l[(i) - 1]
#define atom_radius_sq(i) atom_radius_sq[(i) - 1]
#define basis_fn_start_spl(i) basis_fn_start_spl[(i) - 1]
#define basis_fn_atom(i, j) basis_fn_atom[(i) - 1 + ((j) - 1) * n_basis_fns]

#define batches_size_rho(i) batches_size_rho[(i) - 1]
#define batches_batch_n_compute_rho(i) batches_batch_n_compute_rho[(i) - 1]
// #define batches_batch_i_basis_rho(i, j) batches_batch_i_basis_rho[(i)-1 + n_centers_basis_I * ((j)-1)]
#define batches_batch_i_basis_rho(i, j) batches_batch_i_basis_rho[(i) - 1 + n_max_compute_dens * ((j) - 1)]
#define batches_points_coords_rho(i, j, k) batches_points_coords_rho[(((k) - 1) * n_max_batch_size + (j) - 1) * 3 + (i) - 1]

#define batches_size_h(i) batches_size_h[(i) - 1]
#define batches_batch_n_compute_h(i) batches_batch_n_compute_h[(i) - 1]
#define batches_batch_i_basis_h(i, j) batches_batch_i_basis_h[(i) - 1 + n_max_compute_dens * ((j) - 1)]
#define batches_points_coords_h(i, j, k) batches_points_coords_h[(((k) - 1) * n_max_batch_size + (j) - 1) * 3 + (i) - 1]

#define center_to_atom(i) center_to_atom[(i) - 1]
#define species_center(i) species_center[(i) - 1]
#define Cbasis_to_basis(i) cbasis_to_basis[(i) - 1]
#define Cbasis_to_center(i) cbasis_to_center[(i) - 1]
#define n_grid(i) n_grid[(i) - 1]
#define r_grid_min(i) r_grid_min[(i) - 1]
#define log_r_grid_inc(i) log_r_grid_inc[(i) - 1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i) - 1]
#define outer_radius_sq(i) outer_radius_sq[(i) - 1]
#define basis_fn(i) basis_fn[(i) - 1]
#define basis_l(i) basis_l[(i) - 1]
#define atom_radius_sq(i) atom_radius_sq[(i) - 1]
#define basis_fn_start_spl(i) basis_fn_start_spl[(i) - 1]
#define basis_fn_atom(i, j) basis_fn_atom[(i) - 1 + ((j) - 1) * n_basis_fns]

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
    int i_batch_run)
{
    //--------------------------------------------------
    // if(gid == 0) printf("at integrate_first_order_h_sub_tmp2_ top ");

    int lid = threadIdx.x;
    int lsize = blockDim.x;
    // if (threadIdx.x == 0)
    // if(gid == 0) printf("at integrate_first_order_h_sub_tmp2_ : gid is : %d, lid is : %d, bid is : %d\n", gid, lid, bid);
    __shared__ double A_local[TILE_M * WORK_M][TILE_K];
    __shared__ double B_local[TILE_K][TILE_N * WORK_N];
    // __shared__ int count_batch_local[256][6];
    // for (int i = 0; i < 256; i++)
    // {
    //     for (int j = 0; j < 6; j++)
    //     {
    //         count_batch_local[i][j] = 0;
    //     }
    // }

#define dist_tab_sq(i) dist_tab_sq[(i) - 1]
#define dist_tab(i) dist_tab[(i) - 1]
// #define dir_tab(i, j) dir_tab[(i)-1 + 3 * ((j)-1)]
// #define dir_tab(i, j) dir_tab[lid + lsize * ((i)-1 + 3 * ((j)-1))]
#define wave(i, j) wave[(i) - 1 + n_max_compute_ham * ((j) - 1)]
#define batch_center_all_batches(i, j) batch_center_all_batches[(i) - 1 + max_n_batch_centers * ((j) - 1)]
#define batch_point_to_i_full_point(i, j) batch_point_to_i_full_point[(i) - 1 + n_max_batch_size * ((j) - 1)]

    //-----------------------------------
    double *wave_group;
    double *i_r;
    double *trigonom_tab;
    double *radial_wave;
    double *aux_radial;
    double *dylm_dtheta_tab;
    double *scaled_dylm_dphi_tab;
    double *grid_coord_group;
    double *first_order_H_dense_group;

    int total_points = 0;
    int count_batch = 0;
    int i_my_batch = new_batch_i_start[i_batch_run - 1];
    while (lid >= total_points && i_my_batch < new_batch_i_start[i_batch_run - 1] + new_batch_count[i_batch_run - 1])
    {
        total_points += n_points_all_batches[i_my_batch - 1];
        i_my_batch++;
        count_batch++;
    }
    i_my_batch--; // 调整batch_id以反映正确的batch
    count_batch--;
    int i_point = lid - (total_points - n_points_all_batches[i_my_batch - 1]) + 1;
    int bid = MERGE_BATCH * blockIdx.x + count_batch;
    int gid = bid * lsize + i_point - 1;
    int gsize = MERGE_BATCH * gridDim.x * blockDim.x;

    wave_group = wave__ + bid * n_max_batch_size * n_max_compute_ham;
    i_r = i_r__ + gid * n_max_compute_atoms;
    trigonom_tab = trigonom_tab__ + gid * 4 * n_max_compute_atoms;
    radial_wave = radial_wave__ + gid * n_max_compute_fns_ham;
    aux_radial = aux_radial__ + gid * n_max_compute_atoms * n_basis_fns;
    dylm_dtheta_tab = dylm_dtheta_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);
    scaled_dylm_dphi_tab = dylm_dtheta_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);
    grid_coord_group = grid_coord__ + bid * n_max_batch_size;
    first_order_H_dense_group = first_order_H_dense__ + bid * n_max_compute_ham * n_max_compute_ham * n_spin;

    int col_size = 3 * n_max_compute_ham + 3 * n_max_compute_atoms * (n_basis_fns + 1) + 2 * n_max_compute_atoms;
    int *diverge_matrix_ = diverge_matrix + lid * col_size;
    int counter = 0;
    int n_compute_c = batches_batch_n_compute_h(i_my_batch);
    // if (lid == 0)
    //     printf("bid is : %d, i_my_batch is : %d, n_compute_c is : %d\n", bid, i_my_batch, n_compute_c);
    //     // if (gid == 0)
    //     //     printf("loop 1 outer: gid is : %d, lid is : %d, bid is : %d, n_compute_c is : %d, i_my_batch is : %d \n", gid, lid, bid, n_compute_c, i_my_batch);
    if (n_compute_c > 0)
    {
        // if (gid == 0)
        //     printf("n_compute_c > 0: gid is : %d, lid is : %d, bid is : %d, n_compute_c is : %d, i_my_batch is : %d\n", gid, lid, bid, n_compute_c, i_my_batch);

        if (i_point <= n_points_all_batches[i_my_batch - 1])
        {
            // if (gid == 0)
            //     printf("i_point <= n_points_all_batches[i_my_batch - 1]: gid is : %d, lid is : %d, bid is : %d, i_point is : %d\n", gid, lid, bid, i_point);
            double *dist_tab_sq = dist_tab_sq__ + gid * n_max_compute_atoms;
            double *dist_tab = dist_tab__ + gid * n_max_compute_atoms;
            double *dir_tab = dir_tab__ + bid * lsize * 3 * n_max_compute_atoms;
            int *atom_index = atom_index__ + gid * n_max_compute_atoms; // use private instead

            int *spline_array_start = spline_array_start__ + gid * n_max_compute_atoms; // use private instead
            int *spline_array_end = spline_array_end__ + gid * n_max_compute_atoms;
            // int *rad_index = rad_index__ + gid * n_max_compute_atoms;                    // use private instead
            // double *one_over_dist_tab = one_over_dist_tab__ + gid * n_max_compute_atoms; // use private instead

            int *wave_index = wave_index__ + gid * n_max_compute_fns_ham;

            int *l_count = l_count__ + gid * n_max_compute_fns_ham;
            // for (int i = 0; i < n_max_compute_fns_ham; i++)
            //     l_count[i] = 0;

            int *fn_atom = fn_atom__ + gid * n_max_compute_fns_ham;
            int *zero_index_point = zero_index_point__ + gid * n_max_compute_ham;

            int *rad_index = rad_index__ + gid * n_max_compute_atoms;
            double *one_over_dist_tab = one_over_dist_tab__ + gid * n_max_compute_atoms;

            for (int i = 0; i < n_basis_fns * (n_max_compute_atoms + 1); i++)
                i_basis_fns_inv__[i * gsize + gid] = 0.0;

            double zora_operator[2];

            int n_compute_atoms = 0;
            int n_compute_fns = 0;
            int n_zero_compute;

            // grid_coord__[gid] = batches_points_coords_h(j_coord, i_point, i_my_batch);
            grid_coord_group[i_point - 1] = batches_points_coords_h(j_coord, i_point, i_my_batch);
            // if (threadIdx.x == 0)

            prune_radial_basis_p2_c_h_pre(&n_max_compute_atoms, &n_max_compute_fns_ham, &dist_tab_sq(1), &dist_tab(1),
                                      dir_tab, // (3, n_atom_list)
                                      &n_compute_atoms, atom_index, ZERONULL, &n_compute_fns, ZERONULL,
                                      //  &n_compute_atoms, atom_index, atom_index_inv, &n_compute_fns, ZERONULL,
                                      i_basis_fns_inv__, // (n_basis_fns,n_centers)
                                      ZERONULL, spline_array_start, spline_array_end, &n_centers_integrals,
                                      centers_basis_integrals, &n_compute_c, &batches_batch_i_basis_h(1, i_my_batch),
                                      &n_batch_centers_all_batches[i_my_batch - 1], &batch_center_all_batches(1, i_my_batch),
                                      one_over_dist_tab, rad_index, wave_index, ZERONULL, l_count, fn_atom, &n_zero_compute,
                                      zero_index_point
                                      // outer
                                      ,
                                      n_basis_fns, &center_to_atom(1), &species_center(1), &Cbasis_to_basis(1),
                                      &Cbasis_to_center(1), &perm_basis_fns_spl(1), &outer_radius_sq(1), &basis_fn(1),
                                      &basis_l(1), &atom_radius_sq(1), &basis_fn_start_spl(1), &basis_fn_atom(1, 1),
                                      pbc_lists_coords_center,
                                      batches_points_coords_h(1, i_point, i_my_batch),
                                      batches_points_coords_h(2, i_point, i_my_batch),
                                      batches_points_coords_h(3, i_point, i_my_batch),
                                      diverge_matrix_, &counter, n_max_compute_ham, max_n_batch_centers, n_max_compute_atoms, gid, gsize);

            tab_local_geometry_p2_c_pre(&n_compute_atoms, atom_index, &dist_tab(1),
                                     i_r
                                     // outer
                                     ,
                                     &species_center(1), &r_grid_min(1), &log_r_grid_inc(1), diverge_matrix_);

            tab_gradient_ylm_p0_c_2_pre(ZERONULL, basis_l_max, &l_ylm_max, &n_compute_atoms, atom_index, ylm_tab__,
                                    dylm_dtheta_tab, scaled_dylm_dphi_tab, dir_tab, &species_center(1), diverge_matrix_, &counter, n_max_compute_atoms, gid, gsize);
            int mfalse = 0;
            evaluate_radial_functions_p0_c_pre(
                spline_array_start, spline_array_end, &n_compute_atoms, &n_compute_fns, &dist_tab(1), i_r, atom_index,
                i_basis_fns_inv__, basis_wave_ordered, radial_wave, &mfalse, &n_compute_c,
                &n_max_compute_fns_ham
                // outer
                ,
                n_basis_fns, n_max_grid, &species_center(1), &n_grid(1), &perm_basis_fns_spl(1),
                ZERONULL, diverge_matrix_, gid, gsize);

            evaluate_waves_p2_c_2_pre(&n_compute_c, &n_compute_atoms, &n_compute_fns, &l_ylm_max, ylm_tab__, one_over_dist_tab,
                                  radial_wave, &wave_group[(i_point - 1)], rad_index, wave_index, ZERONULL, l_count, fn_atom,
                                  &n_zero_compute, zero_index_point, aux_radial, n_points_all_batches[i_my_batch - 1], diverge_matrix_, gid, gsize);
        }
        // __syncthreads();
        double *H_times_psi_group = ZERONULL;
        first_order_H_dense_group = first_order_H_dense__;
        wave_group = wave__;
        grid_coord_group = grid_coord__;
        evaluate_first_order_h_polar_reduce_memory_c_pre(first_order_H, &n_points_all_batches[i_my_batch - 1],
                                                      &partition_all_batches[(i_my_batch - 1) * n_max_batch_size], grid_coord_group,
                                                      H_times_psi_group, &n_compute_c, &batches_batch_i_basis_h[n_centers_basis_I * (i_my_batch - 1)],
                                                      wave_group, ZERONULL,
                                                      &local_first_order_rho_all_batches[n_spin * n_max_batch_size * (i_my_batch - 1)],
                                                      &local_first_order_potential_all_batches[n_max_batch_size * (i_my_batch - 1)],
                                                      &local_dVxc_drho_all_batches[3 * n_max_batch_size * (i_my_batch - 1)],
                                                      ZERONULL, ZERONULL, ZERONULL, ZERONULL,
                                                      local_rho_gradient,
                                                      first_order_gradient_rho, &n_matrix_size,
                                                      ins_idx_all_batches, &n_basis_local, &n_spin, &n_max_compute_ham,
                                                      ZERONULL, ZERONULL, first_order_H_dense_group, A_local, B_local,
                                                      total_points, i_my_batch, bid, n_max_batch_size, new_batch_count[i_batch_run - 1], batches_batch_n_compute_h, n_points_all_batches, new_batch_i_start[i_batch_run - 1]);
    }

#undef i_basis_fns_inv

    // m_save_check_h_(first_order_h);
#undef dist_tab_sq
#undef dist_tab
#undef dir_tab
#undef wave
#undef batch_center_all_batches
#undef batch_point_to_i_full_point
}

