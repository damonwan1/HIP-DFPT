#include <hip/hip_runtime.h>
#include "gpuAll.h"
#define ZERONULL 0
// #define atomic_cmpxchg atom_cmpxchg

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

__device__ void prune_radial_basis_p2_c_rho(
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
    double coords_center0, double coords_center1, double coords_center2)
{
// #define dir_tab(i, j) dir_tab[(i)-1 + 3 * ((j)-1)]
#define dir_tab(i, j) dir_tab[lid + lsize * ((i)-1 + 3 * ((j)-1))]
// #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]
#define i_basis_fns_inv(i, j) \
    i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * gsize + gid]
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

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int gsize = gridDim.x * blockDim.x;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
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
        else
        {
            // atom_index_inv[i_center - 1] = 0;
            // atom_index_inv[i_center - 1] = dim_atoms + 1; // dim_atoms ==
            // n_max_compute_atoms
        }
    }
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
                    spline_array_start[i_atom_compute - 1] = (i_spline - 1) + 1;
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
                    i_basis_fns_inv(i_basis_1, i_atom_compute) = n_compute_fns;
                    fn_atom[n_compute_fns - 1] = i_atom_compute;
                }
                i_offset_spl = i_offset_spl + 1;
            }
        }
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

__device__ void tab_gradient_ylm_p0_c_(double *trigonom_tab, // ( 4, n_compute_atoms )
                                       int *basis_l_max, int *l_ylm_max_, int *n_compute_atoms_, int *atom_index,
                                       double *ylm_tab,              // ( (l_ylm_max+1)**2, n_compute_atoms )
                                       double *dylm_dtheta_tab,      // ( (l_ylm_max+1)**2, n_compute_atoms )
                                       double *scaled_dylm_dphi_tab, // ( (l_ylm_max+1)**2, n_compute_atoms )
                                       // outer
                                       double *dir_tab,
                                       int *species_center)
{
#define species_center(i) species_center[(i)-1]
// #define dir_tab(i, j) dir_tab[(i)-1 + ((j)-1) * 3]
#define dir_tab(i, j) dir_tab[lid + lsize * ((i)-1 + ((j)-1) * 3)]
    int n_compute_atoms = *n_compute_atoms_;
    int l_ylm_max = *l_ylm_max_;
    int l_ylm_max_1pow2 = (l_ylm_max + 1) * (l_ylm_max + 1);
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int gsize = gridDim.x * blockDim.x;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
#define trigonom_tab(i, j) trigonom_tab[(i)-1 + ((j)-1) * 4]
#define ylm_tab(i, j) ylm_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
#define dylm_dtheta_tab(i, j) dylm_dtheta_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
#define scaled_dylm_dphi_tab(i, j) scaled_dylm_dphi_tab[i - 1 + l_ylm_max_1pow2 * (j - 1)]
    for (int i_atom = 1; i_atom <= n_compute_atoms; i_atom++)
    {
        double trigonom_tab_reg[4];
        {
            //  local variables
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
        //     increment_ylm_deriv(trigonom_tab(1, i_atom), trigonom_tab(2, i_atom), trigonom_tab(3, i_atom),
        //                         trigonom_tab(4, i_atom), 0, basis_l_max(species_center(atom_index(i_atom))),
        //                         &ylm_tab(1, i_atom), &dylm_dtheta_tab(1, i_atom), &scaled_dylm_dphi_tab(1, i_atom));
        if (1)
        {
            // SHEvalderiv_c_(basis_l_max[species_center(atom_index[i_atom - 1]) - 1], trigonom_tab(1, i_atom),
            //                trigonom_tab(2, i_atom), trigonom_tab(3, i_atom), trigonom_tab(4, i_atom), &ylm_tab(1, i_atom),
            //                &dylm_dtheta_tab(1, i_atom), &scaled_dylm_dphi_tab(1, i_atom));
            {
                int l_atom_max = basis_l_max[species_center(atom_index[(i_atom - 1)]) - 1];

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
                    TEMP1 = -sqrt((double)(2 * L + 1) / (double)(2 * L)) * trigonom_tab_(1);
                    YLLR = TEMP1 * (trigonom_tab_(4) * YL1L1R - trigonom_tab_(3) * YL1L1I);
                    YLLI = TEMP1 * (trigonom_tab_(4) * YL1L1I + trigonom_tab_(3) * YL1L1R);
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
            // printf("%s, not finished\n", __func__); // TODO
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
    double *spline_array_aux)
{
#define species_center(i) species_center[(i)-1]
#define n_grid(i) n_grid[(i)-1]
#define perm_basis_fns_spl(i) perm_basis_fns_spl[(i)-1]

// #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]
#define i_basis_fns_inv(i, j) \
    i_basis_fns_inv[((i)-1 + n_basis_fns * ((j)-1)) * gsize + gid]
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int gsize = gridDim.x * blockDim.x;
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

__device__ void mul_vec_c_(double *wave, int n_mul, double *ylm, double factor)
{
    for (int i = 0; i < n_mul; i++)
        wave[i] = ylm[i] * factor;
}

__device__ void evaluate_waves_p2_c_(int *n_compute_, int *n_compute_atoms_, int *n_compute_fns_, int *l_ylm_max_,
                                     double *ylm_tab, // ((l_ylm_max+1)**2, n_compute_atoms )
                                     double *one_over_dist_tab, double *radial_wave, double *wave, int *rad_index, int *wave_index,
                                     int *l_index, int *l_count, int *fn_atom, int *n_zero_compute_, int *zero_index_point,
                                     // tmp
                                     double *aux_radial)
{
    int n_compute = *n_compute_;
    int n_compute_atoms = *n_compute_atoms_;
    int n_compute_fns = *n_compute_fns_;
    int l_ylm_max = *l_ylm_max_;
    int n_zero_compute = *n_zero_compute_;
    // double aux_radial[n_compute_fns];
    int index_start = 1;
    int index_end;
    int ylm_tab_dim1 = (l_ylm_max + 1) * (l_ylm_max + 1);
    for (int i_compute_atom = 1; i_compute_atom <= n_compute_atoms; i_compute_atom++)
    {
        index_end = rad_index[i_compute_atom - 1];
        for (int i = index_start; i <= index_end; i++)
            aux_radial[i - 1] = radial_wave[i - 1] * one_over_dist_tab[i_compute_atom - 1];
        index_start = index_end + 1;
    }
    for (int i_compute_fn = 1; i_compute_fn <= n_compute_fns; i_compute_fn++)
    {
        int l_aux = l_count[i_compute_fn - 1];
        int l_index_val = l_aux * l_aux + 1;
        int l_count_val = 2 * l_aux;
        mul_vec_c_(&wave[wave_index[i_compute_fn - 1] - 1], l_count_val + 1,
                   &ylm_tab[l_index_val - 1 + (fn_atom[i_compute_fn - 1] - 1) * ylm_tab_dim1],
                   aux_radial[i_compute_fn - 1]);
    }
    for (int i_compute_point = 1; i_compute_point <= n_zero_compute; i_compute_point++)
    {
        int i_compute = zero_index_point[i_compute_point - 1];
        wave[i_compute - 1] = 0.0;
    }
}

__device__ void prune_density_matrix_sparse_polar_reduce_memory_local_index(double *density_matrix_sparse, double *density_matrix_con,
                                                                            int *n_compute_,
                                                                            int *ins_idx,
                                                                            int n_local_matrix_size)
{
    int n_compute_c = *n_compute_;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
    // if(lid == 0){
    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    __syncthreads();
    for (int i = lid; i < n_compute_c * n_compute_c; i += lsize)
    {
        density_matrix_con[i] = 0.0;
    }
    __syncthreads();
    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int i = 0; i < n_compute_c; i++)
    {
        int i_off = (ins_idx[i] * (ins_idx[i] - 1)) / 2;
        for (int j = lid; j <= i; j += lsize)
        {
            if (ins_idx[j] + i_off > n_local_matrix_size)
            {
                break;
            }
            else
            {
                double tmp = density_matrix_sparse[ins_idx[j] + i_off - 1];
                density_matrix_con[j + i * n_compute_c] = i == j ? tmp : tmp * 2;
                // density_matrix_con[i + j*n_compute_c] = density_matrix_sparse[ins_idx[j] + i_off - 1];
            }
        }
    }
    // }
    __syncthreads();
    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
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

__global__ void integrate_first_order_rho_sub_tmp2_(
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
    double *ylm_tab__, double *dylm_dtheta_tab__, double *scaled_dylm_dphi_tab__, int max_n_batch_centers)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int gsize = gridDim.x * blockDim.x;
    int bid = blockIdx.x;
    int lid = threadIdx.x;
    int lsize = blockDim.x;
    // int block_id = gid / lsize;

#define IC_TILE 16
#define JC_TILE 16
#define PT_TILE 16
#define IC_WORK 3
#define PT_WORK 3

    __shared__ double local_density[IC_TILE * IC_WORK][JC_TILE];
    __shared__ double local_wave[JC_TILE][PT_TILE * PT_WORK];
    __shared__ double local_tmp_rho[IC_TILE][PT_TILE * PT_WORK];

    // double dist_tab_sq[n_centers_integrals];
    // double dist_tab[n_centers_integrals];
    // double dir_tab[3 * n_centers_integrals];
#define dist_tab_sq(i) dist_tab_sq[(i)-1]
#define dist_tab(i) dist_tab[(i)-1]
// #define dir_tab(i, j) dir_tab[(i)-1 + 3 * ((j)-1)]
#define wave(i, j) wave[(i)-1 + n_max_compute_ham * ((j)-1)]
#define batch_center_all_batches(i, j) batch_center_all_batches[(i)-1 + max_n_batch_centers * ((j)-1)]
#define batch_point_to_i_full_point(i, j) batch_point_to_i_full_point[(i)-1 + n_max_batch_size * ((j)-1)]
    int l_ylm_max = l_ylm_max_;

    // int atom_index[n_centers_integrals];
    // int atom_index_inv[n_centers];                      // 没用？
    // int i_basis_fns[n_basis_fns * n_centers_integrals]; // 没用?
    // int i_basis_fns_inv[n_basis_fns * n_centers];
    // int i_atom_fns[n_basis_fns * n_centers_integrals]; // 没用？
    // int spline_array_start[n_centers_integrals];
    // int spline_array_end[n_centers_integrals];
    // double one_over_dist_tab[n_max_compute_atoms];
    // int rad_index[n_max_compute_atoms];
    // int wave_index[n_max_compute_fns_ham];
    // int l_index[n_max_compute_fns_ham];
    // int l_count[n_max_compute_fns_ham];
    // int fn_atom[n_max_compute_fns_ham];
    // int zero_index_point[n_max_compute_ham];
    // double wave[n_max_compute_ham * n_max_batch_size];
    // double first_order_density_matrix_con[n_max_compute_dens * n_max_compute_dens];
    double *dist_tab_sq = dist_tab_sq__ + gid * n_max_compute_atoms;
    double *dist_tab = dist_tab__ + gid * n_max_compute_atoms;
    // global double *dir_tab = dir_tab__ + gid * 3 * n_centers_integrals;
    double *dir_tab = dir_tab__ + bid * lsize * 3 * n_max_compute_atoms;
    int *atom_index = atom_index__ + gid * n_max_compute_atoms; // use private instead
    // global int *atom_index_inv = atom_index_inv__ + gid * n_centers;
    // global int *atom_index_inv = atom_index_inv__ + get_group_id(0) * get_local_size(0) * n_centers;
    // global int *i_basis_fns = i_basis_fns__ + gid * n_basis_fns * n_centers_integrals;   // NULL removed
    // global int *i_basis_fns_inv = i_basis_fns_inv__ + gid * n_basis_fns * n_centers;
    // global int *i_atom_fns = i_atom_fns__ + gid * n_basis_fns * n_centers_integrals;     // NULL removed
    int *spline_array_start = spline_array_start__ + gid * n_max_compute_atoms; // use private instead
    int *spline_array_end = spline_array_end__ + gid * n_max_compute_atoms;     // use private instead
    // private int atom_index[MACRO_n_centers_integrals];
    // private int spline_array_start[MACRO_n_centers_integrals];
    // private int spline_array_end[MACRO_n_centers_integrals];

    // double one_over_dist_tab[MACRO_n_max_compute_atoms];
    // int rad_index[MACRO_n_max_compute_atoms];

    int *rad_index = rad_index__ + gid * n_max_compute_atoms;
    double *one_over_dist_tab = one_over_dist_tab__ + gid * n_max_compute_atoms;

    int *wave_index = wave_index__ + gid * n_max_compute_fns_ham;
    // global int *l_index = l_index__ + gid * n_max_compute_fns_ham;  // val[i] = l_aux * l_aux + 1, store in l_count[i]=val  // NULL removed
    int *l_count = l_count__ + gid * n_max_compute_fns_ham; // val[i] = 2 * l_aux, store in l_count[i]=val
    int *fn_atom = fn_atom__ + gid * n_max_compute_fns_ham;
    int *zero_index_point = zero_index_point__ + gid * n_max_compute_ham;
    // global double *wave = wave__ + gid * n_max_compute_ham;
    double *wave_group = wave__ + bid * ((n_max_batch_size + 127) / 128 * 128) * ((n_max_compute_ham + 127) / 128 * 128) + 128;
    double *i_r = i_r__ + gid * n_max_compute_atoms;
    // global double *trigonom_tab = trigonom_tab__ + gid * 4 * n_max_compute_atoms;
    double *radial_wave = radial_wave__ + gid * n_max_compute_fns_ham;

    double *spline_array_aux = spline_array_aux__ + gid * n_basis_fns;
    double *aux_radial = aux_radial__ + gid * n_max_compute_atoms * n_basis_fns; // 有风险, n_max_compute_atoms 是猜的

    double *ylm_tab = ylm_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);
    double *dylm_dtheta_tab = dylm_dtheta_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);
    // 暂时这两个没用的用一样的空间
    double *scaled_dylm_dphi_tab = dylm_dtheta_tab__ + gid * ((l_ylm_max + 1) * (l_ylm_max + 1) * n_max_compute_atoms);

    // #define i_basis_fns_inv(i, j) i_basis_fns_inv[(i)-1 + n_basis_fns * ((j)-1)]

    // int i_my_batch = *i_my_batch_;
    // for (int i_my_batch = 1; i_my_batch <= n_my_batches_work_rho; i_my_batch++) {
    // int pos_id = 0;
    // for (int i_cnt = 0; i_cnt < bid; i_cnt++)
    // {
    //     pos_id += count_batches[i_cnt];
    // }
    // __syncthreads();
    for (int i_my_batch = bid + 1; i_my_batch <= n_my_batches_work_rho; i_my_batch += (gsize / lsize))
    {
        // for (int i_my_cnt = 0; i_my_cnt < count_batches[bid]; i_my_cnt++)
        // {
        // int i_my_batch = block[pos_id + i_my_cnt] + 1;
        // for (int i_my_batch = bid + 1; i_my_batch <= n_my_batches_work_rho; i_my_batch += (gsize / lsize))
        // {
        // for (int i_my_batch = get_group_id(0)+1; i_my_batch <= 2; i_my_batch+=(get_global_size(0) / get_local_size(0))) {
        // int i_my_batch_max = (*i_my_batch_ + 127) < n_my_batches_work_rho ? (*i_my_batch_ + 127) : n_my_batches_work_rho;
        // for (int i_my_batch = *i_my_batch_; i_my_batch <= i_my_batch_max; i_my_batch++) {
        int n_compute_c = batches_batch_n_compute_rho(i_my_batch);
        double *first_order_density_matrix_con = first_order_density_matrix_con__ + bid * n_max_compute_dens * n_max_compute_dens;
        // prune_density_matrix_sparse_dielectric_c_(first_order_density_matrix_sparse, first_order_density_matrix_con,
        //           &n_compute_c, &batches_batch_i_basis_rho(1, i_my_batch), index_hamiltonian_dim2, position_in_hamiltonian_dim1,
        //           &center_to_cell(1), &Cbasis_to_basis(1), &Cbasis_to_center(1), &index_hamiltonian(1, 1, 1),
        //           &position_in_hamiltonian(1, 1), &column_index_hamiltonian(1));
        prune_density_matrix_sparse_polar_reduce_memory_local_index(first_order_density_matrix_sparse, first_order_density_matrix_con,
                                                                    &n_compute_c, &ins_idx_all_batches[(i_my_batch - 1) * (n_basis_local_)], n_local_matrix_size_);

        if (n_compute_c > 0)
        {
            for (int i_point_div = 0; i_point_div < ((n_points_all_batches[i_my_batch - 1] + lsize - 1) / lsize); i_point_div++)
            {
                int i_point = i_point_div * lsize + lid + 1;
                if (i_point <= n_points_all_batches[i_my_batch - 1])
                {

                    for (int i = 0; i < n_basis_fns * (n_max_compute_atoms + 1); i++)
                        i_basis_fns_inv__[i * gsize + gid] = 0.0;

                    int n_compute_atoms = 0;
                    int n_compute_fns = 0;
                    int n_zero_compute;
                    prune_radial_basis_p2_c_rho(&n_max_compute_atoms, &n_max_compute_fns_ham, &dist_tab_sq(1), &dist_tab(1),
                                                //  &dir_tab(1, 1), // (3, n_atom_list)
                                                dir_tab, // (3, n_atom_list)
                                                &n_compute_atoms, atom_index, NULL, &n_compute_fns, NULL,
                                                i_basis_fns_inv__, // (n_basis_fns,n_centers)
                                                NULL, spline_array_start, spline_array_end, &n_centers_integrals,
                                                centers_basis_integrals, &n_compute_c, &batches_batch_i_basis_rho(1, i_my_batch),
                                                &n_batch_centers_all_batches[i_my_batch - 1], &batch_center_all_batches(1, i_my_batch),
                                                one_over_dist_tab, rad_index, wave_index, NULL, l_count, fn_atom, &n_zero_compute,
                                                zero_index_point
                                                // outer
                                                ,
                                                n_basis_fns, &center_to_atom(1), &species_center(1), &Cbasis_to_basis(1),
                                                &Cbasis_to_center(1), &perm_basis_fns_spl(1), &outer_radius_sq(1), &basis_fn(1),
                                                &basis_l(1), &atom_radius_sq(1), &basis_fn_start_spl(1), &basis_fn_atom(1, 1),
                                                pbc_lists_coords_center,
                                                batches_points_coords_rho(1, i_point, i_my_batch),
                                                batches_points_coords_rho(2, i_point, i_my_batch),
                                                batches_points_coords_rho(3, i_point, i_my_batch));

                    tab_local_geometry_p2_c_(&n_compute_atoms, atom_index, &dist_tab(1),
                                             i_r
                                             // outer
                                             ,
                                             &species_center(1), &r_grid_min(1), &log_r_grid_inc(1));

                    tab_gradient_ylm_p0_c_(NULL, basis_l_max, &l_ylm_max, &n_compute_atoms, atom_index, ylm_tab,
                                           dylm_dtheta_tab, scaled_dylm_dphi_tab, dir_tab, &species_center(1));

                    int mfalse = 0;
                    evaluate_radial_functions_p0_c_(
                        spline_array_start, spline_array_end, &n_compute_atoms, &n_compute_fns, &dist_tab(1), i_r, atom_index,
                        i_basis_fns_inv__, basis_wave_ordered, radial_wave, &mfalse, &n_compute_c,
                        &n_max_compute_fns_ham
                        // outer
                        ,
                        n_basis_fns, n_max_grid, &species_center(1), &n_grid(1), &perm_basis_fns_spl(1),
                        spline_array_aux);
                    evaluate_waves_p2_c_(&n_compute_c, &n_compute_atoms, &n_compute_fns, &l_ylm_max, ylm_tab, one_over_dist_tab,
                                         radial_wave, wave_group + n_max_compute_ham * (i_point - 1), rad_index, wave_index, NULL, l_count, fn_atom,
                                         &n_zero_compute, zero_index_point, aux_radial);
                } // if(i_point <= n_points_all_batches[i_my_batch - 1])
                int point_valid = (i_point <= n_points_all_batches[i_my_batch - 1]);
                int tmp_point = i_point - 1;
                double i_point_rho = 0.0;
                __syncthreads();

                int x_point_off_max = min(lsize, n_points_all_batches[i_my_batch - 1] - i_point_div * lsize);
                for (int i_compute_off = 0; i_compute_off < n_compute_c; i_compute_off += (IC_TILE * IC_WORK))
                { // 1,2 是对的，怀疑因为波阵列是 32
                    int i_compute_max = min(n_compute_c - i_compute_off, (IC_TILE * IC_WORK));
                    for (int x_point_off = 0; x_point_off < x_point_off_max; x_point_off += (PT_TILE * PT_WORK))
                    {
                        int x_point_max = min(x_point_off_max - x_point_off, (PT_TILE * PT_WORK));
                        double private_rho[IC_WORK][PT_WORK];
                        for (int i = 0; i < IC_WORK; i++)
                            for (int j = 0; j < PT_WORK; j++)
                                private_rho[i][j] = 0.0;
                        for (int x_point_work = 0; x_point_work < PT_WORK; x_point_work++)
                        {
                            local_tmp_rho[lid / JC_TILE][lid % JC_TILE + PT_TILE * x_point_work] = 0.0;
                        }
                        for (int j_compute_off = 0; j_compute_off < (i_compute_off + i_compute_max); j_compute_off += JC_TILE)
                        {
                            int j_compute_max = min((i_compute_off + i_compute_max) - j_compute_off, JC_TILE);
                            // int j_compute_max = min((n_compute_c+i_compute_max)-j_compute_off, JC_TILE);
                            int id = lid;
                            int i_compute = id / JC_TILE;
                            int x_point = id % JC_TILE;
                            {
                                int i_compute = id / JC_TILE;
                                int x_point = id / JC_TILE;
                                int j_compute = id % JC_TILE;
                                for (int x_point_work = 0; x_point_work < PT_WORK; x_point_work++)
                                {
                                    // 有一点点越界风险
                                    local_wave[j_compute][x_point + PT_TILE * x_point_work] = j_compute < j_compute_max && (x_point + PT_TILE * x_point_work) < x_point_max
                                                                                                  ? wave_group[(x_point_off + x_point + PT_TILE * x_point_work) * n_max_compute_ham + (j_compute_off + j_compute)]
                                                                                                  : 0.0;
                                }
                                for (int i_compute_work = 0; i_compute_work < IC_WORK; i_compute_work++)
                                {
                                    if (i_compute + IC_TILE * i_compute_work < i_compute_max)
                                    {
                                        // double tmp;
                                        // if((j_compute_off + j_compute) < (i_compute_off + i_compute + IC_TILE*i_compute_work))
                                        //   tmp = first_order_density_matrix_con[(i_compute_off + i_compute + IC_TILE*i_compute_work) * n_compute_c + (j_compute_off + j_compute)];
                                        // else
                                        //   tmp = 0.0;
                                        // local_density[i_compute + IC_TILE*i_compute_work][j_compute] = tmp;
                                        local_density[i_compute + IC_TILE * i_compute_work][j_compute] =
                                            first_order_density_matrix_con[(i_compute_off + i_compute + IC_TILE * i_compute_work) * n_compute_c + (j_compute_off + j_compute)];
                                    }
                                }
                            }
                            __syncthreads();
                            // barrier(CLK_LOCAL_MEM_FENCE);
                            for (int i_compute_work = 0; i_compute_work < IC_WORK; i_compute_work++)
                            {
                                if ((i_compute + IC_TILE * i_compute_work) < i_compute_max && x_point < x_point_max)
                                {
                                    for (int x_point_work = 0; x_point_work < PT_WORK; x_point_work++)
                                    {
                                        if ((x_point + PT_TILE * x_point_work) < x_point_max)
                                        {
                                            for (int j_compute = 0; j_compute < JC_TILE; j_compute++)
                                            {
                                                private_rho[i_compute_work][x_point_work] += local_wave[j_compute][x_point + PT_TILE * x_point_work] * local_density[i_compute + IC_TILE * i_compute_work][j_compute];
                                            }
                                        }
                                    }
                                }
                            }
                            __syncthreads();
                            // barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        int id = lid;
                        int i_compute = id / JC_TILE;
                        int x_point = id % JC_TILE;
                        for (int i_compute_work = 0; i_compute_work < IC_WORK; i_compute_work++)
                        {
                            if ((i_compute + IC_TILE * i_compute_work) < i_compute_max && x_point < x_point_max)
                            {
                                for (int x_point_work = 0; x_point_work < PT_WORK; x_point_work++)
                                {
                                    if ((x_point + PT_TILE * x_point_work) < x_point_max)
                                    {
                                        private_rho[i_compute_work][x_point_work] *= wave_group[(x_point_off + x_point + PT_TILE * x_point_work) * n_max_compute_ham + i_compute_off + i_compute + IC_TILE * i_compute_work];
                                        local_tmp_rho[i_compute][x_point + PT_TILE * x_point_work] += private_rho[i_compute_work][x_point_work];
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        // barrier(CLK_LOCAL_MEM_FENCE);
                        if (x_point_off <= lid && lid < (x_point_off + x_point_max))
                        {
                            // i_point_rho += private_rho;
                            int x_point = lid % (PT_TILE * PT_WORK);
                            int maxi = min(IC_TILE, i_compute_max);
                            for (int i = 0; i < maxi; i++)
                            {
                                i_point_rho += local_tmp_rho[i][x_point];
                            }
                        }
                        __syncthreads();
                        // barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
#undef IC_TILE
#undef JC_TILE
#undef PT_TILE
                if (point_valid)
                {
                    first_order_rho[batch_point_to_i_full_point(i_point, i_my_batch) - 1] = i_point_rho;
                }
#undef WAVEJ_TILE_SIZE
#undef WAVEI_TILE_SIZE
            }
        }
    }
#undef i_basis_fns_inv

    // m_save_check_rho_(first_order_rho);
#undef dist_tab_sq
#undef dist_tab
#undef dir_tab
#undef wave
#undef batch_center_all_batches
#undef batch_point_to_i_full_point
}
