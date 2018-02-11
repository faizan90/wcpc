# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef long long DT_LL
ctypedef unsigned long long DT_ULL
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64
DT_UL_NP = np.uint64


cdef void calc_membs_dof_cps(
    const DT_UL_NP_t[:, :] cp_rules,
    DT_D_NP_t[:, :, :] mu_i_k_arr,
    DT_D_NP_t[:, :, :] cp_dof_arr,
    const DT_D_NP_t[:, :] slp_anom,
    const DT_D_NP_t[:, :] fuzz_nos_arr,
    DT_D_NP_t[:, :] dofs_arr,
    DT_UL_NP_t[:] sel_cps,
    DT_UL_NP_t[:] old_sel_cps,
    DT_UL_NP_t[:] chnge_steps,
    const DT_UL no_cp_val,
    const DT_D p_l,
    const DT_UL n_cpus,
    const DT_UL n_time_steps,
    const DT_UL n_pts,
    const DT_UL n_cps,
    const DT_UL n_fuzz_nos,
    ) nogil

cdef void update_membs_dof_cps(
    const DT_UL pre_fuzz_idx,
    const DT_UL curr_fuzz_idx,
    const DT_UL curr_cp,
    const DT_UL curr_pt,
    const DT_UL_NP_t[:, :] cp_rules,
    DT_D_NP_t[:, :, :] mu_i_k_arr,
    DT_D_NP_t[:, :, :] cp_dof_arr,
    const DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    const DT_D_NP_t[:, :] slp_anom,
    const DT_D_NP_t[:, :] fuzz_nos_arr,
    DT_D_NP_t[:, :] dofs_arr,
    DT_UL_NP_t[:] sel_cps,
    DT_UL_NP_t[:] old_sel_cps,
    DT_UL_NP_t[:] chnge_steps,
    const DT_UL no_cp_val,
    const DT_D p_l,
    const DT_UL n_cpus,
    const DT_UL n_time_steps,
    const DT_UL n_cps,
    const DT_UL n_fuzz_nos,
    ) nogil

