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


cdef DT_D obj_ftn_refresh(
    const DT_D mean_wet_dof,
    DT_D_NP_t[:] mean_cp_wet_dof_arr,
    const DT_D_NP_t[:] wet_dofs_arr,
    DT_D_NP_t[:] ppt_cp_n_vals_arr,
    const DT_D_NP_t[:] obj_ftn_wts_arr,
    const DT_UL_NP_t[:] sel_cps,
    const DT_UL n_cpus,
    const DT_UL n_cps,
    const DT_UL n_max,
    const DT_UL n_time_steps,
    ) nogil

cdef DT_D obj_ftn_update(
    const DT_D mean_wet_dof,
    DT_D_NP_t[:] mean_cp_wet_dof_arr,
    const DT_D_NP_t[:] wet_dofs_arr,
    DT_D_NP_t[:] ppt_cp_n_vals_arr,
    const DT_D_NP_t[:] obj_ftn_wts_arr,
    const DT_UL_NP_t[:] sel_cps,
    const DT_UL_NP_t[:] old_sel_cps,
    const DT_UL_NP_t[:] chnge_steps,
    const DT_UL n_cpus,
    const DT_UL n_cps,
    const DT_UL n_max,
    const DT_UL n_time_steps,
    ) nogil

