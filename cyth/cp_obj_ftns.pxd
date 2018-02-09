# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
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
    const DT_D_NP_t[:, :] in_wet_arr_calib,
    const DT_D_NP_t[:, :] ppt_mean_wet_arr,
    const DT_D_NP_t[:] o_4_p_thresh_arr,
    DT_UL_NP_t[:, :, :] ppt_cp_mean_wet_arr,
    DT_D_NP_t[:, :] nebs_wet_obj_vals_arr,
    const DT_UL n_o_4_threshs,
    const DT_UL n_nebs,
    DT_D_NP_t[:] ppt_cp_n_vals_arr,
    const DT_D_NP_t[:] obj_ftn_wts_arr,
    const DT_UL_NP_t[:] sel_cps,
    const DT_D lo_freq_pen_wt,
    const DT_D min_freq,
    const DT_UL n_cpus,
    const DT_UL n_cps,
    const DT_UL n_max,
    const DT_UL n_time_steps,
    ) nogil

cdef DT_D obj_ftn_update(
    const DT_D_NP_t[:, :] in_wet_arr_calib,
    const DT_D_NP_t[:, :] ppt_mean_wet_arr,
    const DT_D_NP_t[:] o_4_p_thresh_arr,
    DT_UL_NP_t[:, :, :] ppt_cp_mean_wet_arr,
    DT_D_NP_t[:, :] nebs_wet_obj_vals_arr,
    const DT_UL n_o_4_threshs,
    const DT_UL n_nebs,
    DT_D_NP_t[:] ppt_cp_n_vals_arr,
    const DT_D_NP_t[:] obj_ftn_wts_arr,
    const DT_UL_NP_t[:] sel_cps,
    const DT_UL_NP_t[:] old_sel_cps,
    const DT_UL_NP_t[:] chnge_steps,
    const DT_D lo_freq_pen_wt,
    const DT_D min_freq,
    const DT_UL n_cpus,
    const DT_UL n_cps,
    const DT_UL n_max,
    const DT_UL n_time_steps,
    ) nogil

