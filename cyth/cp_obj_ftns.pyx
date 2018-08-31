# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

### obj_ftns:False;False;False;True;False;False;False;False

### op_mp_obj_ftn_flag:True

import numpy as np
cimport numpy as np
from cython.parallel import prange

DT_D_NP = np.float64
DT_UL_NP = np.uint64

cdef extern from "math.h" nogil:
    cdef:
        DT_D exp(DT_D x)
        DT_D log(DT_D x)
        DT_D abs(DT_D x)
        bint isnan(DT_D x)

cdef extern from "./rand_gen.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this at least once
        void re_seed(DT_ULL x)  # calls warm_up as well

warm_up()


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
    ) nogil:

    # declare/initialize variables
    cdef:
        Py_ssize_t i, j, s
        DT_UL num_threads
        DT_D _, obj_val = 0.0, obj_val_copy

        Py_ssize_t n, o
        DT_D o_4 = 0.0
        DT_D curr_ppt_wet_diff

    if n_max < n_cpus:
        num_threads = n_max

    else:
        num_threads = n_cpus

    for j in range(n_cps):
        ppt_cp_n_vals_arr[j] = 0
        for i in range(n_time_steps):
            if sel_cps[i] != j:
                continue

            ppt_cp_n_vals_arr[j] += 1

    for n in range(n_nebs):
        for o in range(n_o_4_threshs):
            nebs_wet_obj_vals_arr[n, o] = 0.0

            for j in range(n_cps):
                ppt_cp_mean_wet_arr[n, j, o] = 0

        for j in range(n_cps):
            if ppt_cp_n_vals_arr[j] == 0:
                continue

            for i in range(n_time_steps):
                if sel_cps[i] != j:
                    continue

                for o in range(n_o_4_threshs):
                    if in_wet_arr_calib[i, n] < o_4_p_thresh_arr[o]:
                        break

                    ppt_cp_mean_wet_arr[n, j, o] = ppt_cp_mean_wet_arr[n, j, o] + 1

            for o in range(n_o_4_threshs):
                nebs_wet_obj_vals_arr[n, o] = nebs_wet_obj_vals_arr[n, o] + ppt_cp_n_vals_arr[j] * ((ppt_cp_mean_wet_arr[n, j, o] / ppt_cp_n_vals_arr[j]) - ppt_mean_wet_arr[n, o])**2

    for o in range(n_o_4_threshs):
        curr_ppt_wet_diff = 0.0
        for n in range(n_nebs):
            curr_ppt_wet_diff += nebs_wet_obj_vals_arr[n, o]

        o_4 += (curr_ppt_wet_diff / n_time_steps)**0.5

    obj_val += (o_4 * obj_ftn_wts_arr[3])

    obj_val_copy = obj_val
    for j in range(n_cps):
        _ = (ppt_cp_n_vals_arr[j] / n_time_steps)
        if _ < min_freq:
            obj_val -= ((lo_freq_pen_wt * (min_freq - _) * obj_val_copy))

    return obj_val

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
    ) nogil:

    cdef:
        Py_ssize_t i, j, s
        DT_UL num_threads
        DT_D _, obj_val = 0.0, obj_val_copy
        Py_ssize_t n, o
        DT_D o_4 = 0.0
        DT_D curr_ppt_wet_diff

    if n_max < n_cpus:
        num_threads = n_max

    else:
        num_threads = n_cpus

    for j in range(n_cps):
        for i in range(n_time_steps):
            if not chnge_steps[i]:
                continue

            if old_sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] -= 1

            if sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] += 1

    for n in range(n_nebs):
        for o in range(n_o_4_threshs):
            nebs_wet_obj_vals_arr[n, o] = 0.0

        for j in range(n_cps):
            for i in range(n_time_steps):
                if not chnge_steps[i]:
                    continue

                if old_sel_cps[i] == j:
                    for o in range(n_o_4_threshs):
                        if in_wet_arr_calib[i, n] < o_4_p_thresh_arr[o]:
                            break

                        ppt_cp_mean_wet_arr[n, j, o] = ppt_cp_mean_wet_arr[n, j, o] - 1

                if sel_cps[i] == j:
                    for o in range(n_o_4_threshs):
                        if in_wet_arr_calib[i, n] < o_4_p_thresh_arr[o]:
                            break

                        ppt_cp_mean_wet_arr[n, j, o] = ppt_cp_mean_wet_arr[n, j, o] + 1

            if not ppt_cp_n_vals_arr[j]:
                continue

            for o in range(n_o_4_threshs):
                nebs_wet_obj_vals_arr[n, o] = nebs_wet_obj_vals_arr[n, o] + ppt_cp_n_vals_arr[j] * ((ppt_cp_mean_wet_arr[n, j, o] / ppt_cp_n_vals_arr[j]) - ppt_mean_wet_arr[n, o])**2

    for o in range(n_o_4_threshs):
        curr_ppt_wet_diff = 0.0
        for n in range(n_nebs):
            curr_ppt_wet_diff += nebs_wet_obj_vals_arr[n, o]

        o_4 += (curr_ppt_wet_diff / n_time_steps)**0.5

    obj_val += (o_4 * obj_ftn_wts_arr[3])

    obj_val_copy = obj_val
    for j in range(n_cps):
        _ = (ppt_cp_n_vals_arr[j] / n_time_steps)
        if _ < min_freq:
            obj_val -= ((lo_freq_pen_wt * (min_freq - _) * obj_val_copy))

    return obj_val
