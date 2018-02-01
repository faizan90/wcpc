# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

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
    const DT_D_NP_t[:, :] in_cats_ppt_arr,
    const DT_UL n_cats,
    DT_D_NP_t[:, :] cats_ppt_cp_mean_arr,
    const DT_D_NP_t[:] cats_ppt_mean_arr,
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

        Py_ssize_t q
        DT_D curr_cat_ppt

        DT_D o_5 = 0.0
        DT_D cp_cat_ppt_mean
        DT_D curr_cat_ppt_diff

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

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=num_threads):
        curr_cat_ppt_diff = 0
        if s < n_cats:
            q = s
            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                cp_cat_ppt_mean = 0

                for i in range(n_time_steps):
                    if sel_cps[i] != j:
                        continue

                    curr_cat_ppt = in_cats_ppt_arr[i, q]

                    cp_cat_ppt_mean = cp_cat_ppt_mean + curr_cat_ppt

                cats_ppt_cp_mean_arr[j, q] = cp_cat_ppt_mean
                cp_cat_ppt_mean = cp_cat_ppt_mean / ppt_cp_n_vals_arr[j]

                curr_cat_ppt_diff = curr_cat_ppt_diff + (ppt_cp_n_vals_arr[j] * abs((cp_cat_ppt_mean / cats_ppt_mean_arr[q]) - 1))

        o_5 += (curr_cat_ppt_diff / n_time_steps)

    obj_val += (o_5 * obj_ftn_wts_arr[4])

    obj_val_copy = obj_val
    for j in range(n_cps):
        _ = (ppt_cp_n_vals_arr[j] / n_time_steps)
        if _ < min_freq:
            obj_val -= ((0.001 * rand_c()) + (lo_freq_pen_wt * (min_freq - _) * obj_val_copy))

    return obj_val

cdef DT_D obj_ftn_update(
    const DT_D_NP_t[:, :] in_cats_ppt_arr,
    const DT_UL n_cats,
    DT_D_NP_t[:, :] cats_ppt_cp_mean_arr,
    const DT_D_NP_t[:] cats_ppt_mean_arr,
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
        Py_ssize_t q
        DT_D curr_cat_ppt

        DT_D o_5 = 0.0
        DT_D curr_cat_ppt_diff
        DT_D cp_cat_ppt_mean
        DT_D old_cat_ppt_cp_mean
        DT_D sel_cat_ppt_cp_mean

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

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=num_threads):
        curr_cat_ppt_diff = 0
        if s < n_cats:
            q = s
            # remove the effect of the previous CP
            for j in range(n_cps):
                old_cat_ppt_cp_mean = 0.0
                sel_cat_ppt_cp_mean = 0.0

                for i in range(n_time_steps):
                    if not chnge_steps[i]:
                        continue

                    curr_cat_ppt = in_cats_ppt_arr[i, q]

                    if old_sel_cps[i] == j:
                        old_cat_ppt_cp_mean = old_cat_ppt_cp_mean + curr_cat_ppt

                    if sel_cps[i] == j:
                        sel_cat_ppt_cp_mean = sel_cat_ppt_cp_mean + curr_cat_ppt

                cats_ppt_cp_mean_arr[j, q] = cats_ppt_cp_mean_arr[j, q] - old_cat_ppt_cp_mean + sel_cat_ppt_cp_mean

            # incorporate the effect of the new CP
            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                cp_cat_ppt_mean = cats_ppt_cp_mean_arr[j, q] / ppt_cp_n_vals_arr[j]
                _ = cp_cat_ppt_mean / cats_ppt_mean_arr[q]
                curr_cat_ppt_diff = curr_cat_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(_ - 1))

        o_5 += (curr_cat_ppt_diff / n_time_steps)

    obj_val += (o_5 * obj_ftn_wts_arr[4])

    obj_val_copy = obj_val
    for j in range(n_cps):
        _ = (ppt_cp_n_vals_arr[j] / n_time_steps)
        if _ < min_freq:
            obj_val -= ((0.001 * rand_c()) + (lo_freq_pen_wt * (min_freq - _) * obj_val_copy))

    return obj_val
