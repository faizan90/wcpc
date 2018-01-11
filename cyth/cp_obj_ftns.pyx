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
    const DT_D_NP_t[:, :] in_ppt_arr,
    const DT_UL n_stns,
    const DT_D min_abs_ppt_thresh,
    DT_D_NP_t[:, :] ppt_cp_mean_arr,
    const DT_D_NP_t[:] ppt_mean_arr,
    DT_D_NP_t[:] ppt_cp_n_vals_arr,
    const DT_D_NP_t[:] obj_ftn_wts_arr,
    const DT_UL_NP_t[:] sel_cps,
    const DT_UL n_cpus,
    const DT_UL n_cps,
    const DT_UL n_max,
    const DT_UL n_time_steps,
    ) nogil:

    # declare/initialize variables
    cdef:
        Py_ssize_t i, j, s
        DT_D _, obj_val = 0.0

        Py_ssize_t m
        DT_D curr_ppt

        DT_D o_3 = 0.0
        DT_D cp_ppt_mean
        DT_D curr_ppt_diff

    for j in range(n_cps):
        ppt_cp_n_vals_arr[j] = 0
        for i in range(n_time_steps):
            if sel_cps[i] != j:
                continue

            ppt_cp_n_vals_arr[j] += 1

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=n_cpus):
        if s < n_stns:
            m = s
            curr_ppt_diff = 0

            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                cp_ppt_mean = 0

                for i in range(n_time_steps):
                    if sel_cps[i] != j:
                        continue

                    curr_ppt = in_ppt_arr[i, m]

                    cp_ppt_mean = cp_ppt_mean + curr_ppt

                ppt_cp_mean_arr[j, m] = cp_ppt_mean
                cp_ppt_mean = cp_ppt_mean / ppt_cp_n_vals_arr[j]

                if ppt_mean_arr[m] > min_abs_ppt_thresh:
                    _ = cp_ppt_mean / ppt_mean_arr[m]
                    if _ <= 0:
                        _ = 1e-100

                    curr_ppt_diff = curr_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(log(_)))

            o_3 += (curr_ppt_diff / n_time_steps)

    obj_val += (o_3 * obj_ftn_wts_arr[2])
    return obj_val

cdef DT_D obj_ftn_update(
    const DT_D_NP_t[:, :] in_ppt_arr,
    const DT_UL n_stns,
    const DT_D min_abs_ppt_thresh,
    DT_D_NP_t[:, :] ppt_cp_mean_arr,
    const DT_D_NP_t[:] ppt_mean_arr,
    DT_D_NP_t[:] ppt_cp_n_vals_arr,
    const DT_D_NP_t[:] obj_ftn_wts_arr,
    const DT_UL_NP_t[:] sel_cps,
    const DT_UL_NP_t[:] old_sel_cps,
    const DT_UL_NP_t[:] chnge_steps,
    const DT_UL n_cpus,
    const DT_UL n_cps,
    const DT_UL n_max,
    const DT_UL n_time_steps,
    ) nogil:

    cdef:
        Py_ssize_t i, j, s
        DT_D _, obj_val = 0.0

        Py_ssize_t m
        DT_D curr_ppt

        DT_D o_3 = 0.0
        DT_D curr_ppt_diff
        DT_D cp_ppt_mean
        DT_D old_ppt_cp_mean
        DT_D sel_ppt_cp_mean

    for j in range(n_cps):
        for i in range(n_time_steps):
            if not chnge_steps[i]:
                continue

            if old_sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] -= 1

            if sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] += 1

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=n_cpus):
        if s < n_stns:
            m = s
            curr_ppt_diff = 0.0

            # remove the effect of the previous CP
            for j in range(n_cps):
                old_ppt_cp_mean = 0.0
                sel_ppt_cp_mean = 0.0

                for i in range(n_time_steps):
                    if not chnge_steps[i]:
                        continue

                    curr_ppt = in_ppt_arr[i, m]

                    if old_sel_cps[i] == j:
                        old_ppt_cp_mean = old_ppt_cp_mean + curr_ppt

                    if sel_cps[i] == j:
                        sel_ppt_cp_mean = sel_ppt_cp_mean + curr_ppt

                ppt_cp_mean_arr[j, m] = ppt_cp_mean_arr[j, m] - old_ppt_cp_mean + sel_ppt_cp_mean

            # incorporate the effect of the new CP
            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                if ppt_mean_arr[m] > min_abs_ppt_thresh:
                    _ = (ppt_cp_mean_arr[j, m] / ppt_cp_n_vals_arr[j]) / ppt_mean_arr[m]
                    if _ <= 0:
                        _ = 1e-100

                    curr_ppt_diff = curr_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(log(_)))

            o_3 += (curr_ppt_diff / n_time_steps)

    obj_val += (o_3 * obj_ftn_wts_arr[2])
    return obj_val
