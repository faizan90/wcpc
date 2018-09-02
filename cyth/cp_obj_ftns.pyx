# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

### obj_ftns:False;False;True;False;False;False;False;False

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
    const DT_D_NP_t[:, :] in_ppt_arr,
    const DT_UL n_stns,
    DT_D_NP_t[:, :] ppt_cp_mean_arr,
    const DT_D_NP_t[:] ppt_mean_arr,
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

        Py_ssize_t m
        DT_D curr_ppt

        DT_D o_3 = 0.0
        DT_D cp_ppt_mean
        DT_D curr_ppt_diff

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

    for s in prange(n_max, schedule='static', nogil=True, num_threads=num_threads):
        curr_ppt_diff = 0.0
        if s < n_stns:
            m = s
            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                cp_ppt_mean = 0.0

                for i in range(n_time_steps):
                    if sel_cps[i] != j:
                        continue

                    curr_ppt = in_ppt_arr[i, m]

                    cp_ppt_mean = cp_ppt_mean + curr_ppt

                ppt_cp_mean_arr[j, m] = cp_ppt_mean
                cp_ppt_mean = cp_ppt_mean / ppt_cp_n_vals_arr[j]

                curr_ppt_diff = curr_ppt_diff + (ppt_cp_n_vals_arr[j] * abs((cp_ppt_mean / ppt_mean_arr[m]) - 1))

        o_3 += (curr_ppt_diff / n_time_steps)

    obj_val += (o_3 * obj_ftn_wts_arr[2])

    obj_val_copy = obj_val
    for j in range(n_cps):
        _ = (ppt_cp_n_vals_arr[j] / n_time_steps)
        if _ < min_freq:
            obj_val -= ((lo_freq_pen_wt * (min_freq - _) * obj_val_copy))

    return obj_val

cdef DT_D obj_ftn_update(
    const DT_D_NP_t[:, :] in_ppt_arr,
    const DT_UL n_stns,
    DT_D_NP_t[:, :] ppt_cp_mean_arr,
    const DT_D_NP_t[:] ppt_mean_arr,
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
        Py_ssize_t m
        DT_D curr_ppt

        DT_D o_3 = 0.0
        DT_D curr_ppt_diff
        DT_D cp_ppt_mean
        DT_D old_ppt_cp_mean
        DT_D sel_ppt_cp_mean

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

    for s in prange(n_max, schedule='static', nogil=True, num_threads=num_threads):
        curr_ppt_diff = 0.0
        if s < n_stns:
            m = s
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

                cp_ppt_mean = ppt_cp_mean_arr[j, m] / ppt_cp_n_vals_arr[j]
                _ =  cp_ppt_mean / ppt_mean_arr[m]
                curr_ppt_diff = curr_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(_ - 1))

        o_3 += (curr_ppt_diff / n_time_steps)

    obj_val += (o_3 * obj_ftn_wts_arr[2])

    obj_val_copy = obj_val
    for j in range(n_cps):
        _ = (ppt_cp_n_vals_arr[j] / n_time_steps)
        if _ < min_freq:
            obj_val -= ((lo_freq_pen_wt * (min_freq - _) * obj_val_copy))

    return obj_val
