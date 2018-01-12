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
    const DT_D min_abs_ppt_thresh,
    DT_D_NP_t[:, :, :] cats_ppt_cp_mean_pis_arr,
    DT_D_NP_t[:, :] cats_ppt_mean_pis_arr,
    DT_D_NP_t[:] o_2_ppt_thresh_arr,
    DT_D_NP_t[:, :] cats_obj_2_vals_arr,
    const DT_UL n_o_2_threshs,
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

        Py_ssize_t q
        DT_D curr_cat_ppt

        Py_ssize_t r
        DT_D o_2 = 0.0
        DT_D curr_cat_ppt_pi_diff

    for j in range(n_cps):
        ppt_cp_n_vals_arr[j] = 0
        for i in range(n_time_steps):
            if sel_cps[i] != j:
                continue

            ppt_cp_n_vals_arr[j] += 1

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=n_cpus):
        if s < n_cats:
            q = s
            for r in range(n_o_2_threshs):
                cats_obj_2_vals_arr[q, r] = 0.0

                for j in range(n_cps):
                    cats_ppt_cp_mean_pis_arr[q, j, r] = 0.0

            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                for i in range(n_time_steps):
                    if sel_cps[i] != j:
                        continue

                    curr_cat_ppt = in_cats_ppt_arr[i, q]

                    for r in range(n_o_2_threshs):
                        if curr_cat_ppt < o_2_ppt_thresh_arr[r]:
                            break

                        cats_ppt_cp_mean_pis_arr[q, j, r] = cats_ppt_cp_mean_pis_arr[q, j, r] + 1

                for r in range(n_o_2_threshs):
                    cats_obj_2_vals_arr[q, r] = cats_obj_2_vals_arr[q, r] + ppt_cp_n_vals_arr[j] * ((cats_ppt_cp_mean_pis_arr[q, j, r] / ppt_cp_n_vals_arr[j]) - cats_ppt_mean_pis_arr[q, r])**2

    for r in range(n_o_2_threshs):
        curr_cat_ppt_pi_diff = 0.0
        for q in range(n_cats):
            curr_cat_ppt_pi_diff += cats_obj_2_vals_arr[q, r]

        o_2 += (curr_cat_ppt_pi_diff / n_time_steps)**0.5

    obj_val += (o_2 * obj_ftn_wts_arr[1])
    return obj_val

cdef DT_D obj_ftn_update(
    const DT_D_NP_t[:, :] in_cats_ppt_arr,
    const DT_UL n_cats,
    const DT_D min_abs_ppt_thresh,
    DT_D_NP_t[:, :, :] cats_ppt_cp_mean_pis_arr,
    DT_D_NP_t[:, :] cats_ppt_mean_pis_arr,
    DT_D_NP_t[:] o_2_ppt_thresh_arr,
    DT_D_NP_t[:, :] cats_obj_2_vals_arr,
    const DT_UL n_o_2_threshs,
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

        Py_ssize_t q
        DT_D curr_cat_ppt

        Py_ssize_t r
        DT_D o_2 = 0.0
        DT_D curr_cat_ppt_pi_diff

    for j in range(n_cps):
        for i in range(n_time_steps):
            if not chnge_steps[i]:
                continue

            if old_sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] -= 1

            if sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] += 1

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=n_cpus):
        if s < n_cats:
            q = s
            for r in range(n_o_2_threshs):
                cats_obj_2_vals_arr[q, r] = 0.0

            # remove the effect of the previous CP
            for j in range(n_cps):
                for i in range(n_time_steps):
                    if not chnge_steps[i]:
                        continue

                    curr_cat_ppt = in_cats_ppt_arr[i, q]

                    if old_sel_cps[i] == j:
                        for r in range(n_o_2_threshs):
                            if curr_cat_ppt < o_2_ppt_thresh_arr[r]:
                                break

                            cats_ppt_cp_mean_pis_arr[q, j, r] = cats_ppt_cp_mean_pis_arr[q, j, r] - 1

                    if sel_cps[i] == j:
                        for r in range(n_o_2_threshs):
                            if curr_cat_ppt < o_2_ppt_thresh_arr[r]:
                                break

                            cats_ppt_cp_mean_pis_arr[q, j, r] = cats_ppt_cp_mean_pis_arr[q, j, r] + 1

            # incorporate the effect of the new CP
            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                for r in range(n_o_2_threshs):
                    cats_obj_2_vals_arr[q, r] = cats_obj_2_vals_arr[q, r] + ppt_cp_n_vals_arr[j] * ((cats_ppt_cp_mean_pis_arr[q, j, r] / ppt_cp_n_vals_arr[j]) - cats_ppt_mean_pis_arr[q, r])**2

    for r in range(n_o_2_threshs):
        curr_cat_ppt_pi_diff = 0.0
        for q in range(n_cats):
            curr_cat_ppt_pi_diff += cats_obj_2_vals_arr[q, r]

        o_2 += (curr_cat_ppt_pi_diff / n_time_steps)**0.5

    obj_val += (o_2 * obj_ftn_wts_arr[1])
    return obj_val
