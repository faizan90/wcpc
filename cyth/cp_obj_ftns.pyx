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
    ) nogil:

    # declare/initialize variables
    cdef:
        Py_ssize_t i, j, s
        DT_D _, obj_val = 0.0

        DT_D o_6 = 0.0
        DT_D curr_ppt_wet_dof_diff

    for j in range(n_cps):
        ppt_cp_n_vals_arr[j] = 0
        for i in range(n_time_steps):
            if sel_cps[i] != j:
                continue

            ppt_cp_n_vals_arr[j] += 1

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=n_cpus):
        curr_ppt_wet_dof_diff = 0.0
        if s == 0:
            for j in range(n_cps):
                mean_cp_wet_dof_arr[j] = 0.0

            for j in range(n_cps):
                if ppt_cp_n_vals_arr[j] == 0:
                    continue

                for i in range(n_time_steps):
                    if sel_cps[i] != j:
                        continue

                    mean_cp_wet_dof_arr[j] = mean_cp_wet_dof_arr[j] + wet_dofs_arr[i]

                curr_ppt_wet_dof_diff = curr_ppt_wet_dof_diff + ppt_cp_n_vals_arr[j] * ((mean_cp_wet_dof_arr[j] / ppt_cp_n_vals_arr[j]) - mean_wet_dof)**2

        o_6 += (curr_ppt_wet_dof_diff / n_time_steps)

    obj_val += (o_6 * obj_ftn_wts_arr[5])
    return obj_val

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
    ) nogil:

    cdef:
        Py_ssize_t i, j, s
        DT_D _, obj_val = 0.0

        DT_D o_6 = 0.0
        DT_D curr_ppt_wet_dof_diff

    for j in range(n_cps):
        for i in range(n_time_steps):
            if not chnge_steps[i]:
                continue

            if old_sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] -= 1

            if sel_cps[i] == j:
                ppt_cp_n_vals_arr[j] += 1

    for s in prange(n_max, schedule='dynamic', nogil=True, num_threads=n_cpus):
        curr_ppt_wet_dof_diff = 0.0
        if s == 0:
            for j in range(n_cps):
                for i in range(n_time_steps):
                    if not chnge_steps[i]:
                        continue

                    if old_sel_cps[i] == j:
                        mean_cp_wet_dof_arr[j] = mean_cp_wet_dof_arr[j] - wet_dofs_arr[i]

                    if sel_cps[i] == j:
                        mean_cp_wet_dof_arr[j] = mean_cp_wet_dof_arr[j] + wet_dofs_arr[i]

                if not ppt_cp_n_vals_arr[j]:
                    continue

                curr_ppt_wet_dof_diff = curr_ppt_wet_dof_diff + ppt_cp_n_vals_arr[j] * ((mean_cp_wet_dof_arr[j] / ppt_cp_n_vals_arr[j]) - mean_wet_dof)**2

        o_6 += (curr_ppt_wet_dof_diff / n_time_steps)

    obj_val += (o_6 * obj_ftn_wts_arr[5])
    return obj_val
