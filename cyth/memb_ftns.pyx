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

cdef extern from "./fuzzy_ftns.h" nogil:
    cdef:
        DT_D get_tri_mu(const DT_D *g,
        const DT_D *a,
        const DT_D *b,
        const DT_D *c)

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
    ) nogil:
    '''
    Calculate memberships, DOFs, selected CPs and time steps with changed
    CPs.
    '''

    cdef:
        Py_ssize_t i, j, k, l
        DT_UL curr_fuzz_idx, curr_idxs_sum, best_dof_cp
        DT_D curr_dof, curr_mus_sum, max_dof

    # Fill the membership value matrix at each time step,
    # each CP and each point.
    # Select the CP with the greatest DOF for a given step.
    # Set everything to the previous step in case of a roll back.
    for i in prange(n_time_steps, schedule='dynamic', nogil=True, num_threads=n_cpus):
        for j in range(n_cps):
            for k in range(n_pts):
                curr_fuzz_idx = cp_rules[j, k]
                if curr_fuzz_idx == n_fuzz_nos:
                    mu_i_k_arr[i, j, k] = 1.0

                else:
                    mu_i_k_arr[i, j, k] = \
                        get_tri_mu(&slp_anom[i, k],
                            &fuzz_nos_arr[curr_fuzz_idx, 0],
                            &fuzz_nos_arr[curr_fuzz_idx, 1],
                            &fuzz_nos_arr[curr_fuzz_idx, 2])

        for j in range(n_cps):
            curr_dof = 1.0
            for l in range(n_fuzz_nos):
                curr_idxs_sum = 0
                curr_mus_sum = 0
                for k in range(n_pts):
                    if cp_rules[j, k] != l:
                        continue

                    curr_idxs_sum = curr_idxs_sum + 1
                    curr_mus_sum = curr_mus_sum + (mu_i_k_arr[i, j, k]**p_l)

                if curr_idxs_sum:
                    curr_mus_sum = (curr_mus_sum / curr_idxs_sum)**(1.0 / p_l)

                cp_dof_arr[i, j, l] = curr_mus_sum
                curr_dof = curr_mus_sum * curr_dof

            dofs_arr[i, j] = curr_dof

        best_dof_cp = no_cp_val
        max_dof = 1e-5
        for j in range(n_cps):
            if dofs_arr[i, j] > max_dof:
                max_dof = dofs_arr[i, j]
                best_dof_cp = j

        old_sel_cps[i] = no_cp_val
        sel_cps[i] = best_dof_cp
        if sel_cps[i] != old_sel_cps[i]:
            chnge_steps[i] = 1

        else:
            chnge_steps[i] = 0

    return

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
    ) nogil:
    '''
    Update memberships and DOFs with changed CPs.
    '''

    cdef:
        Py_ssize_t i, j, l
        int pre_cond, curr_cond
        DT_UL pre_idxs_sum, best_dof_cp, curr_idxs_sum
        DT_D curr_dof, curr_mus_sum, max_dof, pre_mus_sum
        DT_D f1, f2, f3

    if curr_fuzz_idx != n_fuzz_nos:
        f1 = fuzz_nos_arr[curr_fuzz_idx, 0]
        f2 = fuzz_nos_arr[curr_fuzz_idx, 1]
        f3 = fuzz_nos_arr[curr_fuzz_idx, 2]
        curr_idxs_sum = cp_rules_idx_ctr[curr_cp, curr_fuzz_idx]
        curr_cond = 1

    else:
        curr_cond = 0

    if pre_fuzz_idx != n_fuzz_nos:
        pre_cond = 1
        pre_idxs_sum = cp_rules_idx_ctr[curr_cp, pre_fuzz_idx]

    else:
        pre_cond = 0

    for i in prange(n_time_steps, schedule='static', nogil=True, num_threads=n_cpus):
        # remove old
        if pre_cond:
            if pre_idxs_sum > 0:
                pre_mus_sum = (cp_dof_arr[i, curr_cp, pre_fuzz_idx]**p_l) * (pre_idxs_sum + 1)
                pre_mus_sum = pre_mus_sum - (mu_i_k_arr[i, curr_cp, curr_pt]**p_l)

                if (pre_mus_sum < 0):  # numerical errors around -1e-18
                    pre_mus_sum = 0.0

                cp_dof_arr[i, curr_cp, pre_fuzz_idx] = (pre_mus_sum / pre_idxs_sum)**(1.0 / p_l)

            else:
                cp_dof_arr[i, curr_cp, pre_fuzz_idx] = 0.0

        # update the mu value at the point
        if curr_fuzz_idx == n_fuzz_nos:
            mu_i_k_arr[i, curr_cp, curr_pt] = 1.0

        else:
            mu_i_k_arr[i, curr_cp, curr_pt] = \
                get_tri_mu(&slp_anom[i, curr_pt], &f1, &f2, &f3)

        # add new
        if curr_cond:
            curr_mus_sum = (cp_dof_arr[i, curr_cp, curr_fuzz_idx]**p_l) * (curr_idxs_sum - 1)
            curr_mus_sum = curr_mus_sum + (mu_i_k_arr[i, curr_cp, curr_pt]**p_l)
            cp_dof_arr[i, curr_cp, curr_fuzz_idx] = (curr_mus_sum / curr_idxs_sum)**(1.0 / p_l)

        # update new dofs for pre and curr cps
        curr_dof = 1.0
        for l in range(n_fuzz_nos):
            curr_mus_sum = cp_dof_arr[i, curr_cp, l]
            curr_dof = curr_mus_sum * curr_dof

        dofs_arr[i, curr_cp] = curr_dof
        best_dof_cp = no_cp_val
        max_dof = 1e-5
        for j in range(n_cps):
            if dofs_arr[i, j] > max_dof:
                max_dof = dofs_arr[i, j]
                best_dof_cp = j

        old_sel_cps[i] = sel_cps[i]
        sel_cps[i] = best_dof_cp
        if sel_cps[i] != old_sel_cps[i]:
            chnge_steps[i] = 1

        else:
            chnge_steps[i] = 0

    return

