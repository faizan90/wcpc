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

cdef extern from "stdio.h" nogil:
    cdef:
        int printf(const char *x)

cdef extern from "math.h" nogil:
    cdef:
        int abs(int x)

cdef extern from "./rand_gen.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this at least once
        void re_seed(DT_ULL x)  # calls warm_up as well

warm_up()


cdef int ret_cont_flag(
    DT_UL_NP_t[:, :] cp_rules,
    const np.uint8_t[:, :] anom_crnr_flags_arr,
    const DT_UL n_anom_cols,
    const DT_UL n_fuzz_nos,
    const int thresh_steep,
    const DT_UL cp_no,
    const DT_UL pt_no,
    const DT_UL fuzz_no,
    ) nogil:

    cdef:
        int curr_rule

    if anom_crnr_flags_arr[pt_no, 0]:
        curr_rule = <int> cp_rules[cp_no, pt_no - n_anom_cols - 1]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 1]:
        curr_rule = <int> cp_rules[cp_no, pt_no - n_anom_cols]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 2]:
        curr_rule = <int> cp_rules[cp_no, pt_no - n_anom_cols + 1]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 3]:
        curr_rule = <int> cp_rules[cp_no, pt_no - 1]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 4]:
        curr_rule = <int> cp_rules[cp_no, pt_no + 1]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 5]:
        curr_rule = <int> cp_rules[cp_no, pt_no + n_anom_cols - 1]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 6]:
        curr_rule = <int> cp_rules[cp_no, pt_no + n_anom_cols]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    if anom_crnr_flags_arr[pt_no, 7]:
        curr_rule = <int> cp_rules[cp_no, pt_no + n_anom_cols + 1]
        if (curr_rule != n_fuzz_nos) and (abs(<int> (fuzz_no - curr_rule)) > thresh_steep):
            return 1

    return 0

cdef void gen_cp_rules(
    DT_UL_NP_t[:, :] cp_rules,
    DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    const np.uint8_t[:, :] anom_crnr_flags_arr,
    const int no_steep_anom_flag,
    const DT_UL max_idxs_ct,
    const DT_UL n_cps,
    const DT_UL n_pts,
    const DT_UL n_fuzz_nos,
    const DT_UL n_cpus,
    const DT_UL n_anom_cols,
    const int thresh_steep,
    int *gen_mod_cp_err_flag,
    ) nogil:

    '''
    Generate CP rules, given the maximum number of a rule that a
    CP can have (max_idxs_ct).

    The number of indicies assigned inside each CP for a given rule
    are between zero and max_idxs_ct.
    '''

    cdef:
        Py_ssize_t j, k, l
        int curr_l
        DT_UL curr_idxs_ctr, curr_iter_ctr
        DT_UL rand_i, rand_v
        DT_UL max_iters = 1000 * n_pts * max_idxs_ct

    for j in prange(n_cps, schedule='static', nogil=True, num_threads=n_cpus):
        for k in range(n_pts):
            cp_rules[j, k] = n_fuzz_nos

        for l in range(n_fuzz_nos):
            curr_iter_ctr = 0
            curr_idxs_ctr = 0
            while (curr_idxs_ctr < max_idxs_ct):
                curr_iter_ctr = curr_iter_ctr + 1

                if curr_iter_ctr > max_iters:
                    with gil: print("\n########Too many iterations in gen_cp_rules (CP: %d, Fuzz No.: %d, curr_idxs_ctr: %d)!########\n" % (j, l, curr_idxs_ctr))
                    gen_mod_cp_err_flag[0] = 1
                    break

                rand_i = <DT_UL> (rand_c() * n_pts)
                if cp_rules[j, rand_i] != n_fuzz_nos:
                    continue

                if (no_steep_anom_flag and 
                    ret_cont_flag(cp_rules, 
                        anom_crnr_flags_arr,
                        n_anom_cols,
                        n_fuzz_nos,
                        thresh_steep,
                        <DT_UL> j,
                        rand_i,
                        <DT_UL> l)):

                    continue

                cp_rules[j, rand_i] = l
                curr_idxs_ctr = curr_idxs_ctr + 1

            cp_rules_idx_ctr[j, l] = curr_idxs_ctr

    return

cdef void mod_cp_rules(
    DT_UL_NP_t[:, :] cp_rules,
    DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    DT_UL_NP_t[:, :] loc_mod_ctr,
    const np.uint8_t[:, :] anom_crnr_flags_arr,
    const int no_steep_anom_flag,
    const DT_UL max_idxs_ct,
    const DT_UL n_cps,
    const DT_UL n_pts,
    const DT_UL n_fuzz_nos,
    const DT_UL run_type,
    DT_UL *rand_k,
    DT_UL *rand_i,
    DT_UL *rand_v,
    DT_UL *old_v_i_k,
    const DT_UL n_anom_cols,
    const int thresh_steep,
    int *gen_mod_cp_err_flag,
    ) nogil:

    cdef:
        int curr_l
        DT_UL dont_stop = 1
        DT_UL max_iters = 1000 * max_idxs_ct * n_pts, curr_iter_ct = 0
        DT_UL rand_k_, rand_i_, rand_v_, old_v_i_k_

    if run_type == 3:
        if old_v_i_k[0] < n_fuzz_nos:
            cp_rules_idx_ctr[rand_k[0], old_v_i_k[0]] += 1

        if rand_v[0] < n_fuzz_nos:
            cp_rules_idx_ctr[rand_k[0], rand_v[0]] -= 1 

        loc_mod_ctr[rand_k[0], rand_i[0]] -= 1
        cp_rules[rand_k[0], rand_i[0]] = old_v_i_k[0]

        dont_stop = 0  # just in case

    else:
        while (dont_stop):
            curr_iter_ct += 1
            if curr_iter_ct > max_iters:
                with gil: print("\n########Too many iterations in mod_cp_rules!########\n")
                gen_mod_cp_err_flag[0] = 1
                break

            rand_k_ = <DT_UL> (rand_c() * n_cps)  # random CP out of n_cps
            rand_i_ = <DT_UL> (rand_c() * n_pts)  # random point in n_pts

            # random fuzzy rule index out of n_fuzz_nos + 1
            # the extra index is for the points that are supposed to have no 
            # effect on the objective function
            rand_v_ = <DT_UL> (rand_c() * (n_fuzz_nos + 1))
            while ((cp_rules[rand_k_, rand_i_] == rand_v_)):
                rand_v_ = <DT_UL> (rand_c() * (n_fuzz_nos + 1))

            old_v_i_k_ = cp_rules[rand_k_, rand_i_]

            if rand_v_ < n_fuzz_nos:
                if cp_rules_idx_ctr[rand_k_, rand_v_] >= max_idxs_ct:
                    continue

                if (no_steep_anom_flag and 
                    ret_cont_flag(cp_rules, 
                        anom_crnr_flags_arr,
                        n_anom_cols,
                        n_fuzz_nos,
                        thresh_steep,
                        rand_k_,
                        rand_i_,
                        rand_v_)):

                    continue

                cp_rules[rand_k_, rand_i_] = rand_v_
                cp_rules_idx_ctr[rand_k_, rand_v_] += 1

                if old_v_i_k_ < n_fuzz_nos:
                    cp_rules_idx_ctr[rand_k_, old_v_i_k_] -= 1

                dont_stop = 0

            else:
                cp_rules[rand_k_, rand_i_] = rand_v_
                cp_rules_idx_ctr[rand_k_, old_v_i_k_] -= 1   
                dont_stop = 0

        loc_mod_ctr[rand_k_, rand_i_] += 1
        rand_k[0] = rand_k_
        rand_i[0] = rand_i_
        rand_v[0] = rand_v_
        old_v_i_k[0] = old_v_i_k_

    return

