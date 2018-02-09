# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
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

cdef extern from "./rand_gen.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this at least once
        void re_seed(DT_ULL x)  # calls warm_up as well

warm_up()


cdef void gen_cp_rules(
    DT_UL_NP_t[:, :] cp_rules,
    DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    const DT_UL max_idxs_ct,
    const DT_UL n_cps,
    const DT_UL n_pts,
    const DT_UL n_fuzz_nos,
    const DT_UL n_cpus,
    ) nogil:

    '''
    Generate CP rules, given the maximum number of a rule that a
    CP can have (max_idxs_ct).

    The number of indicies assigned inside each CP for a given rule
    are between zero and max_idxs_ct.
    '''

    cdef:
        Py_ssize_t j, k, l
        DT_UL curr_idxs_ct, curr_idxs_ctr
        DT_UL rand_i, rand_v
        DT_UL max_iters = 1000000, curr_iter_ctr = 0

    for j in prange(n_cps, schedule='static', nogil=True, num_threads=n_cpus):
        for k in range(n_pts):
            cp_rules[j, k] = n_fuzz_nos

        curr_iter_ctr = 0
        for l in range(n_fuzz_nos):
            curr_idxs_ct = <DT_UL> (rand_c() * (max_idxs_ct + 1))
            cp_rules_idx_ctr[j, l] = curr_idxs_ct

            curr_idxs_ctr = 0
            while (curr_idxs_ctr < curr_idxs_ct):
                curr_iter_ctr = curr_iter_ctr + 1
                if curr_iter_ctr > max_iters:
                    printf("\n\n\n\n########Too many iterations in gen_cp_rules!########\n\n\n\n")
                    break

                rand_i = <DT_UL> (rand_c() * n_pts)
                if cp_rules[j, rand_i] != n_fuzz_nos:
                    continue

                cp_rules[j, rand_i] = l
                curr_idxs_ctr = curr_idxs_ctr + 1

    return

cdef void mod_cp_rules(
    DT_UL_NP_t[:, :] cp_rules,
    DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    DT_UL_NP_t[:, :] loc_mod_ctr,
    const DT_UL max_idxs_ct,
    const DT_UL n_cps,
    const DT_UL n_pts,
    const DT_UL n_fuzz_nos,
    const DT_UL run_type,
    DT_UL *rand_k,
    DT_UL *rand_i,
    DT_UL *rand_v,
    DT_UL *old_v_i_k,
    ) nogil:

    cdef:
        DT_UL dont_stop = 1
        DT_UL max_iters = 1000000, curr_iter_ct = 0
        DT_UL rand_k_, rand_i_, rand_v_, old_v_i_k_

    if run_type == 3:
        if old_v_i_k[0] < n_fuzz_nos:
            cp_rules_idx_ctr[rand_k[0], old_v_i_k[0]] += 1

        if rand_v[0] < n_fuzz_nos:
            cp_rules_idx_ctr[rand_k[0], rand_v[0]] -= 1 

        cp_rules[rand_k[0], rand_i[0]] = old_v_i_k[0]

        dont_stop = 0  # just in case

    else:
        while (dont_stop):
            curr_iter_ct += 1
            if curr_iter_ct > max_iters:
                printf("\n\n\n\n########Too many iterations in mod_cp_rules!########\n\n\n\n")
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
                if cp_rules_idx_ctr[rand_k_, rand_v_] < max_idxs_ct:
                    cp_rules[rand_k_, rand_i_] = rand_v_
                    cp_rules_idx_ctr[rand_k_, rand_v_] += 1

                    if old_v_i_k_ < n_fuzz_nos:
                        cp_rules_idx_ctr[rand_k_, old_v_i_k_] -= 1

                    dont_stop = 0

                else:
                    continue

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

