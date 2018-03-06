# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange

from .gen_mod_cp_rules cimport gen_cp_rules
from .memb_ftns cimport calc_membs_dof_cps

ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef long long DT_LL
ctypedef unsigned long long DT_ULL
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64
DT_UL_NP = np.uint64


cpdef get_rand_cp_rules(dict args_dict):
    cdef:
        Py_ssize_t i, j

        # ulongs
        DT_UL max_idxs_ct, n_gens, n_cps, n_pts, n_fuzz_nos, n_cpus, msgs
        DT_UL curr_gen = 0, no_cp_val, cp_exists_sum

        # doubles for obj. ftns.
        DT_D p_l

        # 1D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] chnge_steps, cp_exists_arr
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] sel_cps, old_sel_cps

        # 2D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] cp_rules
        np.ndarray[DT_UL_NP_t, ndim=3, mode='c'] mult_cp_rules

        # 2D double arrays
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] anom, fuzz_nos_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] dofs_arr

        # 3D double arrays
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] mu_i_k_arr
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] cp_dof_arr
        
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] cp_rules_idx_ctr

    n_cps = args_dict['n_cps']
    n_cpus = args_dict['n_cpus']
    max_idxs_ct = args_dict['max_idxs_ct']
    n_gens = args_dict['n_gens'] 

    no_cp_val = args_dict['no_cp_val']
    p_l = args_dict['p_l']
    fuzz_nos_arr = args_dict['fuzz_nos_arr']
    anom = args_dict['anom']

    n_fuzz_nos = fuzz_nos_arr.shape[0]
    n_pts = anom.shape[1]
    n_time_steps = anom.shape[0]

    if 'msgs' in args_dict:
        msgs = <DT_UL> args_dict[ 'msgs']

    else:
        msgs = 0

    if max_idxs_ct > (n_pts / n_fuzz_nos):
        max_idxs_ct = <DT_UL> (n_pts / n_fuzz_nos)
        print(("\n\n\n\n######### max_idxs_ct reset to %d!#########\n\n\n\n" % 
               max_idxs_ct))

    cp_rules_idx_ctr = np.zeros(shape=(n_cps, n_fuzz_nos), dtype=DT_UL_NP)

    mu_i_k_arr = np.zeros(shape=(n_time_steps, n_cps, n_pts), dtype=DT_D_NP)
    cp_dof_arr = np.zeros(shape=(n_time_steps, n_cps, n_fuzz_nos), 
                          dtype=DT_D_NP)

    mult_cp_rules = np.random.randint(0, 
                                      n_fuzz_nos + 1, 
                                      size=(n_gens, n_cps, n_pts), 
                                      dtype=DT_UL_NP)

    mult_sel_cps = np.full((n_gens, n_time_steps), 
                           0, 
                           dtype=DT_UL_NP)

    old_sel_cps = np.full(n_time_steps, no_cp_val, dtype=DT_UL_NP)
    cp_exists_arr = np.zeros(n_cps, dtype=DT_UL_NP)

    chnge_steps = np.zeros(n_time_steps, dtype=DT_UL_NP)
    dofs_arr = np.zeros((n_time_steps, n_cps), dtype=DT_D_NP)
    
    print('\n')

    while curr_gen < n_gens:
        gen_cp_rules(mult_cp_rules[curr_gen, :, :],
                     cp_rules_idx_ctr,
                     max_idxs_ct,
                     n_cps,
                     n_pts,
                     n_fuzz_nos,
                     n_cpus)

        calc_membs_dof_cps(
            mult_cp_rules[curr_gen, :],
            mu_i_k_arr,
            cp_dof_arr,
            anom,
            fuzz_nos_arr,
            dofs_arr,
            mult_sel_cps[curr_gen, :],
            old_sel_cps,
            chnge_steps,
            no_cp_val,
            p_l,
            n_cpus,
            n_time_steps,
            n_pts,
            n_cps,
            n_fuzz_nos)
        
        for i in range(n_cps):
            cp_exists_arr[i] = 0
            for j in range(n_time_steps):
                if mult_sel_cps[curr_gen, j] == i:
                    cp_exists_arr[i] = 1
                    break
                
        cp_exists_sum = 0
        for i in range(n_cps):
            cp_exists_sum += cp_exists_arr[i]
        
        if cp_exists_sum == n_cps:
            curr_gen += 1
            print('gen %0.3d out of %0.3d gens!' % (curr_gen, n_gens))
        else:
            continue

    print('\n')
    args_dict['mult_sel_cps'] = mult_sel_cps
    args_dict['mult_cp_rules'] = mult_cp_rules
    return args_dict
