# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef long long DT_LL
ctypedef unsigned long long DT_ULL
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64
DT_UL_NP = np.uint64

from .memb_ftns cimport calc_membs_dof_cps

cpdef dict _assign_cps(dict args_dict):
    cdef:
        # ulongs
        DT_UL n_cps, n_pts, n_time_steps, n_fuzz_nos
        DT_UL no_cp_val, n_cpus

        # doubles for obj. ftns.
        DT_D p_l

        # 1D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] chnge_steps
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] sel_cps, old_sel_cps

        # 2D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] cp_rules

        # 2D double arrays
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] anom, fuzz_nos_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] dofs_arr

        # 3D double arrays
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] mu_i_k_arr
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] cp_dof_arr

    no_cp_val = args_dict['no_cp_val']
    p_l = args_dict['p_l']
    fuzz_nos_arr = args_dict['fuzz_nos_arr']
    cp_rules = args_dict['cp_rules']
    n_cpus = args_dict['n_cpus']
    n_fuzz_nos = fuzz_nos_arr.shape[0]
    anom = args_dict['anom']
    
    n_cps = cp_rules.shape[0]
    n_pts = anom.shape[1]
    n_time_steps = anom.shape[0]

    mu_i_k_arr = np.zeros(shape=(n_time_steps, n_cps, n_pts), dtype=DT_D_NP)
    cp_dof_arr = np.zeros(shape=(n_time_steps, n_cps, n_fuzz_nos), 
                          dtype=DT_D_NP)

    sel_cps = np.full(n_time_steps, 0, dtype=DT_UL_NP)
    old_sel_cps = np.full(n_time_steps, no_cp_val, dtype=DT_UL_NP)

    chnge_steps = np.zeros(n_time_steps, dtype=DT_UL_NP)
    dofs_arr = np.zeros((n_time_steps, n_cps), dtype=DT_D_NP)

    calc_membs_dof_cps(
        cp_rules,
        mu_i_k_arr,
        cp_dof_arr,
        anom,
        fuzz_nos_arr,
        dofs_arr,
        sel_cps,
        old_sel_cps,
        chnge_steps,
        no_cp_val,
        p_l,
        n_cpus,
        n_time_steps,
        n_pts,
        n_cps,
        n_fuzz_nos)
    
    assert np.all(np.isfinite(mu_i_k_arr))
    assert np.all(np.isfinite(cp_dof_arr))
    assert np.all(np.isfinite(dofs_arr))
    
    ret_dict = {}
    ret_dict['sel_cps'] = sel_cps
    ret_dict['dofs_arr'] = dofs_arr   
    return ret_dict