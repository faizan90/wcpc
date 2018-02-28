# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

### obj_ftns:False;True;False;False;False;False;False;False

### op_mp_obj_ftn_flag:True

import numpy as np
cimport numpy as np
from cython.parallel import prange

from .cp_obj_ftns cimport obj_ftn_refresh

ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef long long DT_LL
ctypedef unsigned long long DT_ULL
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64
DT_UL_NP = np.uint64


cdef extern from "math.h" nogil:
    cdef:
        DT_D exp(DT_D x)
        bint isnan(DT_D x)

cdef extern from "./rand_gen.h" nogil:
    cdef: 
        DT_D rand_c()
        void warm_up()  # call this at least once
        void re_seed(DT_ULL x)  # calls warm_up as well

warm_up()


cpdef get_obj_val(dict args_dict):
    cdef:
        # ulongs
        Py_ssize_t i, j, k, l
        DT_UL n_cps, n_time_steps, n_cpus, msgs, n_max = 0

        # doubles
        DT_D curr_obj_val, lo_freq_pen_wt, min_freq

        # 1D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] sel_cps

        # arrays for all obj. ftns.
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] obj_ftn_wts_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] ppt_cp_n_vals_arr

        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] in_cats_ppt_arr

        # ulongs for obj. ftns.
        Py_ssize_t q
        DT_UL n_cats

        # doubles obj. ftn. 2
        Py_ssize_t r
        DT_UL n_o_2_threshs

        # arrays for obj. ftn. 2
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] o_2_ppt_thresh_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] cats_ppt_mean_pis_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] cats_obj_2_vals_arr
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] cats_ppt_cp_mean_pis_arr

    # read everythings from the given dict. Must do explicitly.
    in_cats_ppt_arr = args_dict['in_cats_ppt_arr_calib']
    n_cats = in_cats_ppt_arr.shape[1]
    n_max = max(n_max, n_cats)
    n_time_steps = in_cats_ppt_arr.shape[0]
    o_2_ppt_thresh_arr = args_dict['o_2_ppt_thresh_arr']
    n_o_2_threshs = o_2_ppt_thresh_arr.shape[0]

    obj_ftn_wts_arr = args_dict['obj_ftn_wts_arr']
    sel_cps = args_dict['sel_cps']
    n_cps = args_dict['n_cps']
    n_cpus = args_dict['n_cpus']
    lo_freq_pen_wt = args_dict['lo_freq_pen_wt']
    min_freq = args_dict['min_freq']

    if 'msgs' in args_dict:
        msgs = <DT_UL> args_dict[ 'msgs']

    else:
        msgs = 0

    assert n_cps >= 2, 'n_cps cannot be less than 2!'

    if msgs:
        print('\n')
        print('Getting objective function value...')
        print('n_cats:', n_cats)
        print('o_2_ppt_thresh_arr:', o_2_ppt_thresh_arr)
        print('n_o_2_threshs:', n_o_2_threshs)
        print('n_cps:', n_cps)
        print('n_cpus:', n_cpus)
        print('obj_ftn_wts_arr:', obj_ftn_wts_arr)
        print('lo_freq_pen_wt:', lo_freq_pen_wt)
        print('min_freq:', min_freq)
        print('n_max:', n_max)
        print('in_cats_ppt_arr shape: (%d, %d)' % (in_cats_ppt_arr.shape[0], in_cats_ppt_arr.shape[1]))

    # initialize the required variables
    ppt_cp_n_vals_arr = np.full(n_cps, 0.0, dtype=DT_D_NP)

    # initialize obj. ftn. 2 variables
    cats_ppt_mean_pis_arr = np.full((n_cats, n_o_2_threshs), 0.0, dtype=DT_D_NP)
    cats_ppt_cp_mean_pis_arr = np.full((n_cats, n_cps, n_o_2_threshs), 0.0, dtype=DT_D_NP)
    cats_obj_2_vals_arr = np.full((n_cats, n_o_2_threshs), 0.0, dtype=DT_D_NP)

    # fill some arrays used for obj. 2 and 5 ftns.
    for q in range(n_cats):
        for r in range(n_o_2_threshs):
            cats_ppt_mean_pis_arr[q, r] = np.mean(in_cats_ppt_arr[:, q] > o_2_ppt_thresh_arr[r])
            
            assert (not isnan(cats_ppt_mean_pis_arr[q, r]) and (cats_ppt_mean_pis_arr[q, r] > 0)), (
                (q, r, cats_ppt_mean_pis_arr[q, r], o_2_ppt_thresh_arr[r], np.sum(in_cats_ppt_arr[:, q] > o_2_ppt_thresh_arr[r]), in_cats_ppt_arr[:, q].max()))

    # calc obj ftn value
    curr_obj_val = obj_ftn_refresh(
        in_cats_ppt_arr,
        n_cats,
        cats_ppt_cp_mean_pis_arr,
        cats_ppt_mean_pis_arr,
        o_2_ppt_thresh_arr,
        cats_obj_2_vals_arr,
        n_o_2_threshs,
        ppt_cp_n_vals_arr,
        obj_ftn_wts_arr,
        sel_cps,
        lo_freq_pen_wt,
        min_freq,
        n_cpus,
        n_cps,
        n_max,
        n_time_steps,
        )

    out_dict = {}
    for key in args_dict:
        out_dict[key] = args_dict[key]

    out_dict['n_max'] = n_max
    out_dict['n_time_steps_calib'] = n_time_steps
    out_dict['curr_obj_val'] = curr_obj_val
    return out_dict

