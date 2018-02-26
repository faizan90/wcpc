# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

### obj_ftns:False;False;False;False;False;False;False;True

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

        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] in_lorenz_arr

        Py_ssize_t t
        DT_UL n_lors

        # arrays for obj. ftn. 8
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] mean_lor_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] lor_cp_mean_arr

    # read everythings from the given dict. Must do explicitly.
    in_lorenz_arr = args_dict['in_lorenz_arr_calib']
    n_lors = in_lorenz_arr.shape[1]
    n_max = max(n_max, n_lors)
    assert n_lors, 'n_lors cannot be zero!'
    n_time_steps = in_lorenz_arr.shape[0]

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
        print('in_lorenz_arr shape:', (in_lorenz_arr.shape[0], in_lorenz_arr.shape[1]))
        print('n_cps:', n_cps)
        print('n_cpus:', n_cpus)
        print('obj_ftn_wts_arr:', obj_ftn_wts_arr)
        print('lo_freq_pen_wt:', lo_freq_pen_wt)
        print('min_freq:', min_freq)
        print('n_max:', n_max)

    # initialize the required variables
    ppt_cp_n_vals_arr = np.full(n_cps, 0.0, dtype=DT_D_NP)

    # initialize obj. ftn. 8 variables
    mean_lor_arr = np.full(n_lors, 0.0, dtype=DT_D_NP)
    lor_cp_mean_arr = np.full((n_cps, n_lors), 0.0, dtype=DT_D_NP)

    # fill some arrays used for obj. 8 ftn.
    for t in range(n_lors):
        mean_lor_arr[t] = np.mean(in_lorenz_arr[:, t])
        assert ((not isnan(mean_lor_arr[t])) and (mean_lor_arr[t] > 0))

    # calc obj ftn value
    curr_obj_val = obj_ftn_refresh(
        in_lorenz_arr,
        mean_lor_arr,
        lor_cp_mean_arr,
        n_lors,
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

