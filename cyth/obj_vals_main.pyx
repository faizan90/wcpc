# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

### obj_ftns:False;False;False;False;False;False;True;False

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
        DT_UL n_cps, n_time_steps, n_cpus, msgs, n_max = 0, n_gens = 1
        DT_UL mult_obj_vals_flag

        # doubles
        DT_D curr_obj_val, lo_freq_pen_wt, min_freq

        # 1D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] sel_cps
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] mult_sel_cps

        # arrays for all obj. ftns.
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] obj_ftn_wts_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] obj_vals_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] ppt_cp_n_vals_arr

        # for obj. ftn. 7
        DT_D mean_tri_wet = 0.0
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] mean_cp_tri_wet_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] tri_wet_arr

    # read everythings from the given dict. Must do explicitly.
    in_wet_arr_calib = args_dict['in_wet_arr_calib']
    n_time_steps = in_wet_arr_calib.shape[0]

    obj_ftn_wts_arr = args_dict['obj_ftn_wts_arr']
    if 'mult_obj_vals_flag' in args_dict:
        mult_sel_cps = args_dict['mult_sel_cps']
        mult_obj_vals_flag = <DT_UL> args_dict['mult_obj_vals_flag']
        n_gens = mult_sel_cps.shape[0]

    else:
        sel_cps = args_dict['sel_cps']
        mult_obj_vals_flag = 0

    n_cps = args_dict['n_cps']
    n_cpus = args_dict['n_cpus']
    lo_freq_pen_wt = args_dict['lo_freq_pen_wt']
    min_freq = args_dict['min_freq']

    if 'msgs' in args_dict:
        msgs = <DT_UL> args_dict[ 'msgs']

    else:
        msgs = 0

    assert n_cps >= 1, 'n_cps cannot be less than 1!'

    if msgs:
        print('\n')
        print('Getting objective function value...')
        print('in_wet_arr shape:', (in_wet_arr_calib.shape[0], in_wet_arr_calib.shape[1]))
        print('n_cps:', n_cps)
        print('n_cpus:', n_cpus)
        print('obj_ftn_wts_arr:', obj_ftn_wts_arr)
        print('lo_freq_pen_wt:', lo_freq_pen_wt)
        print('min_freq:', min_freq)
        print('n_max:', n_max)
        print('mult_obj_vals_flag:', mult_obj_vals_flag)
        print('n_gens:', n_gens)

    # initialize the required variables
    ppt_cp_n_vals_arr = np.full(n_cps, 0.0, dtype=DT_D_NP)
    obj_vals_arr = np.full(n_gens, 0.0, dtype=DT_D_NP)

    # initialize obj. ftn. 6 variables
    mean_cp_tri_wet_arr = np.full(n_cps, 0.0, dtype=DT_D_NP)
    tri_wet_arr = np.full(n_time_steps, 0.0, dtype=DT_D_NP)

    # obj. 7 ftns.
    for i in range(n_time_steps):
        tri_wet_arr[i] += np.sum(in_wet_arr_calib[i, :])
        tri_wet_arr[i] += in_wet_arr_calib[i, 0] + in_wet_arr_calib[i, 1] - in_wet_arr_calib[i, 2] + 1
        tri_wet_arr[i] += in_wet_arr_calib[i, 1] + in_wet_arr_calib[i, 2] - in_wet_arr_calib[i, 0] + 1
        tri_wet_arr[i] += in_wet_arr_calib[i, 0] + in_wet_arr_calib[i, 2] - in_wet_arr_calib[i, 1] + 1

    mean_tri_wet = tri_wet_arr.mean()
    assert ((not isnan(mean_tri_wet)) and (mean_tri_wet > 0))

    # calc obj ftn value
    if mult_obj_vals_flag:
        for i in range(n_gens):
            curr_obj_val = obj_ftn_refresh(
                mean_tri_wet,
                mean_cp_tri_wet_arr,
                tri_wet_arr,
                ppt_cp_n_vals_arr,
                obj_ftn_wts_arr,
                mult_sel_cps[i],
                lo_freq_pen_wt,
                min_freq,
                n_cpus,
                n_cps,
                n_max,
                n_time_steps,
                )

            obj_vals_arr[i] = curr_obj_val

    else:
        curr_obj_val = obj_ftn_refresh(
            mean_tri_wet,
            mean_cp_tri_wet_arr,
            tri_wet_arr,
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
    if mult_obj_vals_flag:
        out_dict['obj_vals_arr'] = obj_vals_arr

    return out_dict

