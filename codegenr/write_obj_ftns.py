'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path
from .core import CodeGenr, write_pyxbld


def write_obj_ftns_lines(params_dict):
    module_name = 'cp_obj_ftns'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    infer_types = params_dict['infer_types']
    out_dir = params_dict['out_dir']

    obj_1_flag = params_dict['obj_1_flag']
    obj_2_flag = params_dict['obj_2_flag']
    obj_3_flag = params_dict['obj_3_flag']
    obj_4_flag = params_dict['obj_4_flag']
    obj_5_flag = params_dict['obj_5_flag']

    pyxcd = CodeGenr(tab=tab)
    pxdcd = CodeGenr(tab=tab)
    pyxbldcd = CodeGenr(tab=tab)

    #==========================================================================
    # add cython flags
    #==========================================================================
    pyxcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pyxcd.w('# cython: boundscheck=%s' % boundscheck)
    pyxcd.w('# cython: wraparound=%s' % str(wraparound))
    pyxcd.w('# cython: cdivision=%s' % str(cdivision))
    pyxcd.w('# cython: language_level=%d' % int(language_level))
    pyxcd.w('# cython: infer_types(%s)' % str(infer_types))
    pyxcd.els()

    pxdcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pxdcd.w('# cython: boundscheck=%s' % boundscheck)
    pxdcd.w('# cython: wraparound=%s' % str(wraparound))
    pxdcd.w('# cython: cdivision=%s' % str(cdivision))
    pxdcd.w('# cython: language_level=%d' % int(language_level))
    pxdcd.w('# cython: infer_types(%s)' % str(infer_types))
    pxdcd.els()

    #==========================================================================
    # add imports
    #==========================================================================
    pyxcd.w('import numpy as np')
    pyxcd.w('cimport numpy as np')
    pyxcd.w('from cython.parallel import prange')
    pyxcd.els()

    pxdcd.w('import numpy as np')
    pxdcd.w('cimport numpy as np')
    pxdcd.els()

    #==========================================================================
    # declare types
    #==========================================================================
    pxdcd.w('ctypedef double DT_D')
    pxdcd.w('ctypedef unsigned long DT_UL')
    pxdcd.w('ctypedef long long DT_LL')
    pxdcd.w('ctypedef unsigned long long DT_ULL')
    pxdcd.w('ctypedef np.float64_t DT_D_NP_t')
    pxdcd.w('ctypedef np.uint64_t DT_UL_NP_t')
    pxdcd.els()

    pyxcd.w('DT_D_NP = np.float64')
    pyxcd.w('DT_UL_NP = np.uint64')
    pyxcd.els()

    pxdcd.w('DT_D_NP = np.float64')
    pxdcd.w('DT_UL_NP = np.uint64')
    pxdcd.els(2)

    #==========================================================================
    # add external imports
    #==========================================================================
    pyxcd.els()
    pyxcd.w('cdef extern from "math.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')

    pyxcd.ind()
    pyxcd.w('DT_D exp(DT_D x)')
    pyxcd.w('DT_D log(DT_D x)')
    pyxcd.w('DT_D abs(DT_D x)')
    pyxcd.w('bint isnan(DT_D x)')
    pyxcd.ded()
    pyxcd.ded()

    pyxcd.w('cdef extern from "./rand_gen.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('DT_D rand_c()')
    pyxcd.w('void warm_up()  # call this at least once')
    pyxcd.w('void re_seed(DT_ULL x)  # calls warm_up as well')
    pyxcd.ded()
    pyxcd.ded()
    pyxcd.w('warm_up()')
    pyxcd.els(2)

    #==========================================================================
    # The objective function
    #==========================================================================

    # refresh inputs
    pyxcd.w('cdef DT_D obj_ftn_refresh(')
    pyxcd.ind()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('const DT_D_NP_t[:, :] in_ppt_arr,')
        pyxcd.w('const DT_UL n_stns,')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('const DT_D_NP_t[:, :] in_cats_ppt_arr,')
        pyxcd.w('const DT_UL n_cats,')

    if obj_1_flag or obj_3_flag or obj_2_flag or obj_5_flag:
        pyxcd.w('const DT_D min_abs_ppt_thresh,')

    if obj_1_flag:
        pyxcd.w('DT_D_NP_t[:, :, :] ppt_cp_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:, :] ppt_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:] o_1_ppt_thresh_arr,')
        pyxcd.w('DT_D_NP_t[:, :] stns_obj_1_vals_arr,')
        pyxcd.w('const DT_UL n_o_1_threshs,')

    if obj_2_flag:
        pyxcd.w('DT_D_NP_t[:, :, :] cats_ppt_cp_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:, :] cats_ppt_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:] o_2_ppt_thresh_arr,')
        pyxcd.w('DT_D_NP_t[:, :] cats_obj_2_vals_arr,')
        pyxcd.w('const DT_UL n_o_2_threshs,')

    if obj_3_flag:
        pyxcd.w('DT_D_NP_t[:, :] ppt_cp_mean_arr,')
        pyxcd.w('const DT_D_NP_t[:] ppt_mean_arr,')

    if obj_4_flag:
        pyxcd.w('const DT_D_NP_t[:, :] in_wet_arr_calib,')
        pyxcd.w('const DT_D_NP_t[:, :] ppt_mean_wet_arr,')
        pyxcd.w('const DT_D_NP_t[:] o_4_p_thresh_arr,')
        pyxcd.w('DT_UL_NP_t[:, :, :] ppt_cp_mean_wet_arr,')
        pyxcd.w('DT_D_NP_t[:, :] nebs_wet_obj_vals_arr,')
        pyxcd.w('const DT_UL n_nebs,')
        pyxcd.w('const DT_UL n_o_4_threshs,')

    if obj_5_flag:
        pyxcd.w('DT_D_NP_t[:, :] cats_ppt_cp_mean_arr,')
        pyxcd.w('const DT_D_NP_t[:] cats_ppt_mean_arr,')

    pyxcd.w('DT_D_NP_t[:] ppt_cp_n_vals_arr,')
    pyxcd.w('const DT_D_NP_t[:] obj_ftn_wts_arr,')
    pyxcd.w('const DT_UL_NP_t[:] sel_cps,')
    pyxcd.w('const DT_UL n_cpus,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_max,')
    pyxcd.w('const DT_UL n_time_steps,')
    pyxcd.w(') nogil:')
    pyxcd.els()

    pxdcd.w('cdef DT_D obj_ftn_refresh(')
    pxdcd.ind()

    if obj_1_flag or obj_3_flag:
        pxdcd.w('const DT_D_NP_t[:, :] in_ppt_arr,')
        pxdcd.w('const DT_UL n_stns,')

    if obj_2_flag or obj_5_flag:
        pxdcd.w('const DT_D_NP_t[:, :] in_cats_ppt_arr,')
        pxdcd.w('const DT_UL n_cats,')

    if obj_1_flag or obj_3_flag or obj_2_flag or obj_5_flag:
        pxdcd.w('const DT_D min_abs_ppt_thresh,')

    if obj_1_flag:
        pxdcd.w('DT_D_NP_t[:, :, :] ppt_cp_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:, :] ppt_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:] o_1_ppt_thresh_arr,')
        pxdcd.w('DT_D_NP_t[:, :] stns_obj_1_vals_arr,')
        pxdcd.w('const DT_UL n_o_1_threshs,')

    if obj_2_flag:
        pxdcd.w('DT_D_NP_t[:, :, :] cats_ppt_cp_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:, :] cats_ppt_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:] o_2_ppt_thresh_arr,')
        pxdcd.w('DT_D_NP_t[:, :] cats_obj_2_vals_arr,')
        pxdcd.w('const DT_UL n_o_2_threshs,')

    if obj_3_flag:
        pxdcd.w('DT_D_NP_t[:, :] ppt_cp_mean_arr,')
        pxdcd.w('const DT_D_NP_t[:] ppt_mean_arr,')

    if obj_4_flag:
        pxdcd.w('const DT_D_NP_t[:, :] in_wet_arr_calib,')
        pxdcd.w('const DT_D_NP_t[:, :] ppt_mean_wet_arr,')
        pxdcd.w('const DT_D_NP_t[:] o_4_p_thresh_arr,')
        pxdcd.w('DT_UL_NP_t[:, :, :] ppt_cp_mean_wet_arr,')
        pxdcd.w('DT_D_NP_t[:, :] nebs_wet_obj_vals_arr,')
        pxdcd.w('const DT_UL n_nebs,')
        pxdcd.w('const DT_UL n_o_4_threshs,')

    if obj_5_flag:
        pxdcd.w('DT_D_NP_t[:, :] cats_ppt_cp_mean_arr,')
        pxdcd.w('const DT_D_NP_t[:] cats_ppt_mean_arr,')

    pxdcd.w('DT_D_NP_t[:] ppt_cp_n_vals_arr,')
    pxdcd.w('const DT_D_NP_t[:] obj_ftn_wts_arr,')
    pxdcd.w('const DT_UL_NP_t[:] sel_cps,')
    pxdcd.w('const DT_UL n_cpus,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_max,')
    pxdcd.w('const DT_UL n_time_steps,')
    pxdcd.w(') nogil')
    pxdcd.ded()
    pxdcd.els()

    pyxcd.w('# declare/initialize variables')
    pyxcd.w('cdef:')

    pyxcd.ind()
    pyxcd.w('Py_ssize_t i, j, s')
    pyxcd.w('DT_D _, obj_val = 0.0')
    pyxcd.els()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('Py_ssize_t m')
        pyxcd.w('DT_D curr_ppt')
        pyxcd.els()

    if obj_2_flag or obj_5_flag:
        pyxcd.w('Py_ssize_t q')
        pyxcd.w('DT_D curr_cat_ppt')
        pyxcd.els()

    if obj_1_flag:
        pyxcd.w('Py_ssize_t p')
        pyxcd.w('DT_D o_1 = 0.0')
        pyxcd.w('DT_D curr_ppt_pi_diff')
        pyxcd.els()

    if obj_2_flag:
        pyxcd.w('Py_ssize_t r')
        pyxcd.w('DT_D o_2 = 0.0')
        pyxcd.w('DT_D curr_cat_ppt_pi_diff')
        pyxcd.els()

    if obj_3_flag:
        pyxcd.w('DT_D o_3 = 0.0')
        pyxcd.w('DT_D cp_ppt_mean')
        pyxcd.w('DT_D curr_ppt_diff')
        pyxcd.els()

    if obj_4_flag:
        pyxcd.w('Py_ssize_t n, o')
        pyxcd.w('DT_D o_4 = 0.0')
        pyxcd.w('DT_D curr_ppt_wet_diff')
        pyxcd.els()

    if obj_5_flag:
        pyxcd.w('DT_D o_5 = 0.0')
        pyxcd.w('DT_D cp_cat_ppt_mean')
        pyxcd.w('DT_D curr_cat_ppt_diff')
        pyxcd.els()

    pyxcd.ded()

    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('ppt_cp_n_vals_arr[j] = 0')
    pyxcd.w('for i in range(n_time_steps):')
    pyxcd.ind()
    pyxcd.w('if sel_cps[i] != j:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()
    pyxcd.w('ppt_cp_n_vals_arr[j] += 1')
    pyxcd.ded(lev=2)

    # the main loop
    pyxcd.w(
        'for s in prange(n_max, schedule=\'dynamic\', nogil=True, num_threads=n_cpus):')
    if obj_1_flag or obj_3_flag:
        pyxcd.ind()
        pyxcd.w('if s < n_stns:')
        pyxcd.ind()
        pyxcd.w('m = s')

        if obj_3_flag:
            pyxcd.w('curr_ppt_diff = 0')
            pyxcd.els()

        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('stns_obj_1_vals_arr[m, p] = 0.0')
            pyxcd.els()
            pyxcd.w('for j in range(n_cps):')
            pyxcd.ind()
            pyxcd.w('ppt_cp_mean_pis_arr[m, j, p] = 0.0')
            pyxcd.ded(lev=2)

        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('if ppt_cp_n_vals_arr[j] == 0:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('cp_ppt_mean = 0')

        pyxcd.els()

        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('if sel_cps[i] != j:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('curr_ppt = in_ppt_arr[i, m]')
        pyxcd.els()

        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('if curr_ppt < o_1_ppt_thresh_arr[p]:')
            pyxcd.ind()
            pyxcd.w('break')
            pyxcd.ded()

            pyxcd.w(
                'ppt_cp_mean_pis_arr[m, j, p] = ppt_cp_mean_pis_arr[m, j, p] + 1')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('cp_ppt_mean = cp_ppt_mean + curr_ppt')

        pyxcd.ded()

        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('stns_obj_1_vals_arr[m, p] = stns_obj_1_vals_arr[m, p] + '
                    'ppt_cp_n_vals_arr[j] * ((ppt_cp_mean_pis_arr[m, j, p] / '
                    'ppt_cp_n_vals_arr[j]) - ppt_mean_pis_arr[m, p])**2')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('ppt_cp_mean_arr[j, m] = cp_ppt_mean')
            pyxcd.w('cp_ppt_mean = cp_ppt_mean / ppt_cp_n_vals_arr[j]')
            pyxcd.els()

            pyxcd.w('if ppt_mean_arr[m] > min_abs_ppt_thresh:')
            pyxcd.ind()
            pyxcd.w('_ = cp_ppt_mean / ppt_mean_arr[m]')
            pyxcd.w('if _ <= 0:')
            pyxcd.ind()
            pyxcd.w('_ = 1e-100')
            pyxcd.ded()
            pyxcd.w(
                'curr_ppt_diff = curr_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(log(_)))')
            pyxcd.ded(lev=2)
            pyxcd.w('o_3 += (curr_ppt_diff / n_time_steps)')
        else:
            pyxcd.ded()

        pyxcd.ded(lev=2)

    if obj_2_flag or obj_5_flag:
        pyxcd.ind()
        pyxcd.w('if s < n_cats:')
        pyxcd.ind()
        pyxcd.w('q = s')
#         pyxcd.w('for q in prange(n_cats, schedule=\'dynamic\'):')
#         pyxcd.ind()

        if obj_5_flag:
            pyxcd.w('curr_cat_ppt_diff = 0')
            pyxcd.els()

        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('cats_obj_2_vals_arr[q, r] = 0.0')
            pyxcd.els()
            pyxcd.w('for j in range(n_cps):')
            pyxcd.ind()
            pyxcd.w('cats_ppt_cp_mean_pis_arr[q, j, r] = 0.0')
            pyxcd.ded(lev=2)

        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('if ppt_cp_n_vals_arr[j] == 0:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('cp_cat_ppt_mean = 0')

        pyxcd.els()

        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('if sel_cps[i] != j:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('curr_cat_ppt = in_cats_ppt_arr[i, q]')
        pyxcd.els()

        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('if curr_cat_ppt < o_2_ppt_thresh_arr[r]:')
            pyxcd.ind()
            pyxcd.w('break')
            pyxcd.ded()

            pyxcd.w(
                'cats_ppt_cp_mean_pis_arr[q, j, r] = cats_ppt_cp_mean_pis_arr[q, j, r] + 1')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('cp_cat_ppt_mean = cp_cat_ppt_mean + curr_cat_ppt')

        pyxcd.ded()

        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('cats_obj_2_vals_arr[q, r] = cats_obj_2_vals_arr[q, r] + '
                    'ppt_cp_n_vals_arr[j] * ((cats_ppt_cp_mean_pis_arr[q, j, r] / '
                    'ppt_cp_n_vals_arr[j]) - cats_ppt_mean_pis_arr[q, r])**2')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('cats_ppt_cp_mean_arr[j, q] = cp_cat_ppt_mean')
            pyxcd.w('cp_cat_ppt_mean = cp_cat_ppt_mean / ppt_cp_n_vals_arr[j]')
            pyxcd.els()

            pyxcd.w('if cats_ppt_mean_arr[q] > min_abs_ppt_thresh:')
            pyxcd.ind()
            pyxcd.w('_ = cp_cat_ppt_mean / cats_ppt_mean_arr[q]')
            pyxcd.w('if _ <= 0:')
            pyxcd.ind()
            pyxcd.w('_ = 1e-100')
            pyxcd.ded()
            pyxcd.w(
                'curr_cat_ppt_diff = curr_cat_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(log(_)))')
            pyxcd.ded(lev=2)
            pyxcd.w('o_5 += (curr_cat_ppt_diff / n_time_steps)')
        else:
            pyxcd.ded()

        pyxcd.ded(lev=2)

    if obj_4_flag:
        pyxcd.ind()
        pyxcd.w('if s < n_nebs:')
        pyxcd.ind()
        pyxcd.w('n = s')
#         pyxcd.w('for n in prange(n_nebs, schedule=\'dynamic\'):')
#         pyxcd.ind()
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('nebs_wet_obj_vals_arr[n, o] = 0.0')
        pyxcd.els()
        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('ppt_cp_mean_wet_arr[n, j, o] = 0')
        pyxcd.ded(lev=2)

        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('if ppt_cp_n_vals_arr[j] == 0:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('if sel_cps[i] != j:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('if in_wet_arr_calib[i, n] < o_4_p_thresh_arr[o]:')
        pyxcd.ind()
        pyxcd.w('break')
        pyxcd.ded()

        pyxcd.w(
            'ppt_cp_mean_wet_arr[n, j, o] = ppt_cp_mean_wet_arr[n, j, o] + 1')
        pyxcd.ded(lev=2)

        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('nebs_wet_obj_vals_arr[n, o] = nebs_wet_obj_vals_arr[n, o] + '
                'ppt_cp_n_vals_arr[j] * ((ppt_cp_mean_wet_arr[n, j, o] / '
                'ppt_cp_n_vals_arr[j]) - ppt_mean_wet_arr[n, o])**2')
        pyxcd.ded(lev=4)

    if obj_1_flag:
        pyxcd.w('for p in range(n_o_1_threshs):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_pi_diff = 0.0')
        pyxcd.w('for m in range(n_stns):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_pi_diff += stns_obj_1_vals_arr[m, p]')
        pyxcd.ded()

        pyxcd.w('o_1 += (curr_ppt_pi_diff / n_time_steps)**0.5')
        pyxcd.ded()

    if obj_2_flag:
        pyxcd.w('for r in range(n_o_2_threshs):')
        pyxcd.ind()
        pyxcd.w('curr_cat_ppt_pi_diff = 0.0')
        pyxcd.w('for q in range(n_cats):')
        pyxcd.ind()
        pyxcd.w('curr_cat_ppt_pi_diff += cats_obj_2_vals_arr[q, r]')
        pyxcd.ded()

        pyxcd.w('o_2 += (curr_cat_ppt_pi_diff / n_time_steps)**0.5')
        pyxcd.ded()

    if obj_4_flag:
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_wet_diff = 0.0')
        pyxcd.w('for n in range(n_nebs):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_wet_diff += nebs_wet_obj_vals_arr[n, o]')
        pyxcd.ded()
        pyxcd.w('o_4 += (curr_ppt_wet_diff / n_time_steps)**0.5')
        pyxcd.ded()

    if obj_1_flag:
        pyxcd.w('obj_val += (o_1 * obj_ftn_wts_arr[0])')
    if obj_2_flag:
        pyxcd.w('obj_val += (o_2 * obj_ftn_wts_arr[1])')
    if obj_3_flag:
        pyxcd.w('obj_val += (o_3 * obj_ftn_wts_arr[2])')
    if obj_4_flag:
        pyxcd.w('obj_val += (o_4 * obj_ftn_wts_arr[3])')
    if obj_5_flag:
        pyxcd.w('obj_val += (o_5 * obj_ftn_wts_arr[4])')
    pyxcd.w('return obj_val')
    pyxcd.ded()

    assert pyxcd.level == 0, \
        'Level should be zero instead of %d' % pyxcd.level

    # update the obj ftn
    pyxcd.w('cdef DT_D obj_ftn_update(')
    pyxcd.ind()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('const DT_D_NP_t[:, :] in_ppt_arr,')
        pyxcd.w('const DT_UL n_stns,')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('const DT_D_NP_t[:, :] in_cats_ppt_arr,')
        pyxcd.w('const DT_UL n_cats,')

    if obj_1_flag or obj_3_flag or obj_2_flag or obj_5_flag:
        pyxcd.w('const DT_D min_abs_ppt_thresh,')

    if obj_1_flag:
        pyxcd.w('DT_D_NP_t[:, :, :] ppt_cp_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:, :] ppt_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:] o_1_ppt_thresh_arr,')
        pyxcd.w('DT_D_NP_t[:, :] stns_obj_1_vals_arr,')
        pyxcd.w('const DT_UL n_o_1_threshs,')

    if obj_2_flag:
        pyxcd.w('DT_D_NP_t[:, :, :] cats_ppt_cp_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:, :] cats_ppt_mean_pis_arr,')
        pyxcd.w('DT_D_NP_t[:] o_2_ppt_thresh_arr,')
        pyxcd.w('DT_D_NP_t[:, :] cats_obj_2_vals_arr,')
        pyxcd.w('const DT_UL n_o_2_threshs,')

    if obj_3_flag:
        pyxcd.w('DT_D_NP_t[:, :] ppt_cp_mean_arr,')
        pyxcd.w('const DT_D_NP_t[:] ppt_mean_arr,')

    if obj_4_flag:
        pyxcd.w('const DT_D_NP_t[:, :] in_wet_arr_calib,')
        pyxcd.w('const DT_D_NP_t[:, :] ppt_mean_wet_arr,')
        pyxcd.w('const DT_D_NP_t[:] o_4_p_thresh_arr,')
        pyxcd.w('DT_UL_NP_t[:, :, :] ppt_cp_mean_wet_arr,')
        pyxcd.w('DT_D_NP_t[:, :] nebs_wet_obj_vals_arr,')
        pyxcd.w('const DT_UL n_nebs,')
        pyxcd.w('const DT_UL n_o_4_threshs,')

    if obj_5_flag:
        pyxcd.w('DT_D_NP_t[:, :] cats_ppt_cp_mean_arr,')
        pyxcd.w('const DT_D_NP_t[:] cats_ppt_mean_arr,')

    pyxcd.w('DT_D_NP_t[:] ppt_cp_n_vals_arr,')
    pyxcd.w('const DT_D_NP_t[:] obj_ftn_wts_arr,')
    pyxcd.w('const DT_UL_NP_t[:] sel_cps,')
    pyxcd.w('const DT_UL_NP_t[:] old_sel_cps,')
    pyxcd.w('const DT_UL_NP_t[:] chnge_steps,')
    pyxcd.w('const DT_UL n_cpus,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_max,')
    pyxcd.w('const DT_UL n_time_steps,')
    pyxcd.w(') nogil:')
    pyxcd.els()

    pxdcd.w('cdef DT_D obj_ftn_update(')
    pxdcd.ind()
    if obj_1_flag or obj_3_flag:
        pxdcd.w('const DT_D_NP_t[:, :] in_ppt_arr,')
        pxdcd.w('const DT_UL n_stns,')

    if obj_2_flag or obj_5_flag:
        pxdcd.w('const DT_D_NP_t[:, :] in_cats_ppt_arr,')
        pxdcd.w('const DT_UL n_cats,')

    if obj_1_flag or obj_3_flag or obj_2_flag or obj_5_flag:
        pxdcd.w('const DT_D min_abs_ppt_thresh,')

    if obj_1_flag:
        pxdcd.w('DT_D_NP_t[:, :, :] ppt_cp_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:, :] ppt_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:] o_1_ppt_thresh_arr,')
        pxdcd.w('DT_D_NP_t[:, :] stns_obj_1_vals_arr,')
        pxdcd.w('const DT_UL n_o_1_threshs,')

    if obj_2_flag:
        pxdcd.w('DT_D_NP_t[:, :, :] cats_ppt_cp_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:, :] cats_ppt_mean_pis_arr,')
        pxdcd.w('DT_D_NP_t[:] o_2_ppt_thresh_arr,')
        pxdcd.w('DT_D_NP_t[:, :] cats_obj_2_vals_arr,')
        pxdcd.w('const DT_UL n_o_2_threshs,')

    if obj_3_flag:
        pxdcd.w('DT_D_NP_t[:, :] ppt_cp_mean_arr,')
        pxdcd.w('const DT_D_NP_t[:] ppt_mean_arr,')

    if obj_4_flag:
        pxdcd.w('const DT_D_NP_t[:, :] in_wet_arr_calib,')
        pxdcd.w('const DT_D_NP_t[:, :] ppt_mean_wet_arr,')
        pxdcd.w('const DT_D_NP_t[:] o_4_p_thresh_arr,')
        pxdcd.w('DT_UL_NP_t[:, :, :] ppt_cp_mean_wet_arr,')
        pxdcd.w('DT_D_NP_t[:, :] nebs_wet_obj_vals_arr,')
        pxdcd.w('const DT_UL n_nebs,')
        pxdcd.w('const DT_UL n_o_4_threshs,')

    if obj_5_flag:
        pxdcd.w('DT_D_NP_t[:, :] cats_ppt_cp_mean_arr,')
        pxdcd.w('const DT_D_NP_t[:] cats_ppt_mean_arr,')

    pxdcd.w('DT_D_NP_t[:] ppt_cp_n_vals_arr,')
    pxdcd.w('const DT_D_NP_t[:] obj_ftn_wts_arr,')
    pxdcd.w('const DT_UL_NP_t[:] sel_cps,')
    pxdcd.w('const DT_UL_NP_t[:] old_sel_cps,')
    pxdcd.w('const DT_UL_NP_t[:] chnge_steps,')
    pxdcd.w('const DT_UL n_cpus,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_max,')
    pxdcd.w('const DT_UL n_time_steps,')
    pxdcd.w(') nogil')
    pxdcd.els()
    pxdcd.ded()

    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('Py_ssize_t i, j, s')
    pyxcd.w('DT_D _, obj_val = 0.0')
    pyxcd.els()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('Py_ssize_t m')
        pyxcd.w('DT_D curr_ppt')
        pyxcd.els()

    if obj_2_flag or obj_5_flag:
        pyxcd.w('Py_ssize_t q')
        pyxcd.w('DT_D curr_cat_ppt')
        pyxcd.els()

    if obj_1_flag:
        pyxcd.w('Py_ssize_t p')
        pyxcd.w('DT_D o_1 = 0.0')
        pyxcd.w('DT_D curr_ppt_pi_diff')
        pyxcd.els()

    if obj_2_flag:
        pyxcd.w('Py_ssize_t r')
        pyxcd.w('DT_D o_2 = 0.0')
        pyxcd.w('DT_D curr_cat_ppt_pi_diff')
        pyxcd.els()

    if obj_3_flag:
        pyxcd.w('DT_D o_3 = 0.0')
        pyxcd.w('DT_D curr_ppt_diff')
        pyxcd.w('DT_D cp_ppt_mean')
        pyxcd.w('DT_D old_ppt_cp_mean')
        pyxcd.w('DT_D sel_ppt_cp_mean')
        pyxcd.els()

    if obj_4_flag:
        pyxcd.w('Py_ssize_t n, o')
        pyxcd.w('DT_D o_4 = 0.0')
        pyxcd.w('DT_D curr_ppt_wet_diff')
        pyxcd.els()

    if obj_5_flag:
        pyxcd.w('DT_D o_5 = 0.0')
        pyxcd.w('DT_D curr_cat_ppt_diff')
        pyxcd.w('DT_D cp_cat_ppt_mean')
        pyxcd.w('DT_D old_cat_ppt_cp_mean')
        pyxcd.w('DT_D sel_cat_ppt_cp_mean')
        pyxcd.els()

    pyxcd.ded()

    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('for i in range(n_time_steps):')
    pyxcd.ind()
    pyxcd.w('if not chnge_steps[i]:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()
    pyxcd.w('if old_sel_cps[i] == j:')
    pyxcd.ind()
    pyxcd.w('ppt_cp_n_vals_arr[j] -= 1')
    pyxcd.ded()

    pyxcd.w('if sel_cps[i] == j:')
    pyxcd.ind()
    pyxcd.w('ppt_cp_n_vals_arr[j] += 1')
    pyxcd.ded(lev=3)

    pyxcd.w(
        'for s in prange(n_max, schedule=\'dynamic\', nogil=True, num_threads=n_cpus):')

    if obj_1_flag or obj_3_flag:
        pyxcd.ind()
        pyxcd.w('if s < n_stns:')
        pyxcd.ind()
        pyxcd.w('m = s')
        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('stns_obj_1_vals_arr[m, p] = 0.0')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('curr_ppt_diff = 0.0')
            pyxcd.els()

        pyxcd.w('# remove the effect of the previous CP')
        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()

        if obj_3_flag:
            pyxcd.w('old_ppt_cp_mean = 0.0')
            pyxcd.w('sel_ppt_cp_mean = 0.0')
            pyxcd.els()

        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('if not chnge_steps[i]:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('curr_ppt = in_ppt_arr[i, m]')
        pyxcd.els()

        pyxcd.w('if old_sel_cps[i] == j:')
        pyxcd.ind()

        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('if curr_ppt < o_1_ppt_thresh_arr[p]:')
            pyxcd.ind()
            pyxcd.w('break')
            pyxcd.ded()
            pyxcd.w(
                'ppt_cp_mean_pis_arr[m, j, p] = ppt_cp_mean_pis_arr[m, j, p] - 1')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('old_ppt_cp_mean = old_ppt_cp_mean + curr_ppt')

        pyxcd.ded()

        pyxcd.w('if sel_cps[i] == j:')
        pyxcd.ind()
        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('if curr_ppt < o_1_ppt_thresh_arr[p]:')
            pyxcd.ind()
            pyxcd.w('break')
            pyxcd.ded()
            pyxcd.w(
                'ppt_cp_mean_pis_arr[m, j, p] = ppt_cp_mean_pis_arr[m, j, p] + 1')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('sel_ppt_cp_mean = sel_ppt_cp_mean + curr_ppt')

        pyxcd.ded(lev=2)

        if obj_3_flag:
            pyxcd.w(
                'ppt_cp_mean_arr[j, m] = ppt_cp_mean_arr[j, m] - old_ppt_cp_mean + sel_ppt_cp_mean')

        pyxcd.ded()

        pyxcd.w('# incorporate the effect of the new CP')
        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('if ppt_cp_n_vals_arr[j] == 0:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('stns_obj_1_vals_arr[m, p] = stns_obj_1_vals_arr[m, p] + '
                    'ppt_cp_n_vals_arr[j] * ((ppt_cp_mean_pis_arr[m, j, p] / '
                    'ppt_cp_n_vals_arr[j]) - ppt_mean_pis_arr[m, p])**2')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('if ppt_mean_arr[m] > min_abs_ppt_thresh:')
            pyxcd.ind()
            pyxcd.w(
                '_ = (ppt_cp_mean_arr[j, m] / ppt_cp_n_vals_arr[j]) / ppt_mean_arr[m]')

            pyxcd.w('if _ <= 0:')
            pyxcd.ind()
            pyxcd.w('_ = 1e-100')
            pyxcd.ded()
            pyxcd.w(
                'curr_ppt_diff = curr_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(log(_)))')
            pyxcd.ded(lev=2)
            pyxcd.w('o_3 += (curr_ppt_diff / n_time_steps)')
        else:
            pyxcd.ded()

        pyxcd.ded(lev=2)

    if obj_2_flag or obj_5_flag:
        pyxcd.ind()
        pyxcd.w('if s < n_cats:')
        pyxcd.ind()
        pyxcd.w('q = s')
#         pyxcd.w('for q in prange(n_cats, schedule=\'dynamic\'):')
#         pyxcd.ind()
        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('cats_obj_2_vals_arr[q, r] = 0.0')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('curr_cat_ppt_diff = 0.0')
            pyxcd.els()

        pyxcd.w('# remove the effect of the previous CP')
        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()

        if obj_5_flag:
            pyxcd.w('old_cat_ppt_cp_mean = 0.0')
            pyxcd.w('sel_cat_ppt_cp_mean = 0.0')
            pyxcd.els()

        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('if not chnge_steps[i]:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('curr_cat_ppt = in_cats_ppt_arr[i, q]')
        pyxcd.els()

        pyxcd.w('if old_sel_cps[i] == j:')
        pyxcd.ind()

        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('if curr_cat_ppt < o_2_ppt_thresh_arr[r]:')
            pyxcd.ind()
            pyxcd.w('break')
            pyxcd.ded()
            pyxcd.w(
                'cats_ppt_cp_mean_pis_arr[q, j, r] = cats_ppt_cp_mean_pis_arr[q, j, r] - 1')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('old_cat_ppt_cp_mean = old_cat_ppt_cp_mean + curr_cat_ppt')

        pyxcd.ded()

        pyxcd.w('if sel_cps[i] == j:')
        pyxcd.ind()
        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('if curr_cat_ppt < o_2_ppt_thresh_arr[r]:')
            pyxcd.ind()
            pyxcd.w('break')
            pyxcd.ded()
            pyxcd.w(
                'cats_ppt_cp_mean_pis_arr[q, j, r] = cats_ppt_cp_mean_pis_arr[q, j, r] + 1')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('sel_cat_ppt_cp_mean = sel_cat_ppt_cp_mean + curr_cat_ppt')

        pyxcd.ded(lev=2)

        if obj_5_flag:
            pyxcd.w('cats_ppt_cp_mean_arr[j, q] = cats_ppt_cp_mean_arr[j, q] - '
                    'old_cat_ppt_cp_mean + sel_cat_ppt_cp_mean')

        pyxcd.ded()

        pyxcd.w('# incorporate the effect of the new CP')
        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('if ppt_cp_n_vals_arr[j] == 0:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('cats_obj_2_vals_arr[q, r] = cats_obj_2_vals_arr[q, r] + '
                    'ppt_cp_n_vals_arr[j] * ((cats_ppt_cp_mean_pis_arr[q, j, r] / '
                    'ppt_cp_n_vals_arr[j]) - cats_ppt_mean_pis_arr[q, r])**2')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('if cats_ppt_mean_arr[q] > min_abs_ppt_thresh:')
            pyxcd.ind()
            pyxcd.w(
                '_ = (cats_ppt_cp_mean_arr[j, q] / ppt_cp_n_vals_arr[j]) / cats_ppt_mean_arr[q]')

            pyxcd.w('if _ <= 0:')
            pyxcd.ind()
            pyxcd.w('_ = 1e-100')
            pyxcd.ded()
            pyxcd.w(
                'curr_cat_ppt_diff = curr_cat_ppt_diff + (ppt_cp_n_vals_arr[j] * abs(log(_)))')
            pyxcd.ded(lev=2)
            pyxcd.w('o_5 += (curr_cat_ppt_diff / n_time_steps)')
        else:
            pyxcd.ded()

        pyxcd.ded(lev=2)

    if obj_4_flag:
        pyxcd.ind()
        pyxcd.w('if s < n_nebs:')
        pyxcd.ind()
        pyxcd.w('n = s')
#         pyxcd.w('for n in prange(n_nebs, schedule=\'dynamic\'):')
#         pyxcd.ind()
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('nebs_wet_obj_vals_arr[n, o] = 0.0')
        pyxcd.ded()

        pyxcd.w('for j in range(n_cps):')
        pyxcd.ind()
        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('if not chnge_steps[i]:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('if old_sel_cps[i] == j:')
        pyxcd.ind()
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('if in_wet_arr_calib[i, n] < o_4_p_thresh_arr[o]:')
        pyxcd.ind()
        pyxcd.w('break')
        pyxcd.ded()

        pyxcd.w(
            'ppt_cp_mean_wet_arr[n, j, o] = ppt_cp_mean_wet_arr[n, j, o] - 1')
        pyxcd.ded(lev=2)

        pyxcd.w('if sel_cps[i] == j:')
        pyxcd.ind()
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('if in_wet_arr_calib[i, n] < o_4_p_thresh_arr[o]:')
        pyxcd.ind()
        pyxcd.w('break')
        pyxcd.ded()

        pyxcd.w(
            'ppt_cp_mean_wet_arr[n, j, o] = ppt_cp_mean_wet_arr[n, j, o] + 1')
        pyxcd.ded(lev=3)

        pyxcd.w('if not ppt_cp_n_vals_arr[j]:')
        pyxcd.ind()
        pyxcd.w('continue')
        pyxcd.ded()

        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('nebs_wet_obj_vals_arr[n, o] = nebs_wet_obj_vals_arr[n, o] + '
                'ppt_cp_n_vals_arr[j] * ((ppt_cp_mean_wet_arr[n, j, o] / '
                'ppt_cp_n_vals_arr[j]) - ppt_mean_wet_arr[n, o])**2')
        pyxcd.ded(lev=4)

    if obj_1_flag:
        pyxcd.w('for p in range(n_o_1_threshs):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_pi_diff = 0.0')
        pyxcd.w('for m in range(n_stns):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_pi_diff += stns_obj_1_vals_arr[m, p]')
        pyxcd.ded()
        pyxcd.w('o_1 += (curr_ppt_pi_diff / n_time_steps)**0.5')
        pyxcd.ded()

    if obj_2_flag:
        pyxcd.w('for r in range(n_o_2_threshs):')
        pyxcd.ind()
        pyxcd.w('curr_cat_ppt_pi_diff = 0.0')
        pyxcd.w('for q in range(n_cats):')
        pyxcd.ind()
        pyxcd.w('curr_cat_ppt_pi_diff += cats_obj_2_vals_arr[q, r]')
        pyxcd.ded()
        pyxcd.w('o_2 += (curr_cat_ppt_pi_diff / n_time_steps)**0.5')
        pyxcd.ded()

    if obj_4_flag:
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_wet_diff = 0.0')
        pyxcd.w('for n in range(n_nebs):')
        pyxcd.ind()
        pyxcd.w('curr_ppt_wet_diff += nebs_wet_obj_vals_arr[n, o]')
        pyxcd.ded()
        pyxcd.w('o_4 += (curr_ppt_wet_diff / n_time_steps)**0.5')
        pyxcd.ded()

    if obj_1_flag:
        pyxcd.w('obj_val += (o_1 * obj_ftn_wts_arr[0])')
    if obj_2_flag:
        pyxcd.w('obj_val += (o_2 * obj_ftn_wts_arr[1])')
    if obj_3_flag:
        pyxcd.w('obj_val += (o_3 * obj_ftn_wts_arr[2])')
    if obj_4_flag:
        pyxcd.w('obj_val += (o_4 * obj_ftn_wts_arr[3])')
    if obj_5_flag:
        pyxcd.w('obj_val += (o_5 * obj_ftn_wts_arr[4])')

    pyxcd.w('return obj_val')
    pyxcd.ded(False)

    #==========================================================================
    # write the pyxbld
    #==========================================================================

    write_pyxbld(pyxbldcd)

    #==========================================================================
    # Save as pyx, pxd, pyxbld
    #==========================================================================
#     assert pyxcd.level == 0, \
#         'Level should be zero instead of %d' % pyxcd.level
    assert pxdcd.level == 0, \
        'Level should be zero instead of %d' % pxdcd.level
    assert pyxbldcd.level == 0, \
        'Level should be zero instead of %d' % pyxbldcd.level

    out_path = os.path.join(out_dir, module_name)
    pyxcd.stf(out_path + '.pyx')
    pxdcd.stf(out_path + '.pxd')
    pyxbldcd.stf(out_path + '.pyxbld')

#     #==========================================================================
#     # Check for syntax errors
#     #==========================================================================
#     abs_path = os.path.abspath(out_path + '.pyx')
#     arg = (cython, "%s -a" % abs_path)
#     subprocess.call([arg])

    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    tab = '    '
    nonecheck = False
    boundscheck = False
    wraparound = False
    cdivision = True
    language_level = 3
    infer_types = None
    out_dir = os.getcwd()

    obj_1_flag = True
    obj_2_flag = True
    obj_3_flag = True
    obj_4_flag = True
    obj_5_flag = True

#     obj_1_flag = False
#     obj_2_flag = False
#     obj_3_flag = False
#     obj_4_flag = False
#     obj_5_flag = False

    os.chdir(main_dir)

    assert any([obj_1_flag, obj_2_flag, obj_3_flag, obj_4_flag, obj_5_flag])

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    params_dict['obj_1_flag'] = obj_1_flag
    params_dict['obj_2_flag'] = obj_2_flag
    params_dict['obj_3_flag'] = obj_3_flag
    params_dict['obj_4_flag'] = obj_4_flag
    params_dict['obj_5_flag'] = obj_5_flag

    write_obj_ftns_lines(params_dict)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
