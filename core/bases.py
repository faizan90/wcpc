'''
Created on Dec 31, 2017

@author: Faizan
'''
'''
Created on Dec 30, 2017

@author: Faizan
'''

import numpy as np

from ..misc.checks import check_nans_finite
from ..alg_dtypes import DT_D_NP, DT_UL_NP


class CPDataBase:
    def __init__(self, msgs=True):
        assert isinstance(msgs, (bool, int))
        self.msgs = bool(msgs)

        self._stn_ppt_set_flag = False
        self._cat_ppt_set_flag = False
        self._neb_wett_set_flag = False
        self._lorenz_set_flag = False

        self._cp_prms_set_flag = False

        self.obj_1_flag = False
        self.obj_2_flag = False
        self.obj_3_flag = False
        self.obj_4_flag = False
        self.obj_5_flag = False
        self.obj_6_flag = False
        self.obj_7_flag = False
        self.obj_8_flag = False
        self._n_obj_ftns = 8

        self.cyth_nonecheck = False
        self.cyth_boundscheck = False
        self.cyth_wraparound = False
        self.cyth_cdivision = True
        self.cyth_language_level = 3
        self.cyth_infer_types = None

        self.op_mp_memb_flag = True
        self.op_mp_obj_ftn_flag = True
        self.no_steep_anom_flag = True

        self.min_abs_ppt_thresh = 0.0
        return

    def set_stn_ppt(self, stn_ppt_arr):
        assert isinstance(stn_ppt_arr, np.ndarray)
        assert check_nans_finite(stn_ppt_arr)
        assert stn_ppt_arr.ndim == 2
        assert np.all(stn_ppt_arr >= 0)
        assert stn_ppt_arr.shape[0] and stn_ppt_arr.shape[1]

        self.stn_ppt_arr = np.array(stn_ppt_arr, dtype=DT_D_NP, order='C')
        self._stn_ppt_set_flag = True
        return

    def set_cat_ppt(self, cat_ppt_arr):
        assert isinstance(cat_ppt_arr, np.ndarray)
        assert check_nans_finite(cat_ppt_arr)
        assert cat_ppt_arr.ndim == 2
        assert np.all(cat_ppt_arr >= 0)
        assert cat_ppt_arr.shape[0] and cat_ppt_arr.shape[1]

        self.cat_ppt_arr = np.array(cat_ppt_arr, dtype=DT_D_NP, order='C')
        self._cat_ppt_set_flag = True
        return

    def set_neb_wett(self, neb_wett_arr):
        assert isinstance(neb_wett_arr, np.ndarray)
        assert check_nans_finite(neb_wett_arr)
        assert neb_wett_arr.ndim == 2
        assert np.all(neb_wett_arr >= 0)
        assert neb_wett_arr.shape[0] and neb_wett_arr.shape[1]

        self.neb_wett_arr = np.array(neb_wett_arr, dtype=DT_D_NP, order='C')
        self._neb_wett_set_flag = True
        return

    def set_lorenz_arr(self, lorenz_arr):
        assert isinstance(lorenz_arr, np.ndarray)
        assert check_nans_finite(lorenz_arr)
        assert lorenz_arr.ndim == 2
        assert lorenz_arr.shape[0] and lorenz_arr.shape[1]

        self.lorenz_arr = np.array(lorenz_arr, dtype=DT_UL_NP, order='C')
        self._lorenz_set_flag = True
        return

    def _verify_cp_prms(self):
        assert isinstance(self.n_cps, int)
        assert isinstance(self.max_idx_ct, int)
        assert isinstance(self.no_cp_val, int)
        assert isinstance(self.miss_cp_val, int)
        assert isinstance(self.p_l, float)
        assert isinstance(self.fuzz_nos_arr, np.ndarray)

        assert self.n_cps > 0
        assert self.max_idx_ct > 0
        assert self.p_l > 0

        assert check_nans_finite(self.fuzz_nos_arr)
        assert self.fuzz_nos_arr.shape[0]
        assert self.fuzz_nos_arr.shape[1] == 3
        assert np.all(np.ediff1d(self.fuzz_nos_arr[:, 1]) > 0)

        assert isinstance(self.lo_freq_pen_wt, (int, float))
        assert (self.lo_freq_pen_wt >= 0) and (self.lo_freq_pen_wt < np.inf)

        assert isinstance(self.min_freq, float)
        assert 0 <= self.min_freq < (1 / self.n_cps)
        return

    def set_cp_prms(self,
                    n_cps,
                    max_idx_ct,
                    no_cp_val,
                    miss_cp_val,
                    p_l,
                    fuzz_nos_arr,
                    lo_freq_pen_wt,
                    min_freq):

        self.n_cps = n_cps
        self.max_idx_ct = max_idx_ct
        self.no_cp_val = no_cp_val
        self.miss_cp_val = miss_cp_val
        self.p_l = p_l
        self.fuzz_nos_arr = fuzz_nos_arr
        self.lo_freq_pen_wt = lo_freq_pen_wt
        self.min_freq = min_freq

        self._verify_cp_prms()

        self.fuzz_nos_arr = np.array(fuzz_nos_arr, dtype=DT_D_NP, order='C')
        self._cp_prms_set_flag = True
        return

    def set_obj_1_on(self, o_1_ppt_thresh_arr, o_1_obj_wt):

        assert self._stn_ppt_set_flag

        assert isinstance(o_1_ppt_thresh_arr, np.ndarray)
        assert o_1_ppt_thresh_arr.ndim == 1
        assert o_1_ppt_thresh_arr.shape[0]
        assert check_nans_finite(o_1_ppt_thresh_arr)
        assert np.all(np.ediff1d(o_1_ppt_thresh_arr) > 0)
        assert np.all(o_1_ppt_thresh_arr)

        assert isinstance(o_1_obj_wt, (int, float))
        assert check_nans_finite(o_1_obj_wt)
        assert o_1_obj_wt > 0

        self.o_1_ppt_thresh_arr = np.array(o_1_ppt_thresh_arr,
                                           dtype=DT_D_NP,
                                           order='C')
        self.o_1_obj_wt = o_1_obj_wt
        self.obj_1_flag = True
        return

    def set_obj_1_off(self):
        if hasattr(self, 'o_1_ppt_thresh_arr'):
            del self.o_1_ppt_thresh_arr

        if hasattr(self, 'o_1_obj_wt'):
            del self.o_1_obj_wt

        self.obj_1_flag = False
        return

    def set_obj_2_on(self, o_2_ppt_thresh_arr, o_2_obj_wt):
        assert self._cat_ppt_set_flag

        assert isinstance(o_2_ppt_thresh_arr, np.ndarray)
        assert o_2_ppt_thresh_arr.ndim == 1
        assert o_2_ppt_thresh_arr.shape[0]
        assert check_nans_finite(o_2_ppt_thresh_arr)
        assert np.all(np.ediff1d(o_2_ppt_thresh_arr) > 0)
        assert np.all(o_2_ppt_thresh_arr)

        assert isinstance(o_2_obj_wt, (int, float))
        assert check_nans_finite(o_2_obj_wt)
        assert o_2_obj_wt > 0

        self.o_2_ppt_thresh_arr = np.array(o_2_ppt_thresh_arr,
                                           dtype=DT_D_NP,
                                           order='C')
        self.o_2_obj_wt = o_2_obj_wt
        self.obj_2_flag = True
        return

    def set_obj_2_off(self):
        if hasattr(self, 'o_2_ppt_thresh_arr'):
            del self.o_2_ppt_thresh_arr

        if hasattr(self, 'o_2_obj_wt'):
            del self.o_2_obj_wt

        self.obj_2_flag = False
        return

    def set_obj_3_on(self, o_3_obj_wt):
        assert self._stn_ppt_set_flag

        assert isinstance(o_3_obj_wt, (int, float))
        assert check_nans_finite(o_3_obj_wt)
        assert o_3_obj_wt > 0

        self.o_3_obj_wt = o_3_obj_wt

        self.obj_3_flag = True
        return

    def set_obj_3_off(self):
        if hasattr(self, 'o_3_obj_wt'):
            del self.o_3_obj_wt

        self.obj_3_flag = False
        return

    def set_obj_4_on(self, o_4_wett_thresh_arr, o_4_obj_wt):
        assert self._neb_wett_set_flag

        assert isinstance(o_4_wett_thresh_arr, np.ndarray)
        assert o_4_wett_thresh_arr.ndim == 1
        assert o_4_wett_thresh_arr.shape[0] > 0
        assert check_nans_finite(o_4_wett_thresh_arr)
        assert np.all(np.ediff1d(o_4_wett_thresh_arr) > 0)

        assert isinstance(o_4_obj_wt, (int, float))
        assert check_nans_finite(o_4_obj_wt)
        assert o_4_obj_wt > 0

        self.o_4_wett_thresh_arr = np.array(o_4_wett_thresh_arr,
                                            dtype=DT_D_NP,
                                            order='C')
        self.o_4_obj_wt = o_4_obj_wt
        self.obj_4_flag = True
        return

    def set_obj_4_off(self):
        if hasattr(self, 'o_4_wett_thresh_arr'):
            del self.o_4_wett_thresh_arr

        if hasattr(self, 'o_4_obj_wt'):
            del self.o_4_obj_wt

        self.obj_4_flag = False
        return

    def set_obj_5_on(self, o_5_obj_wt):
        assert self._cat_ppt_set_flag

        assert isinstance(o_5_obj_wt, (int, float))
        assert check_nans_finite(o_5_obj_wt)
        assert o_5_obj_wt > 0

        self.o_5_obj_wt = o_5_obj_wt

        self.obj_5_flag = True
        return

    def set_obj_5_off(self):
        if hasattr(self, 'o_5_obj_wt'):
            del self.o_5_obj_wt

        self.obj_5_flag = False
        return

    def set_obj_6_on(self, o_6_obj_wt, min_wettness_thresh):
        assert self._neb_wett_set_flag

        assert isinstance(o_6_obj_wt, (int, float))
        assert check_nans_finite(o_6_obj_wt)
        assert o_6_obj_wt > 0
        assert isinstance(min_wettness_thresh, float)
        assert 0 <= min_wettness_thresh < 1.0

        self.o_6_obj_wt = o_6_obj_wt
        self.min_wettness_thresh = min_wettness_thresh

        self.obj_6_flag = True
        return

    def set_obj_6_off(self):
        if hasattr(self, 'o_6_obj_wt'):
            del self.o_6_obj_wt

        self.obj_6_flag = False
        return

    def set_obj_7_on(self, o_7_obj_wt):
        assert self._neb_wett_set_flag

        assert isinstance(o_7_obj_wt, (int, float))
        assert check_nans_finite(o_7_obj_wt)
        assert o_7_obj_wt > 0

        self.o_7_obj_wt = o_7_obj_wt
        self.obj_7_flag = True
        return

    def set_obj_7_off(self):
        if hasattr(self, 'o_7_obj_wt'):
            del self.o_7_obj_wt

        self.obj_7_flag = False
        return

    def set_obj_8_on(self, o_8_obj_wt):
        assert self._lorenz_set_flag

        assert isinstance(o_8_obj_wt, (int, float))
        assert check_nans_finite(o_8_obj_wt)
        assert o_8_obj_wt > 0

        self.o_8_obj_wt = o_8_obj_wt
        self.obj_8_flag = True
        return

    def set_obj_8_off(self):
        if hasattr(self, 'o_8_obj_wt'):
            del self.o_8_obj_wt

        self.obj_8_flag = False
        return

    def set_cyth_flags(self,
                       cyth_nonecheck=True,
                       cyth_boundscheck=True,
                       cyth_wraparound=True,
                       cyth_cdivision=False,
                       cyth_language_level=3,
                       cyth_infer_types=None):

        self.cyth_nonecheck = cyth_nonecheck
        self.cyth_boundscheck = cyth_boundscheck
        self.cyth_wraparound = cyth_wraparound
        self.cyth_cdivision = cyth_cdivision
        self.cyth_language_level = cyth_language_level
        self.cyth_infer_types = cyth_infer_types
        return


class CPOPTBase(CPDataBase):
    def __init__(self, msgs=True):
        super().__init__(msgs=msgs)

        self._anom_set_flag = False
        return

    def set_anomaly(self, vals_tot_anom, n_anom_rows, n_anom_cols):
        assert isinstance(vals_tot_anom, np.ndarray)
        assert check_nans_finite(vals_tot_anom)
        assert vals_tot_anom.ndim == 2
        assert vals_tot_anom.shape[0] and vals_tot_anom.shape[1]

        assert isinstance(n_anom_rows, int)
        assert isinstance(n_anom_cols, int)
#         assert (n_anom_rows * n_anom_cols) == vals_tot_anom.shape[1]

        self.vals_tot_anom = np.array(vals_tot_anom, dtype=DT_D_NP, order='C')
        self.n_anom_rows = n_anom_rows
        self.n_anom_cols = n_anom_cols

        self._anom_set_flag = True
        return
