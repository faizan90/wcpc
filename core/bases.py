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
from ..alg_dtypes import DT_D_NP


class CPDataBase:

    def __init__(self, msgs=True):

        assert isinstance(msgs, (bool, int))
        self.msgs = bool(msgs)

        self._stn_ppt_set_flag = False
        self._cat_ppt_set_flag = False
        self._neb_wett_set_flag = False

        self._cp_prms_set_flag = False

        self.obj_1_flag = False
        self.obj_2_flag = False
        self.obj_3_flag = False
        self.obj_4_flag = False
        self.obj_5_flag = False

        self.cyth_nonecheck = False
        self.cyth_boundscheck = False
        self.cyth_wraparound = False
        self.cyth_cdivision = True
        self.cyth_language_level = 3
        self.cyth_infer_types = None

        self._n_obj_ftns = 5
        self.min_abs_ppt_thresh = 0.0
        return

    def set_stn_ppt(self, stn_ppt_arr):

        assert isinstance(stn_ppt_arr, np.ndarray)
        assert check_nans_finite(stn_ppt_arr)
        assert len(stn_ppt_arr.shape) == 2
        assert np.all(stn_ppt_arr >= 0)

        self.stn_ppt_arr = np.array(stn_ppt_arr, dtype=DT_D_NP, order='C')

        self._stn_ppt_set_flag = True
        return

    def set_cat_ppt(self, cat_ppt_arr):

        assert isinstance(cat_ppt_arr, np.ndarray)
        assert check_nans_finite(cat_ppt_arr)
        assert len(cat_ppt_arr.shape) == 2
        assert np.all(cat_ppt_arr >= 0)

        self.cat_ppt_arr = np.array(cat_ppt_arr, dtype=DT_D_NP, order='C')

        self._cat_ppt_set_flag = True
        return

    def set_neb_wett(self, neb_wett_arr):

        assert isinstance(neb_wett_arr, np.ndarray)
        assert check_nans_finite(neb_wett_arr)
        assert len(neb_wett_arr.shape) == 2
        assert np.all(neb_wett_arr >= 0)

        self.neb_wett_arr = np.array(neb_wett_arr, dtype=DT_D_NP, order='C')

        self._neb_wett_set_flag = True
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
        assert self.fuzz_nos_arr.shape[0] > 0
        assert self.fuzz_nos_arr.shape[1] == 3
        return

    def set_cp_prms(self,
                    n_cps,
                    max_idx_ct,
                    no_cp_val,
                    miss_cp_val,
                    p_l,
                    fuzz_nos_arr):

        self.n_cps = n_cps
        self.max_idx_ct = max_idx_ct
        self.no_cp_val = no_cp_val
        self.miss_cp_val = miss_cp_val
        self.p_l = p_l
        self.fuzz_nos_arr = fuzz_nos_arr

        self._verify_cp_prms()

        self.fuzz_nos_arr = np.array(fuzz_nos_arr, dtype=DT_D_NP, order='C')

        self._cp_prms_set_flag = True
        return

    def set_obj_1_on(self, o_1_ppt_thresh_arr, o_1_obj_wt):

        assert self._stn_ppt_set_flag

        assert isinstance(o_1_ppt_thresh_arr, np.ndarray)
        assert len(o_1_ppt_thresh_arr.shape) == 1
        assert o_1_ppt_thresh_arr.shape[0] > 0
        assert check_nans_finite(o_1_ppt_thresh_arr)
        assert np.all(np.ediff1d(o_1_ppt_thresh_arr) > 0)

        assert isinstance(o_1_obj_wt, (int, float))
        assert check_nans_finite(o_1_obj_wt)

        self.o_1_ppt_thresh_arr = np.array(o_1_ppt_thresh_arr,
                                           dtype=DT_D_NP,
                                           order='C')
        self.o_1_obj_wt = o_1_obj_wt

        self.obj_1_flag = True
        return

    def set_obj_1_off(self):

        del self.o_1_ppt_thresh_arr
        del self.o_1_obj_wt

        self.obj_1_flag = False
        return

    def set_obj_2_on(self, o_2_ppt_thresh_arr, o_2_obj_wt):

        assert self._cat_ppt_set_flag

        assert isinstance(o_2_ppt_thresh_arr, np.ndarray)
        assert len(o_2_ppt_thresh_arr.shape) == 1
        assert o_2_ppt_thresh_arr.shape[0] > 0
        assert check_nans_finite(o_2_ppt_thresh_arr)
        assert np.all(np.ediff1d(o_2_ppt_thresh_arr) > 0)

        assert isinstance(o_2_obj_wt, (int, float))
        assert check_nans_finite(o_2_obj_wt)

        self.o_2_ppt_thresh_arr = np.array(o_2_ppt_thresh_arr,
                                           dtype=DT_D_NP,
                                           order='C')
        self.o_2_obj_wt = o_2_obj_wt

        self.obj_2_flag = True
        return

    def set_obj_2_off(self):

        del self.o_2_ppt_thresh_arr
        del self.o_2_obj_wt

        self.obj_2_flag = False
        return

    def set_obj_3_on(self, o_3_obj_wt):

        assert self._stn_ppt_set_flag

        assert isinstance(o_3_obj_wt, (int, float))
        assert check_nans_finite(o_3_obj_wt)

        self.o_3_obj_wt = o_3_obj_wt

        self.obj_3_flag = True
        return

    def set_obj_3_off(self):

        del self.o_3_obj_wt

        self.obj_3_flag = False
        return

    def set_obj_4_on(self, o_4_wett_thresh_arr, o_4_obj_wt):

        assert self._neb_wett_set_flag

        assert isinstance(o_4_wett_thresh_arr, np.ndarray)
        assert len(o_4_wett_thresh_arr.shape) == 1
        assert o_4_wett_thresh_arr.shape[0] > 0
        assert check_nans_finite(o_4_wett_thresh_arr)
        assert np.all(np.ediff1d(o_4_wett_thresh_arr) > 0)

        assert isinstance(o_4_obj_wt, (int, float))
        assert check_nans_finite(o_4_obj_wt)

        self.o_4_wett_thresh_arr = np.array(o_4_wett_thresh_arr,
                                            dtype=DT_D_NP,
                                            order='C')
        self.o_4_obj_wt = o_4_obj_wt

        self.obj_4_flag = True
        return

    def set_obj_4_off(self):

        del self.o_4_wett_thresh_arr
        del self.o_4_obj_wt

        self.obj_4_flag = False
        return

    def set_obj_5_on(self, o_5_obj_wt):

        assert self._cat_ppt_set_flag

        assert isinstance(o_5_obj_wt, (int, float))
        assert check_nans_finite(o_5_obj_wt)

        self.o_5_obj_wt = o_5_obj_wt

        self.obj_5_flag = True
        return

    def set_obj_5_off(self):

        del self.o_5_obj_wt

        self.obj_5_flag = False
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
        super(CPOPTBase, self).__init__(msgs=msgs)

        self._anom_set_flag = False
        return

    def set_anomaly(self, vals_tot_anom):

        assert isinstance(vals_tot_anom, np.ndarray)
        assert check_nans_finite(vals_tot_anom)
        assert len(vals_tot_anom.shape) == 2

        self.vals_tot_anom = np.array(vals_tot_anom, dtype=DT_D_NP, order='C')

        self._anom_set_flag = True
        return
