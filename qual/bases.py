'''
Created on Jan 2, 2018

@author: Faizan-Uni
'''
import numpy as np

from ..misc.checks import check_nans_finite
from ..alg_dtypes import DT_D_NP, DT_UL_NP


class QualBases:

    def __init__(self, msgs=True):
        assert isinstance(msgs, (int, bool))

        self.msgs = msgs

        self._ppt_set_flag = False
        self._cps_set_flag = False
        return

    def set_ppt_arr(self, ppt_arr):
        assert isinstance(ppt_arr, np.ndarray)
        assert check_nans_finite(ppt_arr)
        assert len(ppt_arr.shape) == 2
        assert np.all(ppt_arr >= 0)

        self.ppt_arr = np.array(ppt_arr, dtype=DT_D_NP, order='C')
        self.n_ppt_cols = ppt_arr.shape[1]
        self.n_time_steps = ppt_arr.shape[0]

        assert self.n_ppt_cols > 0
        assert self.n_time_steps > 0

        self._ppt_set_flag = True
        return
    
    def set_cps_arr(self, sel_cps_arr, n_cps):
        assert isinstance(n_cps, int)
        assert n_cps > 0

        assert isinstance(sel_cps_arr, np.ndarray)
        assert check_nans_finite(sel_cps_arr)
        assert len(sel_cps_arr.shape) == 1

        assert np.unique(sel_cps_arr).shape[0] >= n_cps

        self.sel_cps_arr = np.array(sel_cps_arr, dtype=DT_UL_NP, order='C')
        self.n_cps = n_cps

        self.cp_cts_arr = np.zeros(self.n_cps, dtype=DT_UL_NP, order='C')

        for j in range(self.n_cps):
            curr_cp_idxs = self.sel_cps_arr == j
            self.cp_cts_arr[j] = np.sum(curr_cp_idxs)

        self._cps_set_flag = True
        return

    def _verify_input(self):
        assert self._ppt_set_flag
        assert self._cps_set_flag
        assert self.n_time_steps == self.sel_cps_arr.shape[0]
        return
