'''
Created on Jan 2, 2018

@author: Faizan-Uni
'''
import numpy as np

from .bases import QualBases
from ..misc.checks import check_nans_finite
from ..alg_dtypes import DT_D_NP


class ThreshPPT(QualBases):
    def __init__(self, msgs=True):
        super(ThreshPPT, self).__init__(msgs)

        self._ge_vals_set_flag = False
        self._le_vals_set_flag = False
        return

    def set_ge_vals_arr(self, ge_vals_arr):
        assert isinstance(ge_vals_arr, np.ndarray)
        assert check_nans_finite(ge_vals_arr)
        assert ge_vals_arr.ndim == 1
        assert all(ge_vals_arr.shape)

        self.ge_vals_arr = np.array(ge_vals_arr, dtype=DT_D_NP, order='C')
        self.n_ge_vals = self.ge_vals_arr.shape[0]
        assert self.n_ge_vals

        self._ge_vals_set_flag = True
        return

    def cmpt_ge_qual(self):
        self._verify_input()
        assert self._ge_vals_set_flag

        self.cp_ge_qual_arr = np.zeros((self.n_ppt_cols, self.n_ge_vals),
                                       dtype=DT_D_NP,
                                       order='C')

        self.ppt_ge_pis_arr = self.cp_ge_qual_arr.copy()

        for i in range(self.n_ge_vals):
            for m in range(self.n_ppt_cols):
                self.ppt_ge_pis_arr[m, i] = (np.sum(self.ppt_arr[:, m] >=
                                                   self.ge_vals_arr[i]) /
                                             self.n_time_steps)
                assert ((not np.isnan(self.ppt_ge_pis_arr[m, i])) and
                        (self.ppt_ge_pis_arr[m, i] > 0))

        for i in range(self.n_ge_vals):
            for m in range(self.n_ppt_cols):
                curr_pi_diff = 0.0
                for j in range(self.n_cps):
                    curr_cp_idxs = self.sel_cps_arr == j
                    curr_n_vals = np.sum(curr_cp_idxs)

                    curr_n_pi = np.sum(self.ppt_arr[curr_cp_idxs, m] >=
                                       self.ge_vals_arr[i])

                    curr_pi_diff += (curr_n_vals *
                                     ((curr_n_pi / curr_n_vals) -
                                      self.ppt_ge_pis_arr[m, i]) ** 2)

                self.cp_ge_qual_arr[m, i] = (curr_pi_diff /
                                             self.n_time_steps) ** 0.5
        return

    def set_le_vals_arr(self, le_vals_arr):
        assert isinstance(le_vals_arr, np.ndarray)
        assert check_nans_finite(le_vals_arr)
        assert le_vals_arr.ndim == 1
        assert all(le_vals_arr.shape)

        self.le_vals_arr = np.array(le_vals_arr, dtype=DT_D_NP, order='C')
        self.n_le_vals = self.le_vals_arr.shape[0]
        assert self.n_le_vals

        self._le_vals_set_flag = True
        return

    def cmpt_le_qual(self):
        self._verify_input()
        assert self._le_vals_set_flag

        self.cp_le_qual_arr = np.zeros((self.n_ppt_cols, self.n_le_vals),
                                       dtype=DT_D_NP,
                                       order='C')

        self.ppt_le_pis_arr = self.cp_le_qual_arr.copy()

        for i in range(self.n_le_vals):
            for m in range(self.n_ppt_cols):
                self.ppt_le_pis_arr[m, i] = (np.sum(self.ppt_arr[:, m] <=
                                                   self.ge_vals_arr[i]) /
                                             self.n_time_steps)
                assert ((not np.isnan(self.ppt_le_pis_arr[m, i])) and
                        (self.ppt_le_pis_arr[m, i] > 0))

        for i in range(self.n_le_vals):
            for m in range(self.n_ppt_cols):
                curr_pi_diff = 0.0
                for j in range(self.n_cps):
                    curr_cp_idxs = self.sel_cps_arr == j
                    curr_n_vals = np.sum(curr_cp_idxs)

                    curr_n_pi = np.sum(self.ppt_arr[curr_cp_idxs, m] <=
                                       self.le_vals_arr[i])

                    curr_pi_diff += (curr_n_vals *
                                     ((curr_n_pi / curr_n_vals) -
                                      self.ppt_le_pis_arr[m, i]) ** 2)

                self.cp_le_qual_arr[m, i] = (curr_pi_diff /
                                             self.n_time_steps) ** 0.5
        return

