'''
Created on Jan 2, 2018

@author: Faizan-Uni
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

from .bases import QualBases
from ..alg_dtypes import DT_D_NP
from ..misc.checks import check_nans_finite


class WettnessIndex(QualBases):

    def __init__(self, msgs=True):
        super(WettnessIndex, self).__init__(msgs)

        self.old_new_cp_map_arr = None

        self._wettness_cmptd_flag = False
        self._cps_reordered_flag = False
        return
    
    def cmpt_wettness_idx(self):
        self._verify_input()

        self.ppt_mean_arr = self.ppt_arr.mean(axis=0).copy(order='C')
        assert np.all(self.ppt_mean_arr > 0)

        self.ppt_cp_wett_arr = np.zeros((self.n_cps,
                                         self.n_ppt_cols),
                                        dtype=DT_D_NP,
                                        order='C')

        for j in range(self.n_cps):
            curr_cp_idxs = self.sel_cps_arr == j
            for m in range(self.n_ppt_cols):
                _ = (self.ppt_arr[curr_cp_idxs, m].mean() /
                     self.ppt_mean_arr[m])

                self.ppt_cp_wett_arr[j, m] = _

        self.mean_cp_wett_arr = np.round(self.ppt_cp_wett_arr.mean(axis=1), 5)
        self._wettness_cmptd_flag = True
        return

    def reorder_cp_rules(self, cp_rules):
        if not self._wettness_cmptd_flag:
            self.cmpt_wettness_idx()

        assert isinstance(cp_rules, np.ndarray)
        assert len(cp_rules.shape) == 2
        assert cp_rules.shape[0] == self.n_cps
        assert check_nans_finite(cp_rules)

        sorted_idxs = np.argsort(self.mean_cp_wett_arr)

        self.cp_rules_sorted = cp_rules[sorted_idxs, :]
        self.mean_cp_wett_sorted_arr = self.mean_cp_wett_arr[sorted_idxs]
        self.old_new_cp_map_arr = np.zeros((self.n_cps, 2), dtype=np.int64)

        for i in range(self.n_cps):
            self.old_new_cp_map_arr[i, :] = sorted_idxs[i], i

        if self.msgs:
            print('\n\nSorted wettness sequence:')
            print('Old CP, New CP, Wettness')
            for i in range(self.n_cps):
                print('%6s, %6s, %12f' % (sorted_idxs[i],
                                          i,
                                          self.mean_cp_wett_sorted_arr[i]))
            print('\n\n')

        self._cps_reordered_flag = True
        return

    def plot_wettness(self,
                      label,
                      out_fig_path,
                      obj_val=None,
                      fig_size=(15, 10)):

        assert self._wettness_cmptd_flag or self._cps_reordered_flag

        if self._cps_reordered_flag:
            plot_wett_arr = self.mean_cp_wett_sorted_arr
        else:
            plot_wett_arr = self.mean_cp_wett_arr

        assert isinstance(label, str)
        assert isinstance(out_fig_path, (str, Path))

        out_fig_path = Path(out_fig_path)
        assert out_fig_path.parents[0].exists()

        if obj_val is not None:
            assert isinstance(obj_val, float)

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert fig_size[0] > 0
        assert fig_size[1] > 0
        
        if self.msgs:
            print('Plotting Wettness index...')

        plt.figure(figsize=fig_size)

        plt.bar(range(self.n_cps), plot_wett_arr)

        plt.xticks(range(self.n_cps), range(self.n_cps))

        title = ''
        title += 'Wettness index for the classification: %s' % label

        if obj_val is not None:
            title += '\n(obj_val: %0.2f)' % obj_val

        plt.title(title)

        plt.xlabel('CP')
        plt.ylabel('Wettness Index')

        plt.grid()

        plt.savefig(str(out_fig_path), bbox_inches='tight')
        plt.close()
        return
