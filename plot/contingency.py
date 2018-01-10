'''
Created on Jan 8, 2018

@author: Faizan-Uni
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps

from ..misc.checks import check_nans_finite
from ..alg_dtypes import DT_D_NP, DT_UL_NP


class ContingencyTablePlot:
    def __init__(self, msgs=True):
        assert isinstance(msgs, (int, bool))
        self.msgs = msgs

        self._sel_cps_arrs_set_flag = False
        self._table_cmptd_flag = False
        return

    def set_sel_cps_arr(self,
                        sel_cps_1_arr,
                        sel_cps_2_arr,
                        n_cps,
                        no_cp_val,
                        miss_day_val=None):

        assert isinstance(sel_cps_1_arr, np.ndarray)
        assert np.issubdtype(sel_cps_1_arr.dtype, np.integer)
        assert check_nans_finite(sel_cps_1_arr)
        assert len(sel_cps_1_arr.shape) == 1

        assert isinstance(sel_cps_2_arr, np.ndarray)
        assert np.issubdtype(sel_cps_2_arr.dtype, np.integer)
        assert check_nans_finite(sel_cps_2_arr)
        assert len(sel_cps_2_arr.shape) == 1

        assert sel_cps_1_arr.shape[0] == sel_cps_2_arr.shape[0]

        assert isinstance(n_cps, int)
        assert n_cps > 0

        assert isinstance(no_cp_val, int)
        assert no_cp_val > n_cps

        if miss_day_val is not None:
            assert isinstance(miss_day_val, int)
            assert miss_day_val > n_cps
            assert miss_day_val != no_cp_val

        self.sel_cps_1_arr = np.array(sel_cps_1_arr, dtype=DT_UL_NP, order='C')
        self.sel_cps_2_arr = np.array(sel_cps_2_arr, dtype=DT_UL_NP, order='C')

        self.n_cps = n_cps
        self.no_cp_val = no_cp_val
        self.miss_day_val = miss_day_val

        self._sel_cps_arrs_set_flag = True
        return
    
    def cmpt_table(self):
        assert self._sel_cps_arrs_set_flag

        (self.unique_sel_cps_1_arr,
         self.unique_sel_cps_1_cts_arr) = np.unique(self.sel_cps_1_arr,
                                                    return_counts=True)
        assert check_nans_finite(self.unique_sel_cps_1_arr)
        
        (self.unique_sel_cps_2_arr,
         self.unique_sel_cps_2_cts_arr) = np.unique(self.sel_cps_2_arr,
                                                    return_counts=True)
        assert check_nans_finite(self.unique_sel_cps_2_arr)

        self.unique_cp_vals_arr = np.union1d(self.sel_cps_1_arr,
                                             self.sel_cps_2_arr)

        if self.miss_day_val is None:
            _miss_cond = False
        else:
            _miss_cond = True
            
        for uni_val in self.unique_cp_vals_arr:
            cond_1 = (0 <= uni_val < self.n_cps)
            cond_2 = uni_val == self.no_cp_val
            
            if _miss_cond:
                cond_3 = uni_val == self.miss_day_val
            else:
                cond_3 = False

            assert any([cond_1, cond_2, cond_3]), 'Unknown value: %d' % uni_val

        _ = (self.unique_sel_cps_1_arr.shape[0], 
             self.unique_sel_cps_2_arr.shape[0])

        self.cont_table_1_arr = np.full(shape=_,
                                        fill_value=np.nan,
                                        dtype=DT_D_NP)
        self.cont_table_2_arr = np.full(shape=_[::-1],
                                        fill_value=np.nan,
                                        dtype=DT_D_NP)

        self.cont_table_str_1_arr = np.full(shape=_,
                                            fill_value=' NaN ',
                                            dtype='|U5')
        self.cont_table_str_2_arr = np.full(shape=_[::-1],
                                            fill_value=' NaN ',
                                            dtype='|U5')

        self._cmpt_table(self.sel_cps_1_arr,
                         self.unique_sel_cps_1_arr,
                         self.sel_cps_2_arr,
                         self.unique_sel_cps_2_arr,
                         self.cont_table_1_arr,
                         self.cont_table_str_1_arr)

        self._cmpt_table(self.sel_cps_2_arr,
                         self.unique_sel_cps_2_arr,
                         self.sel_cps_1_arr,
                         self.unique_sel_cps_1_arr,
                         self.cont_table_2_arr,
                         self.cont_table_str_2_arr)

        self._table_cmptd_flag = True
        return

    @staticmethod
    def _cmpt_table(sel_cps_1_arr,
                    uni_sel_cps_1_arr,
                    sel_cps_2_arr,
                    uni_sel_cps_2_arr,
                    cont_tab_val_arr,
                    cont_tab_str_arr):

        for i, cp_val_1 in enumerate(uni_sel_cps_1_arr):
            cp_val_1_idxs = sel_cps_1_arr == cp_val_1
            n_cp_val_1 = np.sum(cp_val_1_idxs)

            sel_cps_2_vals = sel_cps_2_arr[cp_val_1_idxs]
            (unique_sel_cps_2_vals,
             unique_sel_cps_2_vals_cts) = np.unique(sel_cps_2_vals,
                                                    return_counts=True)
            
            for cp_val_2, n_cp_val_2 in zip(unique_sel_cps_2_vals,
                                            unique_sel_cps_2_vals_cts):
                cp_val_2_col_no = np.where(uni_sel_cps_2_arr ==
                                           cp_val_2)

                prob_cp_2 = n_cp_val_2 / float(n_cp_val_1)
                prob_str = '%4.3f' % prob_cp_2

                cont_tab_val_arr[i, cp_val_2_col_no] = prob_cp_2
                cont_tab_str_arr[i, cp_val_2_col_no] = prob_str

        return

    def plot_cont_table(self,
                        out_figs_dir,
                        lab_1='1',
                        lab_2='2',
                        fig_size=(15, 15)):

        if not self._table_cmptd_flag:
            self.cmpt_table()

        assert isinstance(out_figs_dir, (Path, str))
        out_figs_dir = Path(out_figs_dir)
        assert out_figs_dir.parents[0].exists()
        if not out_figs_dir.exists():
            out_figs_dir.mkdir()

        assert isinstance(lab_1, str)
        assert isinstance(lab_2, str)

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert fig_size[0] > 0
        assert fig_size[1] > 0

        _ = [[self.cont_table_1_arr, self.cont_table_str_1_arr],
             [self.cont_table_2_arr, self.cont_table_str_2_arr]]

        _suffs = [[lab_1,
                   lab_2,
                   '%s_%s' % (lab_1, lab_2),
                   self.unique_sel_cps_1_arr,
                   self.unique_sel_cps_2_arr],
                  [lab_2,
                   lab_1,
                   '%s_%s' % (lab_2, lab_1),
                   self.unique_sel_cps_1_arr,
                   self.unique_sel_cps_2_arr]]

        for i, (tab, strs) in enumerate(_):
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            cax = ax.imshow(tab, vmin=0, vmax=1, cmap=cmaps.Blues)
            cbar = fig.colorbar(cax, orientation='vertical')
            cbar.set_label(('Relative frequency of classification '
                            '%s w.r.t %s') %
                           (_suffs[i][1], _suffs[i][0]))

            ax.set_title('CP Classifications Contingency Table')

            ax.set_xlabel('Classification ' + _suffs[i][1] + ' (CP No.)')
            ax.set_ylabel('Classification ' + _suffs[i][0] + ' (CP No.)')

            ax.set_xticks(range(_suffs[i][4].shape[0]))
            ax.set_xticklabels(_suffs[i][4])

            ax.set_yticks(range(_suffs[i][3].shape[0]))
            ax.set_yticklabels(_suffs[i][3])

            txt_x_corrs = np.tile(range(_suffs[i][4].shape[0]),
                                  _suffs[i][3].shape[0])

            txt_y_corrs = np.repeat(range(_suffs[i][3].shape[0]),
                                    _suffs[i][4].shape[0])

            for j in range(txt_x_corrs.shape[0]):
                ax.text(txt_x_corrs[j],
                        txt_y_corrs[j],
                        strs[txt_y_corrs[j], txt_x_corrs[j]],
                        va='center',
                        ha='center',)

            out_fig_name = 'rel_cp_freq_%s.png' % _suffs[i][2]
            out_fig_path = out_figs_dir / out_fig_name

            plt.savefig(str(out_fig_path), bbox_inches='tight')
            plt.close()

        return

