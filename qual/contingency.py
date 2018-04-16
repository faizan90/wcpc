'''
Created on Jan 8, 2018

@author: Faizan-Uni
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

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
                        n_cps_1,
                        n_cps_2,
                        no_cp_val_1,
                        no_cp_val_2,
                        miss_day_val_1,
                        miss_day_val_2):

        assert isinstance(sel_cps_1_arr, np.ndarray)
        assert np.issubdtype(sel_cps_1_arr.dtype, np.integer)
        assert check_nans_finite(sel_cps_1_arr)
        assert sel_cps_1_arr.ndim == 1
        assert sel_cps_1_arr.shape[0]

        assert isinstance(sel_cps_2_arr, np.ndarray)
        assert np.issubdtype(sel_cps_2_arr.dtype, np.integer)
        assert check_nans_finite(sel_cps_2_arr)
        assert sel_cps_2_arr.ndim == 1
        assert sel_cps_2_arr.shape[0]

        assert sel_cps_1_arr.shape[0] == sel_cps_2_arr.shape[0]

        assert isinstance(n_cps_1, int)
        assert n_cps_1 > 0

        assert isinstance(no_cp_val_1, int)
        assert no_cp_val_1 > n_cps_1

        assert isinstance(miss_day_val_1, int)
        assert miss_day_val_1 > n_cps_1
        assert miss_day_val_1 != no_cp_val_1

        assert isinstance(n_cps_2, int)
        assert n_cps_2 > 0

        assert isinstance(no_cp_val_2, int)
        assert no_cp_val_2 > n_cps_2

        assert isinstance(miss_day_val_2, int)
        assert miss_day_val_2 > n_cps_2
        assert miss_day_val_2 != no_cp_val_2

        self.sel_cps_1_arr = np.array(sel_cps_1_arr, dtype=DT_UL_NP, order='C')
        self.sel_cps_2_arr = np.array(sel_cps_2_arr, dtype=DT_UL_NP, order='C')

        self.n_cps_1 = n_cps_1
        self.no_cp_val_1 = no_cp_val_1
        self.miss_day_val_1 = miss_day_val_1

        self.n_cps_2 = n_cps_2
        self.no_cp_val_2 = no_cp_val_2
        self.miss_day_val_2 = miss_day_val_2

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

        for uni_val in self.unique_sel_cps_1_arr:
            cond_1 = (0 <= uni_val < self.n_cps_1)
            cond_2 = uni_val == self.no_cp_val_1
            cond_3 = uni_val == self.miss_day_val_1

            assert any([cond_1, cond_2, cond_3]), 'Unknown value: %d' % uni_val

        for uni_val in self.unique_sel_cps_2_arr:
            cond_1 = (0 <= uni_val < self.n_cps_2)
            cond_2 = uni_val == self.no_cp_val_2
            cond_3 = uni_val == self.miss_day_val_2

            assert any([cond_1, cond_2, cond_3]), 'Unknown value: %d' % uni_val

        _ = (self.unique_sel_cps_1_arr.shape[0], 
             self.unique_sel_cps_2_arr.shape[0])

        self.cont_table_1_arr = np.full(shape=_,
                                        fill_value=np.nan,
                                        dtype=DT_D_NP)
        self.cont_table_2_arr = np.full(shape=_[::-1],
                                        fill_value=np.nan,
                                        dtype=DT_D_NP)
        
        self.cont_table_nvals_1_arr = self.cont_table_1_arr.astype(np.int64)
        self.cont_table_nvals_2_arr = self.cont_table_2_arr.astype(np.int64)

        self.cont_table_nvals_1_arr[:, :] = 0
        self.cont_table_nvals_2_arr[:, :] = 0

        self.cont_table_str_1_arr = np.full(shape=_,
                                            fill_value=' NaN ',
                                            dtype='|U5')
        self.cont_table_str_2_arr = np.full(shape=_[::-1],
                                            fill_value=' NaN ',
                                            dtype='|U5')

        self.stats_dict_1 = {}
        self.stats_dict_2 = {}

        self._cmpt_table(self.sel_cps_1_arr,
                         self.unique_sel_cps_1_arr,
                         self.sel_cps_2_arr,
                         self.unique_sel_cps_2_arr,
                         self.cont_table_1_arr,
                         self.cont_table_str_1_arr,
                         self.cont_table_nvals_1_arr)

        self._cmpt_table(self.sel_cps_2_arr,
                         self.unique_sel_cps_2_arr,
                         self.sel_cps_1_arr,
                         self.unique_sel_cps_1_arr,
                         self.cont_table_2_arr,
                         self.cont_table_str_2_arr,
                         self.cont_table_nvals_2_arr)

        self._table_cmptd_flag = True
        return
    
    @staticmethod
    def _cmpt_stats(cont_tab_nvals_arr, stats_dict):
        '''
        After Stehlik & Bardossy 2003
        '''

        chi_sq = 0
        n = np.nansum(cont_tab_nvals_arr)
        assert n

        min_rs = min(cont_tab_nvals_arr.shape)
        assert min_rs > 1
        
        for i in range(cont_tab_nvals_arr.shape[0]):
            n_j = np.nansum(cont_tab_nvals_arr[i])
            for j in range(cont_tab_nvals_arr.shape[1]):
                n_ij = cont_tab_nvals_arr[i, j]
                n_i = np.nansum(cont_tab_nvals_arr[:, j])

                _ = ((n_i * n_j) / n)

                chi_sq += ((n_ij - _) ** 2) / _

        corr_pear_coeff = (((min_rs / (min_rs - 1)) ** 0.5) *
                           ((chi_sq / (chi_sq + n)) ** 0.5))
        cram_coeff = (chi_sq / (n * (min_rs - 1))) ** 0.5

        etr = n - max(np.sum(cont_tab_nvals_arr, axis=1))
        ec = np.sum(np.sum(cont_tab_nvals_arr, axis=0) -
                    np.max(cont_tab_nvals_arr, axis=0))
        lam = (etr - ec) / etr

        stats_dict['chi_sq'] = round(chi_sq, 3)
        stats_dict['corr_pear_coeff'] = round(corr_pear_coeff, 3)
        stats_dict['cram_coeff'] = round(cram_coeff, 3)
        stats_dict['lam'] = round(lam, 3)
        return

    @staticmethod
    def _cmpt_table(sel_cps_1_arr,
                    uni_sel_cps_1_arr,
                    sel_cps_2_arr,
                    uni_sel_cps_2_arr,
                    cont_tab_val_arr,
                    cont_tab_str_arr,
                    cont_tab_nvals_arr):

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

                cont_tab_nvals_arr[i, cp_val_2_col_no] = n_cp_val_2

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
        assert fig_size[0] and fig_size[1]

        _ = [[self.cont_table_1_arr, self.cont_table_str_1_arr],
             [self.cont_table_2_arr, self.cont_table_str_2_arr]]

        self._cmpt_stats(self.cont_table_nvals_1_arr, self.stats_dict_1)
        self._cmpt_stats(self.cont_table_nvals_2_arr, self.stats_dict_2)

        _suffs = [[lab_1,
                   lab_2,
                   '%s_%s' % (lab_1, lab_2),
                   self.unique_sel_cps_1_arr,
                   self.unique_sel_cps_2_arr,
                   self.stats_dict_1],
                  [lab_2,
                   lab_1,
                   '%s_%s' % (lab_2, lab_1),
                   self.unique_sel_cps_2_arr,
                   self.unique_sel_cps_1_arr,
                   self.stats_dict_2]]

        for i, (tab, strs) in enumerate(_):
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            cax = ax.imshow(tab, vmin=0, vmax=1, cmap=plt.get_cmap('Blues'))
            cbar = fig.colorbar(cax, orientation='vertical')
            cbar.set_label(('Relative frequency of classification '
                            '%s w.r.t %s') %
                           (_suffs[i][1], _suffs[i][0]))

            _titl_str = ''
            _titl_str += 'CP Classifications Contingency Table'
            _titl_str += '\n$\chi^2$: %s, ' % str(_suffs[i][5]['chi_sq'])
            _titl_str += ('Corrected Pearson Coeff.: %s, ' %
                          str(_suffs[i][5]['corr_pear_coeff']))
            _titl_str += ('Cramer Coeff.: %s, ' %
                          str(_suffs[i][5]['cram_coeff']))
            _titl_str += ('Lambda.: %s' %
                          str(_suffs[i][5]['lam']))

            ax.set_title(_titl_str)

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

