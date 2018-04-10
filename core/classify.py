'''
Created on Dec 30, 2017

@author: Faizan
'''
import timeit
import importlib
from pathlib import Path
from psutil import cpu_count

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .bases import CPOPTBase
from ..alg_dtypes import DT_D_NP
from ..misc.checks import check_nans_finite
from ..codegenr.gen_all import create_classi_cython_files

plt.ioff()


class CPClassiA(CPOPTBase):

    def __init__(self, msgs=True):
        super(CPClassiA, self).__init__(msgs=msgs)

        self._sim_anneal_prms_set_flag = False
        self._classified_flag = False
        return

    def _verify_sim_anneal_prms(self):
        assert isinstance(self.ini_anneal_temp, float)
        assert isinstance(self.tem_alpha, float)
        assert 0 < self.tem_alpha < 1.0

        assert isinstance(self.tem_chng_iters, int)
        assert isinstance(self.max_iters, int)
        assert isinstance(self.max_iters_wo_chng, int)

        assert self.tem_chng_iters > 0
        assert self.max_iters > 0
        assert self.tem_chng_iters <= self.max_iters_wo_chng <= self.max_iters

        assert isinstance(self.temp_adj_iters, (str, int))
        assert isinstance(self.min_acc_rate, int)
        assert isinstance(self.max_acc_rate, int)
        assert isinstance(self.max_temp_adj_atmps, int)

        if isinstance(self.temp_adj_iters, str):
            assert self.temp_adj_iters == 'auto'
        else:
            assert 0 < self.temp_adj_iters
            
        assert 0 <= self.min_acc_rate < 100
        assert 0 <= self.max_acc_rate <= 100
        assert self.min_acc_rate < self.max_acc_rate
        assert 0 < self.max_temp_adj_atmps < self.max_iters
        return

    def set_sim_anneal_prms(self,
                            ini_anneal_temp,
                            tem_alpha,
                            tem_chng_iters,
                            max_iters,
                            max_iters_wo_chng,
                            temp_adj_iters='auto',
                            min_acc_rate=60,
                            max_acc_rate=80,
                            max_temp_adj_atmps=500):

        self.ini_anneal_temp = ini_anneal_temp
        self.tem_alpha = tem_alpha
        self.tem_chng_iters = tem_chng_iters
        self.max_iters = max_iters
        self.max_iters_wo_chng = max_iters_wo_chng

        self.temp_adj_iters = temp_adj_iters
        self.min_acc_rate = min_acc_rate
        self.max_acc_rate = max_acc_rate
        self.max_temp_adj_atmps = max_temp_adj_atmps

        self._verify_sim_anneal_prms()

        self._sim_anneal_prms_set_flag = True
        return

    def _verify_input(self):
        assert self._anom_set_flag
        assert self._cp_prms_set_flag

        assert any((self.obj_1_flag,
                    self.obj_2_flag,
                    self.obj_3_flag,
                    self.obj_4_flag,
                    self.obj_5_flag,
                    self.obj_6_flag,
                    self.obj_7_flag,
                    self.obj_8_flag))

        self.obj_ftn_wts_arr = np.zeros(self._n_obj_ftns,
                                        dtype=DT_D_NP,
                                        order='C')

        in_max_cols_list = []
        in_lens_list = [self.vals_tot_anom.shape[0]]
        if self.obj_1_flag:
            self.obj_ftn_wts_arr[0] = self.o_1_obj_wt
            in_lens_list.append(self.stn_ppt_arr.shape[0])
            in_max_cols_list.append(self.stn_ppt_arr.shape[1])

        if self.obj_2_flag:
            self.obj_ftn_wts_arr[1] = self.o_2_obj_wt
            in_lens_list.append(self.cat_ppt_arr.shape[0])
            in_max_cols_list.append(self.cat_ppt_arr.shape[1])

        if self.obj_3_flag:
            self.obj_ftn_wts_arr[2] = self.o_3_obj_wt
            in_lens_list.append(self.stn_ppt_arr.shape[0])
            in_max_cols_list.append(self.stn_ppt_arr.shape[1])

        if self.obj_4_flag:
            self.obj_ftn_wts_arr[3] = self.o_4_obj_wt
            in_lens_list.append(self.neb_wett_arr.shape[0])
            in_max_cols_list.append(self.neb_wett_arr.shape[1])

        if self.obj_5_flag:
            self.obj_ftn_wts_arr[4] = self.o_5_obj_wt
            in_lens_list.append(self.cat_ppt_arr.shape[0])
            in_max_cols_list.append(self.cat_ppt_arr.shape[1])

        if self.obj_6_flag:
            self.obj_ftn_wts_arr[5] = self.o_6_obj_wt
            assert self.neb_wett_arr.shape[1] == 2, 'For two nebs right now!'
            in_lens_list.append(self.neb_wett_arr.shape[0])
            in_max_cols_list.append(self.neb_wett_arr.shape[1])

        if self.obj_7_flag:
            self.obj_ftn_wts_arr[6] = self.o_7_obj_wt
            assert self.neb_wett_arr.shape[1] == 3, 'For three nebs right now!'
            in_lens_list.append(self.neb_wett_arr.shape[0])
            in_max_cols_list.append(self.neb_wett_arr.shape[1])

        if self.obj_8_flag:
            self.obj_ftn_wts_arr[7] = self.o_8_obj_wt
            in_lens_list.append(self.lorenz_arr.shape[0])
            in_max_cols_list.append(self.lorenz_arr.shape[1])

        _len = in_lens_list[0]
        for curr_len in in_lens_list[1:]:
            assert curr_len == _len

        assert isinstance(self.op_mp_memb_flag, bool)
        assert isinstance(self.op_mp_obj_ftn_flag, bool)
        
        if max(in_max_cols_list) < (2 * self.n_threads):
            self.op_mp_obj_ftn_flag = False
            if self.msgs:
                print('####op_mp_obj_ftn_flag set to False!')

        return

    def _gen_classi_cyth_mods(self, force_compile=False):
        
        cyth_dir = Path(__file__).parents[1] / 'cyth'

        create_classi_cython_files(self.obj_1_flag,
                                   self.obj_2_flag,
                                   self.obj_3_flag,
                                   self.obj_4_flag,
                                   self.obj_5_flag,
                                   self.obj_6_flag,
                                   self.obj_7_flag,
                                   self.obj_8_flag,
                                   self.cyth_nonecheck,
                                   self.cyth_boundscheck,
                                   self.cyth_wraparound,
                                   self.cyth_cdivision,
                                   self.cyth_infer_types,
                                   self.cyth_language_level,
                                   force_compile,
                                   cyth_dir,
                                   self.op_mp_memb_flag,
                                   self.op_mp_obj_ftn_flag)

#         raise Exception
        importlib.invalidate_caches()

        return importlib.import_module('..cyth.classi_alg',
                                       package='wcpc.core').classify_cps
    
    def classify(self, n_threads='auto', force_compile=False):
        assert isinstance(n_threads, (int, str))

        if n_threads == 'auto':
            n_threads = cpu_count() - 1
        else:
            assert n_threads > 0

        self.n_threads = n_threads
    
        assert self._sim_anneal_prms_set_flag
        self._verify_input()

        classify_cps_ftn = self._gen_classi_cyth_mods(force_compile)

#         raise Exception

        calib_dict = {}

        calib_dict['obj_ftn_wts_arr'] = self.obj_ftn_wts_arr

        if self.obj_1_flag:
            calib_dict['o_1_ppt_thresh_arr'] = self.o_1_ppt_thresh_arr

        if self.obj_2_flag:
            calib_dict['o_2_ppt_thresh_arr'] = self.o_2_ppt_thresh_arr

        if self.obj_4_flag:
            calib_dict['o_4_p_thresh_arr'] = self.o_4_wett_thresh_arr
            assert self.neb_wett_arr.shape[1] == 2, (
                'Implmented for two neibors only!')

        if self.obj_7_flag:
            assert self.neb_wett_arr.shape[1] == 3, (
                'Implmented for three neibors only!')

        if self.temp_adj_iters == 'auto':
            self.temp_adj_iters = self.vals_tot_anom.shape[1] * 1500

        calib_dict['n_cps'] = self.n_cps
        calib_dict['no_cp_val'] = self.no_cp_val
        calib_dict['p_l'] = self.p_l
        calib_dict['fuzz_nos_arr'] = self.fuzz_nos_arr
        calib_dict['slp_anom_calib'] = self.vals_tot_anom

        if self.obj_1_flag or self.obj_3_flag:
            calib_dict['in_ppt_arr_calib'] = self.stn_ppt_arr

        if self.obj_2_flag or self.obj_5_flag:
            calib_dict['in_cats_ppt_arr_calib'] = self.cat_ppt_arr

        if self.obj_4_flag or self.obj_6_flag or self.obj_7_flag:
            calib_dict['in_wet_arr_calib'] = self.neb_wett_arr

        if self.obj_6_flag:
            calib_dict['min_wettness_thresh'] = self.min_wettness_thresh

        if self.obj_8_flag:
            calib_dict['in_lorenz_arr_calib'] = self.lorenz_arr

        calib_dict['anneal_temp_ini'] = self.ini_anneal_temp
        calib_dict['temp_red_alpha'] = self.tem_alpha
        calib_dict['max_m_iters'] = self.tem_chng_iters
        calib_dict['max_n_iters'] = self.max_iters
        calib_dict['n_cpus'] = self.n_threads
        calib_dict['msgs'] = int(self.msgs)
        calib_dict['max_idxs_ct'] = self.max_idx_ct
        calib_dict['max_iters_wo_chng'] = self.max_iters_wo_chng
        calib_dict['min_abs_ppt_thresh'] = self.min_abs_ppt_thresh

        calib_dict['temp_adj_iters'] = self.temp_adj_iters
        calib_dict['min_acc_rate'] = self.min_acc_rate
        calib_dict['max_acc_rate'] = self.max_acc_rate
        calib_dict['max_temp_adj_atmps'] = self.max_temp_adj_atmps
        
        calib_dict['lo_freq_pen_wt'] = self.lo_freq_pen_wt
        calib_dict['min_freq'] = self.min_freq

        _strt = timeit.default_timer()
        self.calib_dict = classify_cps_ftn(calib_dict)
        _stop = timeit.default_timer()

        if self.msgs:
            print('Took %0.1f seconds to calibrate' % (_stop - _strt))

        self.cp_rules = self.calib_dict['best_cp_rules']

        self.curr_obj_vals_arr = self.calib_dict['curr_obj_vals_arr']
        self.best_obj_vals_arr = self.calib_dict['best_obj_vals_arr']
        self.acc_rate_arr = self.calib_dict['acc_rate_arr']
        self.cp_pcntge_arr = self.calib_dict['cp_pcntge_arr']
        self.curr_n_iters_arr = self.calib_dict['curr_n_iters_arr']

        assert check_nans_finite(self.calib_dict['mu_i_k_arr_calib'])
        assert check_nans_finite(self.calib_dict['dofs_arr_calib'])
        assert check_nans_finite(self.calib_dict['last_obj_val'])

        self._classified_flag = True
        return
    
    def plot_iter_obj_vals(self, out_fig_loc, fig_size=(17, 10)):
        if self.msgs:
            print('\n\nPlotting objective function evolution...')

        assert self._classified_flag

        assert isinstance(out_fig_loc, (str, Path))
        out_fig_loc = Path(out_fig_loc)
        assert out_fig_loc.parents[0].exists()

        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        pl_1 = ax.plot(self.curr_n_iters_arr,
                       self.curr_obj_vals_arr,
                       label='curr_obj_val',
                       color='red',
                       alpha=0.75)
        pl_2 = ax.plot(self.curr_n_iters_arr,
                       self.best_obj_vals_arr,
                       label='best_obj_val',
                       color='blue',
                       alpha=0.75)
        ax.set_xlabel('Iteration No. (-)')
        ax.set_ylabel('Objective ftn. value (-)')
        ax.set_ylim(max(-0.1, self.best_obj_vals_arr.min()),
                    self.best_obj_vals_arr.max() * 1.05)

        ax.grid()

        ax_twin = ax.twinx()
        pl_3 = ax_twin.plot(self.curr_n_iters_arr,
                            self.acc_rate_arr,
                            label='acc_rate',
                            color='black',
                            alpha=0.5)
        ax_twin.set_ylabel('Acceptance Rate (%)')
        ax_twin.set_ylim(0, 100)

        pls = pl_1 + pl_2 + pl_3
        labs = [pl.get_label() for pl in pls]
        ax.legend(pls, labs, loc=0)

        ax.set_title(('CP classification - objective function evolution '
                      '(max=%0.3f)') % self.best_obj_vals_arr[-1])

        plt.savefig(str(out_fig_loc), bbox_inches='tight')
        plt.close()
        return

    def plot_verifs(self, out_dir, fig_size=(17, 10)):
        if self.msgs:
            print('\n\nPlotting calibration verification variables...')

        assert self._classified_flag

        assert isinstance(out_dir, (str, Path))
        out_dir = Path(out_dir)
        assert out_dir.exists()

        # rand stuff
        cps_occ_arr = self.calib_dict['rands_rec_arr'][:, 0]
        cps_occ_arr = cps_occ_arr[cps_occ_arr != 9999]

        pts_occ_arr = self.calib_dict['rands_rec_arr'][:, 1]
        pts_occ_arr = pts_occ_arr[pts_occ_arr != 9999]

        fuz_occ_arr = self.calib_dict['rands_rec_arr'][:, 2]
        fuz_occ_arr = fuz_occ_arr[fuz_occ_arr != 9999]

        unq_cps, unq_cps_cts = np.unique(cps_occ_arr, return_counts=True)
        unq_pts, unq_pts_cts = np.unique(pts_occ_arr, return_counts=True)
        unq_fuz, unq_fuz_cts = np.unique(fuz_occ_arr, return_counts=True)

        _vars = [('CP', 'rand_gen_cps_ct.png'),
                 ('Point', 'rand_gen_pts_ct.png'),
                 ('Fuzzy', 'rand_gen_fuz_ct.png')]

        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()
        
        for i, (unq, cts) in enumerate(((unq_cps, unq_cps_cts), 
                                        (unq_pts, unq_pts_cts), 
                                        (unq_fuz, unq_fuz_cts))):
            
            ax.plot(unq, cts / float(cts[0]), alpha=0.7)
            ax.set_xlabel(_vars[i][0] + ' No.')
            ax.set_ylabel('Relative count')
            ax.set_title('Randomly generated %s no. counts during calibration'
                         % _vars[i][0])
            ax.grid()
            plt.savefig(str(out_dir / _vars[i][1]), bbox_inches='tight')
            ax.cla()

        plt.close()

        # loc_mod stuff
        loc_mod_ctr_arr = self.calib_dict['loc_mod_ctr']
        assert loc_mod_ctr_arr.shape[0] == self.n_cps
        loc_rows = max(1, int(0.25 * self.n_cps))
        loc_cols = max(1, int(np.ceil(self.n_cps / loc_rows)))
        tot_sub_plots = loc_rows * loc_cols
        assert tot_sub_plots >= self.n_cps

        rows_idxs, cols_idxs = np.meshgrid(np.arange(loc_rows),
                                           np.arange(loc_cols),
                                           indexing='ij')
        rows_idxs, cols_idxs = rows_idxs.ravel(), cols_idxs.ravel()

        plt.figure(figsize=fig_size)
        hists_grid = gridspec.GridSpec(loc_rows, loc_cols)
        pt_rng = np.arange(loc_mod_ctr_arr.shape[1])
        for i in range(self.n_cps):
            hist_ax = plt.subplot(hists_grid[rows_idxs[i], cols_idxs[i]])
            hist_ax.bar(pt_rng, loc_mod_ctr_arr[i, :])
            hist_ax.set_title('CP No. %2d' % i)
            
            if rows_idxs[i] == rows_idxs[-1]:
                hist_ax.set_xlabel('Point no.')
            else:
                hist_ax.set_xticklabels([])

            if cols_idxs[i] == 0:
                hist_ax.set_ylabel('Count')
            else:
                hist_ax.set_yticklabels([])

        for i in range(self.n_cps, tot_sub_plots):
            hist_ax = plt.subplot(hists_grid[rows_idxs[i], cols_idxs[i]])
            hist_ax.set_axis_off()

        plt.suptitle('Counts of visited points per CP during calibration')
        plt.savefig(str(out_dir / 'loc_mod_ctr.png'), bbox_inches='tight')
        plt.close()
        return
