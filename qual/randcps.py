'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import importlib
from pathlib import Path
from psutil import cpu_count

import numpy as np
import matplotlib.pyplot as plt

from ..codegenr.gen_all import gen_mod_cp_rules_cyth_files
from ..core.assigncp import CPAssignA
from ..qual.objvals import ObjVals
from ..qual.wettness import WettnessIndex
from ..alg_dtypes import DT_UL_NP, DT_D_NP
from ..misc.checks import check_nans_finite

plt.ioff()


class RandCPsGen:
    def __init__(self, msgs=True):
        self.msgs = msgs
        self._mult_cps_gened = False

        self.cyth_nonecheck = False
        self.cyth_boundscheck = False
        self.cyth_wraparound = False
        self.cyth_cdivision = True
        self.cyth_language_level = 3
        self.cyth_infer_types = None
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

    def _gen_mod_cp_rules_cyth_mods(self, force_compile=False):

        cyth_dir = Path(__file__).parents[1] / 'cyth'

        gen_mod_cp_rules_cyth_files(self.cyth_nonecheck,
                                    self.cyth_boundscheck,
                                    self.cyth_wraparound,
                                    self.cyth_cdivision,
                                    self.cyth_infer_types,
                                    self.cyth_language_level,
                                    force_compile,
                                    cyth_dir)

#         raise Exception
        importlib.invalidate_caches()

        return importlib.import_module('..cyth.gen_mod_cp_rules_main',
                                       package='wcpc.core').get_rand_cp_rules

    def gen_cp_rules(self,
                     n_cps,
                     n_gens,
                     max_idxs_ct,
                     n_pts,
                     p_l,
                     no_cp_val,
                     fuzz_nos_arr,
                     anom,
                     n_threads='auto',
                     force_compile=False):

        assert isinstance(n_cps, int)
        assert n_cps > 0

        assert isinstance(n_gens, int)
        assert n_gens > 0
        self.n_gens = n_gens

        assert isinstance(max_idxs_ct, int)
        assert max_idxs_ct > 0

        assert isinstance(no_cp_val, int)
        assert no_cp_val > n_cps

        assert isinstance(n_pts, int)
        assert n_pts > 0

        assert isinstance(p_l, float)
        assert p_l > 0

        assert isinstance(fuzz_nos_arr, np.ndarray)
        assert check_nans_finite(fuzz_nos_arr)
        assert fuzz_nos_arr.shape[0] > 0
        assert fuzz_nos_arr.shape[1] == 3

        assert isinstance(anom, np.ndarray)
        assert check_nans_finite(anom)
        assert len(anom.shape) == 2
        assert anom.shape[0] > 0
        assert anom.shape[1] > 0

        assert isinstance(n_threads, (int, str))

        if n_threads == 'auto':
            n_threads = cpu_count() - 1
        else:
            assert n_threads > 0

        gen_cps_ftn = self._gen_mod_cp_rules_cyth_mods(force_compile)

        gen_dict = {}
        gen_dict['n_cps'] = n_cps
        gen_dict['n_cpus'] = n_threads
        gen_dict['max_idxs_ct'] = max_idxs_ct
        gen_dict['n_gens'] = self.n_gens
        gen_dict['n_pts'] = n_pts

        gen_dict['no_cp_val'] = no_cp_val
        gen_dict['p_l'] = p_l
        gen_dict['fuzz_nos_arr'] = fuzz_nos_arr
        gen_dict['anom'] = anom
    
        self.gen_dict = gen_cps_ftn(gen_dict)
        self.mult_cp_rules = self.gen_dict['mult_cp_rules']
        self.mult_sel_cps = self.gen_dict['mult_sel_cps']
        self._mult_cps_gened = True
        return


class RandCPsPerfComp(CPAssignA, RandCPsGen, ObjVals):

    def __init__(self):
#         super().__init__()
        CPAssignA.__init__(self)
        RandCPsGen.__init__(self)
        ObjVals.__init__(self)

        self._mult_cp_rules_set_flag = False
        self._mult_sel_cps_arr_set_flag = False
        self._mult_obj_calced = False
        self._mult_wett_calced = False

        self.obj_vals_arr = None
        return

    def set_mult_cp_rules(self, mult_cp_rules):
        assert isinstance(mult_cp_rules, np.ndarray)
        assert check_nans_finite(mult_cp_rules)
        assert len(mult_cp_rules.shape) == 3

        self.mult_cp_rules = np.array(mult_cp_rules, dtype=DT_UL_NP, order='C')
        self.n_gens = self.mult_cp_rules.shape[0]

        self._mult_cp_rules_set_flag = True
        return

    def set_mult_sel_cps_arr(self, mult_sel_cps_arr):
        assert isinstance(mult_sel_cps_arr, np.ndarray)
        assert check_nans_finite(mult_sel_cps_arr)
        assert len(mult_sel_cps_arr.shape) == 2

        self.mult_sel_cps_arr = np.array(mult_sel_cps_arr, dtype=DT_UL_NP, order='C')

        self._mult_sel_cps_arr_set_flag = True
        return

    def _verify_obj_vals_input(self):
        assert any((self.obj_1_flag,
                    self.obj_2_flag,
                    self.obj_3_flag,
                    self.obj_4_flag,
                    self.obj_5_flag,
                    self.obj_6_flag,
                    self.obj_7_flag,
                    self.obj_8_flag))

        assert self._mult_cp_rules_set_flag
        assert self._mult_sel_cps_arr_set_flag
        self.obj_ftn_wts_arr = np.zeros(self._n_obj_ftns,
                                        dtype=DT_D_NP,
                                        order='C')

        in_lens_list = []
        if self.obj_1_flag:
            self.obj_ftn_wts_arr[0] = self.o_1_obj_wt
            in_lens_list.append(self.stn_ppt_arr.shape[0])

        if self.obj_2_flag:
            self.obj_ftn_wts_arr[1] = self.o_2_obj_wt
            in_lens_list.append(self.cat_ppt_arr.shape[0])

        if self.obj_3_flag:
            self.obj_ftn_wts_arr[2] = self.o_3_obj_wt
            in_lens_list.append(self.stn_ppt_arr.shape[0])

        if self.obj_4_flag:
            self.obj_ftn_wts_arr[3] = self.o_4_obj_wt
            in_lens_list.append(self.neb_wett_arr.shape[0])

        if self.obj_5_flag:
            self.obj_ftn_wts_arr[4] = self.o_5_obj_wt
            in_lens_list.append(self.cat_ppt_arr.shape[0])

        if self.obj_6_flag:
            self.obj_ftn_wts_arr[5] = self.o_6_obj_wt
            assert self.neb_wett_arr.shape[1] == 2, 'For two nebs right now!'
            in_lens_list.append(self.neb_wett_arr.shape[0])

        if self.obj_7_flag:
            self.obj_ftn_wts_arr[6] = self.o_7_obj_wt
            assert self.neb_wett_arr.shape[1] == 3, 'For three nebs right now!'
            in_lens_list.append(self.neb_wett_arr.shape[0])

        if self.obj_8_flag:
            self.obj_ftn_wts_arr[7] = self.o_8_obj_wt
            in_lens_list.append(self.lorenz_arr.shape[0])

        in_lens_list.append(self.mult_sel_cps_arr.shape[1])
        
        _len = in_lens_list[0]
        
        for curr_len in in_lens_list[1:]:
            assert curr_len == _len
        
        assert isinstance(self.op_mp_obj_ftn_flag, bool)
        return

    def cmpt_mult_obj_val(self, n_threads='auto', force_compile=False):
        assert isinstance(n_threads, (int, str))

        if n_threads == 'auto':
            n_threads = cpu_count() - 1
        else:
            assert n_threads > 0

        self._verify_obj_vals_input()

        obj_ftn = self._gen_obj_cyth_mods(force_compile)

#         raise Exception

        obj_dict = {}

        obj_dict['obj_ftn_wts_arr'] = self.obj_ftn_wts_arr
        obj_dict['mult_sel_cps'] = self.mult_sel_cps_arr

        if self.obj_1_flag:
            obj_dict['o_1_ppt_thresh_arr'] = self.o_1_ppt_thresh_arr

        if self.obj_2_flag:
            obj_dict['o_2_ppt_thresh_arr'] = self.o_2_ppt_thresh_arr

        if self.obj_4_flag:
            obj_dict['o_4_p_thresh_arr'] = self.o_4_wett_thresh_arr
            assert self.neb_wett_arr.shape[1] == 2, (
                'Implmented for two neibors only!')

        if self.obj_7_flag:
            assert self.neb_wett_arr.shape[1] == 3, (
                'Implmented for three neibors only!')

        obj_dict['n_cps'] = self.n_cps

        if self.obj_1_flag or self.obj_3_flag:
            obj_dict['in_ppt_arr_calib'] = self.stn_ppt_arr

        if self.obj_2_flag or self.obj_5_flag:
            obj_dict['in_cats_ppt_arr_calib'] = self.cat_ppt_arr

        if self.obj_4_flag or self.obj_6_flag or self.obj_7_flag:
            obj_dict['in_wet_arr_calib'] = self.neb_wett_arr

        if self.obj_6_flag:
            obj_dict['min_wettness_thresh'] = self.min_wettness_thresh

        if self.obj_8_flag:
            obj_dict['in_lorenz_arr_calib'] = self.lorenz_arr

        obj_dict['n_cpus'] = n_threads
        obj_dict['msgs'] = int(self.msgs)
        obj_dict['mult_obj_vals_flag'] = True

        obj_dict['lo_freq_pen_wt'] = self.lo_freq_pen_wt
        obj_dict['min_freq'] = self.min_freq

        _strt = timeit.default_timer()
        self.obj_dict = obj_ftn(obj_dict)
        _stop = timeit.default_timer()

        self.obj_vals_arr = self.obj_dict['obj_vals_arr']

        if self.msgs:
            print('Took %0.1f seconds to compute obj. val.' % (_stop - _strt))
        self._mult_obj_calced = True
        return

    def cmpt_wettnesses(self, in_ppt_arr):
        assert self._mult_cp_rules_set_flag
        assert self._mult_sel_cps_arr_set_flag

        wettness = WettnessIndex(False)
        wettness.set_ppt_arr(in_ppt_arr)
        
        self.mult_cp_rules_sorted = np.zeros_like(self.mult_cp_rules)
        self.mean_wett_arrs = np.zeros((self.n_gens, self.n_cps),
                                       dtype=DT_D_NP,
                                       order='C')

        for i in range(self.n_gens):
            wettness.set_cps_arr(self.mult_sel_cps_arr[i], self.n_cps)
            wettness.cmpt_wettness_idx()
            wettness.reorder_cp_rules(self.mult_cp_rules[i])
            self.mult_cp_rules_sorted[i] = wettness.cp_rules_sorted

            _mean_wett_arr = wettness.mean_cp_wett_sorted_arr
            self.mean_wett_arrs[i] = _mean_wett_arr

        print(self.mean_wett_arrs)
        self._mult_wett_calced = True
        return

    def compare_wettnesses(self,
                           in_wettness_arr,
                           out_fig_path,
                           fig_size=(15, 10)):
        assert self._mult_wett_calced

        assert isinstance(in_wettness_arr, np.ndarray)
        assert check_nans_finite(in_wettness_arr)
        assert len(in_wettness_arr.shape) == 1
        assert in_wettness_arr.shape[0] == self.n_cps

        out_fig_path = Path(out_fig_path)
        assert out_fig_path.parents[0].exists()

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert fig_size[0] > 0
        assert fig_size[1] > 0

        n_cps_rng = list(range(self.n_cps))
        plt.figure(figsize=fig_size)

        box_plots_list = []
        wett_pts_list = []
        for i in range(self.n_cps):
            _idxs = ~np.isnan(self.mean_wett_arrs[:, i])
            box_plots_list.append(self.mean_wett_arrs[:, i][_idxs])
            wett_pts_list.append(_idxs.sum())

#             print(i,
#                   self.mean_wett_arrs[:, i][_idxs],
#                   np.nanmin(self.mean_wett_arrs[:, i]),
#                   np.nanmax(self.mean_wett_arrs[:, i]))

        plt.boxplot(box_plots_list, positions=n_cps_rng)
        plt.scatter(n_cps_rng, in_wettness_arr, label='calibrated')

        plt.ylim(0, 1.1 * max(np.nanmax(self.mean_wett_arrs),
                              in_wettness_arr.max()))

        xtick_labs = ['%d' % (n_cps_rng[i]) for i in range(self.n_cps)]
        plt.xlabel('CP')
        plt.ylabel('Wettness Index')
        plt.xticks(n_cps_rng, xtick_labs)
        plt.title('Calibrated vs. Random (n=%d) CPs comparison' % self.n_gens)
        plt.grid()
        plt.legend()
        plt.savefig(str(out_fig_path), bbox_inches='tight')
        plt.close()
        return

    def compare_obj_vals(self,
                         in_obj_val,
                         out_fig_path,
                         fig_size=(15, 10)):
        assert self._mult_obj_calced

        assert isinstance(in_obj_val, float)
        assert check_nans_finite(in_obj_val)

        out_fig_path = Path(out_fig_path)
        assert out_fig_path.parents[0].exists()

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert fig_size[0] > 0
        assert fig_size[1] > 0

        sorted_obj_vals = np.sort(self.obj_vals_arr)
        probs = np.arange(1, self.n_gens + 1) / (self.n_gens + 1)

        obj_val_idx = np.searchsorted(sorted_obj_vals, in_obj_val)
        if obj_val_idx == self.n_gens:
            interp_prob = 1
        else:
            interp_prob = probs[obj_val_idx]

        plt.figure(figsize=fig_size)
        plt.plot(sorted_obj_vals, probs, marker='o', color='b', label='Random')
        plt.scatter(in_obj_val, interp_prob, color='k', label='Calibrated')
        plt.ylim(0, 1.05)
        plt.xlabel('Objective function value')
        plt.ylabel('Probability')
        plt.title('Calibrated vs. Random (n=%d) CPs comparison' % self.n_gens)
        plt.legend()
        plt.grid()

        plt.savefig(str(out_fig_path), bbox_inches='tight')
        plt.close()
        return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
