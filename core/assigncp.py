'''
Created on Jan 2, 2018

@author: Faizan-Uni
'''
import importlib
from pathlib import Path
from psutil import cpu_count

import numpy as np

from .bases import CPOPTBase
from ..alg_dtypes import DT_UL_NP, DT_D_NP
from ..misc.checks import check_nans_finite

from ..codegenr.gen_all import create_justi_cython_files


class CPAssignA(CPOPTBase):
    def __init__(self, msgs=True):
        super().__init__(msgs=msgs)

        self._cp_rules_set_flag = False
        self._mult_cp_rules_set_flag = False
        self._cps_assigned_flag = False
        self._sim_sel_cps_dofs_arr_set_flag = False
        return
    
    def set_cp_rules(self, cp_rules):
        assert isinstance(cp_rules, np.ndarray)
        assert check_nans_finite(cp_rules)
        assert cp_rules.ndim == 2

        self.cp_rules = np.array(cp_rules, dtype=DT_UL_NP, order='C')
        self._cp_rules_set_flag = True
        return

    def set_mult_cp_rules(self, mult_cp_rules):
        assert isinstance(mult_cp_rules, np.ndarray)
        assert check_nans_finite(mult_cp_rules)
        assert mult_cp_rules.ndim == 3

        self.mult_cp_rules = np.array(mult_cp_rules, dtype=DT_UL_NP, order='C')
        self._mult_cp_rules_set_flag = True
        return

    def set_sim_sel_cps_dofs_arr(self, dofs_arr):
        assert isinstance(dofs_arr, np.ndarray)
        assert check_nans_finite(dofs_arr)
        assert dofs_arr.ndim == 2

        self.sim_sel_cps_dofs_arr = np.array(dofs_arr,
                                             dtype=DT_D_NP,
                                             order='C')
        self._sim_sel_cps_dofs_arr_set_flag = True
        return

    def sim_sel_cps(self, n_sel_cp_sims, n_nrst_cps, samp_type='unif'):
        assert isinstance(n_sel_cp_sims, int)
        assert n_sel_cp_sims > 0

        assert isinstance(n_nrst_cps, int)
        assert n_nrst_cps > 0

        assert isinstance(samp_type, str)
        assert (samp_type == 'unif')

        self._verify_sim_input()

        if not self._sim_sel_cps_dofs_arr_set_flag:
            self.sim_sel_cps_dofs_arr = self.dofs_arr.copy(order='C')

        n_time_steps = self.sim_sel_cps_dofs_arr.shape[0]
        n_cps = self.sim_sel_cps_dofs_arr.shape[1]
        assert n_nrst_cps <= n_cps

        desc_dofs_sel_cps_arr = np.full((n_time_steps, n_cps),
                                        9999,
                                        dtype=DT_UL_NP)
        _cps_rng = np.arange(n_cps)
        for i in range(n_time_steps):
            curr_sort_dof_idxs = (
                np.argsort(self.sim_sel_cps_dofs_arr[i, :])[::-1])
            desc_dofs_sel_cps_arr[i, :] = _cps_rng[curr_sort_dof_idxs]

        sim_sel_shape = (n_sel_cp_sims, n_time_steps)
        sim_sel_cps_arr = np.full(sim_sel_shape, 9999, dtype=DT_UL_NP)

        _time_steps_idxs = np.arange(n_time_steps)
        for i in range(n_sel_cp_sims):
            rand_idxs_arr = np.random.randint(0, n_nrst_cps, n_time_steps)
            sim_sel_cps_arr[i] = desc_dofs_sel_cps_arr[_time_steps_idxs,
                                                       rand_idxs_arr]

        self.desc_dofs_sel_cps_arr = desc_dofs_sel_cps_arr
        self.sim_sel_cps_arr = sim_sel_cps_arr
        return

    def _verify_sim_input(self):
        assert self._cp_prms_set_flag
        assert (self._sim_sel_cps_dofs_arr_set_flag or self._cps_assigned_flag)
        return

    def _verify_input(self):
        assert self._anom_set_flag
        assert self._cp_prms_set_flag
        assert self._cp_rules_set_flag
        assert isinstance(self.op_mp_memb_flag, bool)
        return

    def _verify_mult_input(self):
        assert self._anom_set_flag
        assert self._cp_prms_set_flag
        assert self._mult_cp_rules_set_flag
        assert isinstance(self.op_mp_memb_flag, bool)
        return
    
    def _gen_justi_cyth_mods(self, force_compile):
        cyth_dir = Path(__file__).parents[1] / 'cyth'

        create_justi_cython_files(self.cyth_nonecheck,
                                  self.cyth_boundscheck,
                                  self.cyth_wraparound,
                                  self.cyth_cdivision,
                                  self.cyth_infer_types,
                                  self.cyth_language_level,
                                  force_compile,
                                  cyth_dir,
                                  self.op_mp_memb_flag)

#         raise Exception
        importlib.invalidate_caches()

        return importlib.import_module('..cyth.justi_alg',
                                       package='wcpc.core')._assign_cps

    def assign_cps(self, n_threads='auto', force_compile=False):

        assert isinstance(n_threads, (int, str))

        if n_threads == 'auto':
            n_threads = cpu_count() - 1
        else:
            assert n_threads > 0

        self._verify_input()

        assign_dict = {}
        assign_dict['no_cp_val'] = self.no_cp_val
        assign_dict['p_l'] = self.p_l
        assign_dict['fuzz_nos_arr'] = self.fuzz_nos_arr
        assign_dict['cp_rules'] = self.cp_rules
        assign_dict['n_cpus'] = n_threads
        assign_dict['anom'] = self.vals_tot_anom

        _assign_cps = self._gen_justi_cyth_mods(force_compile)

#         raise Exception
        self.assign_dict = _assign_cps(assign_dict)

        self.sel_cps_arr = self.assign_dict['sel_cps']
        self.dofs_arr = self.assign_dict['dofs_arr']
        self.fuzz_dofs_arr = self.assign_dict['cp_dof_arr']

        if self.msgs:
            uni_cps, cps_freqs = np.unique(self.sel_cps_arr, return_counts=True)
            cp_rel_freqs = 100 * cps_freqs / float(self.sel_cps_arr.shape[0])
            cp_rel_freqs = np.round(cp_rel_freqs, 2)
            print('\n%-10s:%s' % ('Unique CPs', 'Relative Frequencies (%)'))
            for x, y in zip(uni_cps, cp_rel_freqs):
                print('%10d:%-20.2f' % (x, y))

        self._cps_assigned_flag = True
        return
    
    def assign_mult_cps(self, n_threads='auto', force_compile=False):

        assert isinstance(n_threads, (int, str))

        if n_threads == 'auto':
            n_threads = cpu_count() - 1
        else:
            assert n_threads > 0

        self._verify_mult_input()

        assign_dict = {}
        assign_dict['no_cp_val'] = self.no_cp_val
        assign_dict['p_l'] = self.p_l
        assign_dict['fuzz_nos_arr'] = self.fuzz_nos_arr
        assign_dict['mult_cp_rules'] = self.mult_cp_rules
        assign_dict['n_cpus'] = n_threads
        assign_dict['anom'] = self.vals_tot_anom
        assign_dict['mult_cps_assign_flag'] = True
    
        _assign_cps = self._gen_justi_cyth_mods(force_compile)

#         raise Exception
        self.assign_dict = _assign_cps(assign_dict)

        self.mult_sel_cps_arr = self.assign_dict['mult_sel_cps']
        return
    
