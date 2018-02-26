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

from ..core.bases import CPDataBase
from ..alg_dtypes import DT_D_NP, DT_UL_NP
from ..misc.checks import check_nans_finite
from ..codegenr.gen_all import create_obj_cython_files


class ObjVals(CPDataBase):
    def __init__(self, msgs=True):
        super(ObjVals, self).__init__(msgs=msgs)

        self._cps_set_flag = False

        return

    def set_cps_arr(self, sel_cps_arr, n_cps):
        assert isinstance(n_cps, int)
        assert n_cps > 0

        assert isinstance(sel_cps_arr, np.ndarray)
        assert check_nans_finite(sel_cps_arr)
        assert len(sel_cps_arr.shape) == 1

        self.sel_cps_arr = np.array(sel_cps_arr, dtype=DT_UL_NP, order='C')
        self.n_cps = n_cps

        self._cps_set_flag = True
        return

    def _verify_input(self):
        assert any((self.obj_1_flag,
                    self.obj_2_flag,
                    self.obj_3_flag,
                    self.obj_4_flag,
                    self.obj_5_flag,
                    self.obj_6_flag,
                    self.obj_7_flag,
                    self.obj_8_flag))

        assert self._cps_set_flag
        self.obj_ftn_wts_arr = np.zeros(self._n_obj_ftns,
                                        dtype=DT_D_NP,
                                        order='C')

        if self.obj_1_flag:
            self.obj_ftn_wts_arr[0] = self.o_1_obj_wt

        if self.obj_2_flag:
            self.obj_ftn_wts_arr[1] = self.o_2_obj_wt

        if self.obj_3_flag:
            self.obj_ftn_wts_arr[2] = self.o_3_obj_wt

        if self.obj_4_flag:
            self.obj_ftn_wts_arr[3] = self.o_4_obj_wt

        if self.obj_5_flag:
            self.obj_ftn_wts_arr[4] = self.o_5_obj_wt

        if self.obj_6_flag:
            self.obj_ftn_wts_arr[5] = self.o_6_obj_wt
            assert self.neb_wett_arr.shape[1] == 2, 'For two nebs right now!'

        if self.obj_7_flag:
            self.obj_ftn_wts_arr[6] = self.o_7_obj_wt
            assert self.neb_wett_arr.shape[1] == 3, 'For three nebs right now!'

        if self.obj_8_flag:
            self.obj_ftn_wts_arr[7] = self.o_8_obj_wt

        assert isinstance(self.op_mp_memb_flag, bool)
        assert isinstance(self.op_mp_obj_ftn_flag, bool)

        return

    def _gen_obj_cyth_mods(self, force_compile=False):

        cyth_dir = Path(__file__).parents[1] / 'cyth'

        create_obj_cython_files(self.obj_1_flag,
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

        return importlib.import_module('..cyth.obj_alg',
                                       package='wcpc.core').get_obj_val

    def cmpt_obj_val(self, n_threads='auto', force_compile=False):
        assert isinstance(n_threads, (int, str))

        if n_threads == 'auto':
            n_threads = cpu_count() - 1
        else:
            assert n_threads > 0

        self._verify_input()

        obj_ftn = self._gen_obj_cyth_mods(force_compile)

#         raise Exception

        obj_dict = {}

        obj_dict['obj_ftn_wts_arr'] = self.obj_ftn_wts_arr
        obj_dict['sel_cps'] = self.sel_cps_arr

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
        obj_dict['n_cpus'] = n_threads
        obj_dict['msgs'] = int(self.msgs)

        obj_dict['lo_freq_pen_wt'] = self.lo_freq_pen_wt
        obj_dict['min_freq'] = self.min_freq

        _strt = timeit.default_timer()
        self.obj_dict = obj_ftn(obj_dict)
        _stop = timeit.default_timer()

        self.obj_val = self.obj_dict['curr_obj_val']

        if self.msgs:
            print('Took %0.1f seconds to compute obj. val.' % (_stop - _strt))
        return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
