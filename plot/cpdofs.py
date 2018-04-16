'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from ..misc.checks import check_nans_finite
from ..misc.ftns import KS_two_samp_test

plt.ioff()


class PlotDOFs:
    def __init__(self, msgs=True):
        self._calib_dofs_set_flag = False
        self._valid_dofs_set_flag = False
        self._all_dofs_set_flag = False
        
        self._labs_list = ['calib', 'valid', 'all']
        self._probs_dofs_arr_dict = {}
        self.n_cps = None

        self.msgs = msgs
        return
    
    def _verify_dof(self, dofs_arr):
        assert check_nans_finite(dofs_arr)
        assert dofs_arr.ndim == 2
        assert dofs_arr.shape[0] and dofs_arr.shape[1]

        assert np.all(dofs_arr >= 0)
        assert np.all(dofs_arr <= 1)

        if self.n_cps is not None:
            assert dofs_arr.shape[1] == self.n_cps
        else:
            self.n_cps = dofs_arr.shape[1]
            self._bins_min_arr = np.full(self.n_cps, +np.inf)
            self._bins_max_arr = np.full(self.n_cps, -np.inf)

        _curr_mins = dofs_arr.min(axis=0)
        self._bins_min_arr = np.where(_curr_mins < self._bins_min_arr,
                                      _curr_mins,
                                      self._bins_min_arr)

        _curr_maxs = dofs_arr.max(axis=0)
        self._bins_max_arr = np.where(_curr_maxs > self._bins_max_arr,
                                      _curr_maxs,
                                      self._bins_max_arr)
        return

    def set_calib_dofs(self, dofs_arr):
        self._verify_dof(dofs_arr)

        self.calib_dofs_arr = dofs_arr.copy()
        self.calib_probs_arr = (np.arange(1.0, dofs_arr.shape[0] + 1) /
                                (dofs_arr.shape[0] + 1))

        self._probs_dofs_arr_dict[self._labs_list[0]] = (self.calib_dofs_arr,
                                                         self.calib_probs_arr)

        self._calib_dofs_set_flag = True
        return

    def set_valid_dofs(self, dofs_arr):
        self._verify_dof(dofs_arr)

        self.valid_dofs_arr = dofs_arr.copy()
        self.valid_probs_arr = (np.arange(1.0, dofs_arr.shape[0] + 1) /
                                (dofs_arr.shape[0] + 1))

        self._probs_dofs_arr_dict[self._labs_list[1]] = (self.valid_dofs_arr,
                                                         self.valid_probs_arr)

        self._valid_dofs_set_flag = True
        return

    def set_all_dofs(self, dofs_arr):
        self._verify_dof(dofs_arr)

        self.all_dofs_arr = dofs_arr.copy()
        self.all_probs_arr = (np.arange(1.0, dofs_arr.shape[0] + 1) /
                                (dofs_arr.shape[0] + 1))

        self._probs_dofs_arr_dict[self._labs_list[2]] = (self.all_dofs_arr,
                                                         self.all_probs_arr)

        self._all_dofs_set_flag = True
        return

    def plot_verifs(self, out_dir, fig_size=(10, 7)):
        if self.msgs:
            print('\n\nPlotting CP dofs...')

        assert any([self._calib_dofs_set_flag,
                    self._valid_dofs_set_flag,
                    self._all_dofs_set_flag])

        assert isinstance(out_dir, (str, Path))
        out_dir = Path(out_dir)
        assert out_dir.exists()

        plt.figure(figsize=fig_size)
        cdf_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        hist_ax = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
        
        n_bins = 10
        for i in range(self.n_cps):
            rwidth = 0.98
            line_width = 3

            curr_title = (
                'Degree of fulfilment\'s CDF/Histogram for CP No. %d\n' % i)

            if self._calib_dofs_set_flag:
                ks_str = ''
                _ = self.calib_dofs_arr[:, i].copy()
                _.sort()
                ks_part_ftn = partial(KS_two_samp_test,
                                      ref_vals=_,
                                      alpha=0.95)

            curr_bins_arr = np.linspace(self._bins_min_arr[i],
                                        self._bins_max_arr[i],
                                        n_bins + 1)
            for _lab in self._labs_list:
                if _lab not in self._probs_dofs_arr_dict:
                    continue
                
                curr_dofs, curr_probs = self._probs_dofs_arr_dict[_lab]
                curr_dofs = curr_dofs[:, i].copy()
                curr_dofs.sort()

                if self._calib_dofs_set_flag:
                    ks_res = ks_part_ftn(sim_vals=curr_dofs)
                    ks_str += (', ' + _lab + ': ' + ks_res)

                cdf_ax.plot(curr_dofs,
                            curr_probs,
                            alpha=0.3,
                            label=_lab,
                            linewidth=line_width)

                hist_ax.hist(curr_dofs,
                             bins=curr_bins_arr,
                             alpha=0.3,
                             label=_lab,
                             rwidth=rwidth,
                             align='mid',
                             density=True)
                rwidth -= 0.17
                line_width -= 0.5

            cdf_ax.set_ylabel('Probability')
            cdf_ax.set_xticklabels([])
            cdf_ax.get_xaxis().set_tick_params(width=0)
            cdf_ax.grid()
            cdf_ax.legend()

            cdf_ax.set_title(curr_title + 'KS test - ' + ks_str[2:])
            hist_ax.set_xlabel('Degree of fulfilment')
            hist_ax.set_ylabel('Density')
            hist_ax.legend()

            plt.savefig(str(out_dir / ('dofs_cp_%0.2d.png' % i)),
                        bbox_inches='tight')

            cdf_ax.cla()
            hist_ax.cla()
        plt.close()
        return


class PlotFuzzDOFs:
    def __init__(self, msgs=True):
        self._calib_dofs_set_flag = False
        self._valid_dofs_set_flag = False
        self._all_dofs_set_flag = False

        self._labs_list = ['calib', 'valid', 'all']
        self._probs_dofs_arr_dict = {}
        self.n_cps = None

        self.msgs = msgs
        return

    def _verify_dof(self, dofs_arr):
        assert check_nans_finite(dofs_arr)
        assert dofs_arr.ndim == 3, dofs_arr.ndim
        assert dofs_arr.shape[0] and dofs_arr.shape[1] and dofs_arr.shape[2]
        assert np.all(dofs_arr >= 0) and np.all(dofs_arr <= 1)

        dofs_arr = dofs_arr.sum(axis=-1)

        if self.n_cps is not None:
            assert dofs_arr.shape[1] == self.n_cps
        else:
            self.n_cps = dofs_arr.shape[1]
            self._bins_min_arr = np.full(self.n_cps, +np.inf)
            self._bins_max_arr = np.full(self.n_cps, -np.inf)

        _curr_mins = dofs_arr.min(axis=0)
        self._bins_min_arr = np.where(_curr_mins < self._bins_min_arr,
                                      _curr_mins,
                                      self._bins_min_arr)

        _curr_maxs = dofs_arr.max(axis=0)
        self._bins_max_arr = np.where(_curr_maxs > self._bins_max_arr,
                                      _curr_maxs,
                                      self._bins_max_arr)
        return dofs_arr

    def set_calib_dofs(self, dofs_arr):

        self.calib_dofs_arr = self._verify_dof(dofs_arr)
        self.calib_probs_arr = (
            (np.arange(1.0, self.calib_dofs_arr.shape[0] + 1) /
             (self.calib_dofs_arr.shape[0] + 1)))

        self._probs_dofs_arr_dict[self._labs_list[0]] = (self.calib_dofs_arr,
                                                         self.calib_probs_arr)

        self._calib_dofs_set_flag = True
        return

    def set_valid_dofs(self, dofs_arr):

        self.valid_dofs_arr = self._verify_dof(dofs_arr)
        self.valid_probs_arr = (
            (np.arange(1.0, self.valid_dofs_arr.shape[0] + 1) /
             (self.valid_dofs_arr.shape[0] + 1)))

        self._probs_dofs_arr_dict[self._labs_list[1]] = (self.valid_dofs_arr,
                                                         self.valid_probs_arr)

        self._valid_dofs_set_flag = True
        return

    def set_all_dofs(self, dofs_arr):

        self.all_dofs_arr = self._verify_dof(dofs_arr)
        self.all_probs_arr = (np.arange(1.0, self.all_dofs_arr.shape[0] + 1) /
                                (self.all_dofs_arr.shape[0] + 1))

        self._probs_dofs_arr_dict[self._labs_list[2]] = (self.all_dofs_arr,
                                                         self.all_probs_arr)

        self._all_dofs_set_flag = True
        return

    def plot_verifs(self, out_dir, fig_size=(10, 7)):
        if self.msgs:
            print('\n\nPlotting Fuzzy CP dofs...')

        assert any([self._calib_dofs_set_flag,
                    self._valid_dofs_set_flag,
                    self._all_dofs_set_flag])

        assert isinstance(out_dir, (str, Path))
        out_dir = Path(out_dir)
        assert out_dir.exists()

        plt.figure(figsize=fig_size)
        cdf_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        hist_ax = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

        n_bins = 10
        for i in range(self.n_cps):
            rwidth = 0.98
            line_width = 3

            curr_title = (('Fuzzy sum degree of fulfilment\'s CDF/Histogram '
                           'for CP No. %d\n') % i)

            if self._calib_dofs_set_flag:
                ks_str = ''
                _ = self.calib_dofs_arr[:, i].copy()
                _.sort()
                ks_part_ftn = partial(KS_two_samp_test,
                                      ref_vals=_,
                                      alpha=0.95)

            curr_bins_arr = np.linspace(self._bins_min_arr[i],
                                        self._bins_max_arr[i],
                                        n_bins + 1)
            for _lab in self._labs_list:
                if _lab not in self._probs_dofs_arr_dict:
                    continue

                curr_dofs, curr_probs = self._probs_dofs_arr_dict[_lab]
                curr_dofs = curr_dofs[:, i].copy()
                curr_dofs.sort()

                if self._calib_dofs_set_flag:
                    ks_res = ks_part_ftn(sim_vals=curr_dofs)
                    ks_str += (', ' + _lab + ': ' + ks_res)

                cdf_ax.plot(curr_dofs,
                            curr_probs,
                            alpha=0.3,
                            label=_lab,
                            linewidth=line_width)

                hist_ax.hist(curr_dofs,
                             bins=curr_bins_arr,
                             alpha=0.3,
                             label=_lab,
                             rwidth=rwidth,
                             align='mid',
                             density=True)
                rwidth -= 0.17
                line_width -= 0.5

            cdf_ax.set_ylabel('Probability')
            cdf_ax.set_xticklabels([])
            cdf_ax.get_xaxis().set_tick_params(width=0)
            cdf_ax.grid()
            cdf_ax.legend()

            cdf_ax.set_title(curr_title + 'KS test - ' + ks_str[2:])
            hist_ax.set_xlabel('Degree of fulfilment')
            hist_ax.set_ylabel('Density')
            hist_ax.legend()

            plt.savefig(str(out_dir / ('fuzz_dofs_cp_%0.2d.png' % i)),
                        bbox_inches='tight')

            cdf_ax.cla()
            hist_ax.cla()
        plt.close()
        return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program
    
    main_dir = Path(os.getcwd())

    os.chdir(main_dir)
    
    STOP = timeit.default_timer()  # Ending time
    print(('\n### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ###' % (time.asctime(), STOP - START)))
