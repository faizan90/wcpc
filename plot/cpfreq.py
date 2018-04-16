'''
Created on Jan 3, 2018

@author: Faizan-Uni
'''
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

plt.ioff()

from ..misc.checks import check_nans_finite
from ..alg_dtypes import DT_D_NP


class CPHistPlot:
    def __init__(self, msgs=True):
        assert isinstance(msgs, (bool, int))
        self.msgs = msgs

        self._values_ser_set_flag = False
        self._sel_cps_ser_set_flag = False
        self._prms_set_flag = False
        self._cmpt_hist_flag = False
        return

    def set_values_ser(self, values_ser):
        assert isinstance(values_ser, pd.Series)
        assert check_nans_finite(values_ser.values)
        assert values_ser.ndim == 1

        self.values_ser = pd.Series(values_ser, dtype=DT_D_NP)
        self._values_ser_set_flag = True
        return

    def set_sel_cps_ser(self, sel_cps_ser):
        assert isinstance(sel_cps_ser, pd.Series)
        assert check_nans_finite(sel_cps_ser)
        assert sel_cps_ser.ndim == 1

        self.sel_cps_ser = pd.Series(sel_cps_ser, dtype=DT_D_NP)
        self._sel_cps_ser_set_flag = True
        return

    def set_hist_plot_prms(self,
                           prev_steps,
                           post_steps,
                           miss_cp_val,
                           season_months,
                           min_prob=None,
                           max_prob=None,
                           n_cens_time=None,
                           freq='D'):

        assert isinstance(prev_steps, int)
        assert isinstance(post_steps, int)
        assert isinstance(miss_cp_val, int)

        if min_prob is not None:
            assert isinstance(min_prob, float)
        if max_prob is not None:
            assert isinstance(max_prob, float)

        if n_cens_time is not None:
            assert isinstance(n_cens_time, int)
            assert 0 < n_cens_time < float('inf')

        assert isinstance(freq, str)
        assert freq == 'D'

        assert (prev_steps > 0) or (post_steps > 0)

        self.prev_steps = prev_steps
        self.post_steps = post_steps
        self.miss_cp_val = miss_cp_val
        self.n_cens_time = n_cens_time

        if min_prob is not None:
            assert 0 < min_prob < 1

        self.min_prob = min_prob

        if max_prob is not None:
            assert 0 < max_prob < 1

        self.max_prob = max_prob

        if (min_prob is not None) and (max_prob is not None):
            assert min_prob < max_prob

        assert (min_prob is not None) or (max_prob is not None)

        assert isinstance(season_months, np.ndarray)
        assert check_nans_finite(season_months)
        assert season_months.ndim == 1
        assert np.all(season_months > 0)
        assert season_months.shape[0]

        self.season_months = season_months

        tot_splots = prev_steps + post_steps + 1
        self.tot_splot_cols = int(tot_splots / 2) + (tot_splots % 2)
        self.tot_splot_rows = 2

        self.time_lags_list = (list(range(-prev_steps, 0)) +
                               list(range(post_steps + 1)))

        assert self.time_lags_list

        if freq == 'D':
            self.freq = pd.offsets.Day()

        self._prms_set_flag = True
        return

    def cmpt_cp_hists(self):
        assert self._values_ser_set_flag
        assert self._sel_cps_ser_set_flag
        assert self._prms_set_flag

        assert self.values_ser.shape[0] == self.sel_cps_ser.shape[0]
        if self.n_cens_time is not None:
            assert self.n_cens_time < self.values_ser.shape[0]

        _dates = pd.date_range(self.values_ser.index[0],
                               self.values_ser.index[-1],
                               freq=self.freq)

        month_idxs = np.zeros(_dates.shape[0], dtype=bool)
        for month in self.season_months:
            month_idxs = month_idxs | (_dates.month == month)
        _dates = _dates[month_idxs]

        self.values_ser = self.values_ser.reindex(_dates)
        self.sel_cps_ser = self.sel_cps_ser.reindex(_dates)

        assert self.values_ser.shape[0] == self.sel_cps_ser.shape[0]

        self.sel_cps_ser.fillna(value=self.miss_cp_val, inplace=True)

        _cps, _cp_cts = np.unique(self.sel_cps_ser,
                                  return_counts=True)

        self.cp_freqs_ser = pd.Series(index=_cps, data=_cp_cts)

        _ = self.values_ser.rank(method='max')
        self.values_prob_ser = (_ / (_.max() + 1))

        self.all_cps_list = np.unique(self.sel_cps_ser.values)
        self.n_all_cps = self.all_cps_list.shape[0]

        self._xx_list = []

        if self.min_prob is not None:
            self.lag_cp_ct_min_dict = {}

            temp_dates = (self.values_prob_ser[self.values_prob_ser <=
                                               self.min_prob].index)

            if self.n_cens_time is not None:
                keep_dates = []
                for temp_date in temp_dates:
                    _time_lag = pd.Timedelta(self.n_cens_time * self.freq)
                    d_1 = temp_date - _time_lag
                    d_2 = temp_date + _time_lag
                    min_val_date = self.values_prob_ser.loc[d_1:d_2].idxmin()
                    if min_val_date not in keep_dates:
                        keep_dates += [min_val_date]

                assert keep_dates
            else:
                keep_dates = temp_dates

            self.values_lo_prob_dates = pd.DatetimeIndex(keep_dates)

            self._xx_list.append([self.lag_cp_ct_min_dict,
                                  self.values_lo_prob_dates,
                                  'le_events'])

        if self.max_prob is not None:
            self.lag_cp_ct_max_dict = {}

            temp_dates = (self.values_prob_ser[self.values_prob_ser >=
                                               self.max_prob].index)
            if self.n_cens_time is not None:
                keep_dates = []
                for temp_date in temp_dates:
                    _time_lag = pd.Timedelta(self.n_cens_time * self.freq)
                    d_1 = temp_date - _time_lag
                    d_2 = temp_date + _time_lag
                    max_val_date = self.values_prob_ser.loc[d_1:d_2].idxmax()
                    if max_val_date not in keep_dates:
                        keep_dates += [max_val_date]

                assert keep_dates
            else:
                keep_dates = temp_dates

            self.values_hi_prob_dates = pd.DatetimeIndex(keep_dates)

            self._xx_list.append([self.lag_cp_ct_max_dict,
                                  self.values_hi_prob_dates,
                                  'ge_events'])

        assert self._xx_list

        for i in range(len(self._xx_list)):
            for lag in self.time_lags_list:
                _time_lag = pd.Timedelta(lag * self.freq)

                _time_lag_dates = self._xx_list[i][1] + _time_lag
                _time_lag_dates = (
                    _time_lag_dates.intersection(self.sel_cps_ser.index))

                _time_lag_cps = self.sel_cps_ser.loc[_time_lag_dates]

                _time_lag_cps.dropna(inplace=True)

                (_unique_time_lag_cps,
                 _time_lag_cp_cts) = np.unique(_time_lag_cps.values,
                                               return_counts=True)

                _cp_ct_ser = pd.Series(index=self.all_cps_list,
                                       dtype=int,
                                       data=np.full(self.n_all_cps, 0))
                _cp_ct_ser.loc[_unique_time_lag_cps] = _time_lag_cp_cts

                self._xx_list[i][0][lag] = _cp_ct_ser

        self._cmpt_hist_flag = True
        return

    def plot_cp_hists(self,
                      fig_title,
                      fig_name_suff,
                      out_fig_dir,
                      fig_size=(15, 7),
                      tick_txt_size=5,
                      dpi=150):

        assert self._cmpt_hist_flag

        assert isinstance(fig_title, str)
        assert isinstance(fig_name_suff, str)

        out_fig_dir = Path(out_fig_dir)
        assert out_fig_dir.parents[0].exists()
        if not out_fig_dir.exists():
            out_fig_dir.mkdir()

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert all(fig_size)

        assert isinstance(tick_txt_size, int)
        assert tick_txt_size > 0

        assert isinstance(dpi, int)
        assert dpi > 0

        rows_seq = np.repeat(range(self.tot_splot_rows),
                                   self.tot_splot_cols)
        cols_seq = np.tile(range(self.tot_splot_cols),
                                 self.tot_splot_rows)

        _dumm_cp_range = list(range(self.n_all_cps))

        for j in range(len(self._xx_list)):
            # Frequencies before/after le_/ge_events
            fig, ax = plt.subplots(self.tot_splot_rows,
                                   self.tot_splot_cols,
                                   sharex=True,
                                   sharey=False)

            prob_lab = self._xx_list[j][2]

            for i, lag in enumerate(self.time_lags_list):
                _ax = ax[rows_seq[i], cols_seq[i]]

                _cp_ct_ser = self._xx_list[j][0][lag]

                _ax.bar(_dumm_cp_range, _cp_ct_ser.values)
                _ax.set_xticks(_dumm_cp_range)

                if rows_seq[i] == rows_seq[-1]:
                    _ax.set_xticklabels(_cp_ct_ser.index.astype(int))
                    _ax.set_xlabel('CP')
                else:
                    _ax.set_xticklabels([])
                    _ax.set_xlabel('')

                if cols_seq[i] == 0:
                    _ax.set_ylabel('CP Frequency')
                else:
                    _ax.set_ylabel('')

                _ax.tick_params(axis='both',
                                which='major',
                                labelsize=tick_txt_size)

                _at = AnchoredText('lag: %d step(s)' % lag,
                                   frameon=False,
                                   loc=2)

                _at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                _ax.add_artist(_at)

            fig.set_size_inches(*fig_size)
            fig.suptitle(fig_title + '\n(%s)' % prob_lab)

            out_path = out_fig_dir / ((fig_name_suff + '_%s.png') %
                                      (prob_lab))
            plt.savefig(str(out_path), bbox_inches='tight', dpi=dpi)
            plt.close()

            # Over all relative frequency of each CP
            if not j:
                fig, ax = plt.subplots(1, 1, figsize=fig_size)

                ax.bar(range(self.cp_freqs_ser.shape[0]),
                       self.cp_freqs_ser.values / self.sel_cps_ser.shape[0])
                ax.set_xticks(range(self.cp_freqs_ser.shape[0]))

                _labs = ['%d (%d)' % (self.cp_freqs_ser.index[i],
                                      self.cp_freqs_ser.values[i])
                         for i in range(self.cp_freqs_ser.shape[0])]
                ax.set_xticklabels(_labs)

                ax.set_xlabel('CP (counts)')
                ax.set_ylabel('Relative CP frequency')

                ax.grid()

                fig.suptitle('Relative CP frequencies (N=%d)' %
                             self.sel_cps_ser.shape[0])

                out_path = out_fig_dir / (fig_name_suff + '_cp_frequency.png')
                plt.savefig(str(out_path), bbox_inches='tight', dpi=dpi)
                plt.close()

            # Conditional frequencies before/after le_/ge_events
            fig, ax = plt.subplots(self.tot_splot_rows,
                                   self.tot_splot_cols,
                                   sharex=True,
                                   sharey=False,
                                   figsize=fig_size)

            prob_lab = self._xx_list[j][2]

            for i, lag in enumerate(self.time_lags_list):
                _ax = ax[rows_seq[i], cols_seq[i]]

                _cp_ct_ser = self._xx_list[j][0][lag]

                _rel_freq = _cp_ct_ser / self.cp_freqs_ser[_cp_ct_ser.index]
                
                _ax.bar(_dumm_cp_range, _rel_freq.values)
                _ax.set_xticks(_dumm_cp_range)

                if rows_seq[i] == rows_seq[-1]:
                    _ax.set_xticklabels(_cp_ct_ser.index.astype(int))
                    _ax.set_xlabel('CP')
                else:
                    _ax.set_xticklabels([])
                    _ax.set_xlabel('')

                if cols_seq[i] == 0:
                    _ax.set_ylabel('Conditional CP Frequency')
                else:
                    _ax.set_ylabel('')

                _ax.tick_params(axis='both',
                                which='major',
                                labelsize=tick_txt_size)

                _at = AnchoredText('lag: %d step(s)' % lag,
                                   frameon=False,
                                   loc=2)

                _at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                _ax.add_artist(_at)

            fig.set_size_inches(*fig_size)
            fig.suptitle(fig_title + '\n(%s)' % prob_lab)
            out_path = out_fig_dir / ((fig_name_suff + '_cond_%s.png') %
                                      (prob_lab))
            plt.savefig(str(out_path), bbox_inches='tight', dpi=dpi)
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        for cp in self.cp_freqs_ser.index:
            idxs = self.sel_cps_ser == cp

            if not idxs.sum():
                continue

            cp_ser = self.sel_cps_ser.loc[idxs]
            cp_months = cp_ser.index.month
            months, months_freq = np.unique(cp_months, return_counts=True)
            _rng = np.arange(months.shape[0])
            ax.bar(_rng, months_freq)
            ax.set_xticks(_rng)
            ax.set_xticklabels(months)

            ax.set_xlabel('Month')
            ax.set_ylabel('CP Frequency')
            ax.set_title('Monthly Frequency of CP:%2d (N=%d)' %
                         (cp, self.cp_freqs_ser.loc[cp]))
            ax.grid()

            out_path = out_fig_dir / (fig_name_suff +
                                      ('_cp_%0.2d_monthly_freq.png' % cp))
            plt.savefig(str(out_path), bbox_inches='tight', dpi=dpi)
            ax.cla()

        plt.close('all')
        return
