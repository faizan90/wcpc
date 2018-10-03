'''
Created on Dec 29, 2017

@author: Faizan
'''
from itertools import product as iprod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool

from ..misc.error_msgs import print_warning
from ..misc.checks import check_nans_finite, check_nats
from ..misc.ftns import ret_mp_idxs

plt.ioff()


class Anomaly:

    def __init__(self, msgs=True):

        assert isinstance(msgs, (bool, int, float))
        self.msgs = msgs

        self._vars_read_flag = False
        self.nc_ext = 'nc'
        return

    def _verify_input(self):

        assert self.in_ds_path.is_file()

        assert isinstance(self.x_dim_lab, str)
        assert isinstance(self.y_dim_lab, str)
        assert isinstance(self.vals_dim_lab, str)
        assert isinstance(self.time_dim_lab, str)
        assert isinstance(self.time_int, str)
        assert isinstance(self.file_type, str)
        assert isinstance(self.sub_daily_flag, bool)

        # TODO: implement for other resolutions
        assert self.time_int == 'D'

        # TODO: implement for other types
        assert self.file_type == self.nc_ext

        return

    def _read_nc(self):

        in_ds = xr.open_dataset(str(self.in_ds_path))

        _ = pd.DatetimeIndex(getattr(in_ds, self.time_dim_lab).values)
        self.times_tot = pd.DatetimeIndex(_)

        if not self.sub_daily_flag:
            self.times_tot = pd.DatetimeIndex(self.times_tot.date)

        assert not np.sum(self.times_tot.duplicated())

        assert not check_nats(self.times_tot)

        self.x_coords = getattr(in_ds, self.x_dim_lab).values
        self.y_coords = getattr(in_ds, self.y_dim_lab).values

        assert self.x_coords.shape[0]
        assert self.y_coords.shape[0]

        assert self.x_coords.ndim == 1
        assert self.y_coords.ndim == 1

        assert check_nans_finite(self.x_coords)
        assert check_nans_finite(self.y_coords)

        self.x_coords_mesh, self.y_coords_mesh = np.meshgrid(
            self.x_coords, self.y_coords)

        self.x_coords_rav = self.x_coords_mesh.ravel()
        self.y_coords_rav = self.y_coords_mesh.ravel()

        self.vals_tot = getattr(in_ds, self.vals_dim_lab).values
        self.n_timesteps_tot = self.vals_tot.shape[0]

        self.vals_tot_rav = self.vals_tot.reshape(self.n_timesteps_tot, -1)
        assert self.vals_tot_rav.ndim == 2

        self.n_tot_vals = (
            self.vals_tot_rav.shape[0] * self.vals_tot_rav.shape[1])

        self.min_vals_tot_rav = np.nanmin(self.vals_tot_rav, axis=1)
        self.max_vals_tot_rav = np.nanmax(self.vals_tot_rav, axis=1)

        _nan_idxs = np.isnan(self.vals_tot_rav)
        nan_ct = np.sum(_nan_idxs)
        _msg = '%d NaN(s) out of %d input values.' % (nan_ct, self.n_tot_vals)

        if nan_ct:
            if self.msgs:
                print_warning((
                    '\nWarning in read_vars: %s'
                    ' Setting all to the mean of the D8 at the '
                    'given time steps.') % (_msg))

            nan_rows, nan_cols = np.where(_nan_idxs)
            max_row = self.vals_tot_rav.shape[0] - 1
            max_col = self.vals_tot_rav.shape[1] - 1

            for i in range(nan_rows.shape[0]):
                curr_row = nan_rows[i]
                curr_col = nan_cols[i]

                _back_row = curr_row
                _back_col = curr_col
                _fore_row = curr_row
                _fore_col = curr_col

                if _back_row:
                    _back_row -= 1

                if _back_col:
                    _back_col -= 1

                if _fore_row < max_row:
                    _fore_row += 1

                if _fore_col < max_col:
                    _fore_col += 1

                _rows_seq = np.repeat(
                    np.arange(_back_row, _fore_row + 1),
                    _fore_row - _back_row)
                _cols_seq = np.tile(
                    np.arange(_back_col, _fore_col + 1),
                    _fore_col - _back_col)

                _row_col_seq = list(iprod(
                    np.arange(_back_row, _fore_row + 1),
                    np.arange(_back_col, _fore_col + 1)))

                _row_col_seq = np.array(_row_col_seq, dtype=int)

                self.vals_tot_rav[curr_row, curr_col] = (
                    np.nanmean(
                        self.vals_tot_rav[
                            _row_col_seq[:, 0], _row_col_seq[:, 1]]))

        return

    def read_vars(
            self,
            in_ds_path,
            x_dim_lab='lon',
            y_dim_lab='lat',
            time_dim_lab='time',
            vals_dim_lab='slp',
            file_type='nc',
            time_int='D',
            sub_daily_flag=False):

        self.in_ds_path = Path(in_ds_path)

        self.x_dim_lab = x_dim_lab
        self.y_dim_lab = y_dim_lab
        self.vals_dim_lab = vals_dim_lab
        self.time_dim_lab = time_dim_lab
        self.time_int = time_int
        self.sub_daily_flag = sub_daily_flag

        self.file_type = file_type

        self._verify_input()

        if self.file_type == self.nc_ext:
            self._read_nc()

        else:
            raise ValueError('Only configure to read netCDF4 files!')

        self._vars_read_flag = True
        return

    def get_time_range_idxs(
            self,
            strt_time,
            end_time,
            season_months,
            time_fmt):

        strt_time, end_time = pd.to_datetime(
            [strt_time, end_time], format=time_fmt)

        assert strt_time < end_time, (strt_time, end_time)
        assert strt_time >= self.times_tot[0], (strt_time, self.times_tot[0])
        assert end_time <= self.times_tot[-1], (end_time, self.times_tot[-1])

        # just in case
        assert (self.times_tot.shape[0] == self.vals_tot_rav.shape[0])

        curr_idxs = (
            (self.times_tot >= strt_time) & ((self.times_tot <= end_time)))

        month_idxs = np.zeros(self.times_tot.shape[0], dtype=bool)

        for month in season_months:
            month_idxs = month_idxs | (self.times_tot.month == month)

        curr_idxs = curr_idxs & month_idxs

        assert curr_idxs.sum()
        return curr_idxs

    def calc_anomaly_type_a(self, anom_type_a_nan_rep=None):

        assert self._vars_read_flag
        raise NotImplementedError('Dont use it!')

        if anom_type_a_nan_rep is not None:
            assert isinstance(anom_type_a_nan_rep, (int, float))

        self.anom_type_a_nan_rep = anom_type_a_nan_rep

        _1 = self.vals_tot_rav - self.min_vals_tot_rav[:, None]
        _2 = (self.max_vals_tot_rav - self.min_vals_tot_rav)[:, None]

        self.vals_tot_anom = _1 / _2

        assert self.vals_tot_rav.ndim == self.vals_tot_anom.ndim

        nan_ct = np.sum(np.isnan(self.vals_tot_anom))
        _msg = '%d NaNs out of %d in anomaly of type A.' % (
            nan_ct, self.n_tot_vals)

        if self.anom_type_a_nan_rep is None:
            assert not nan_ct, _msg
        elif nan_ct:
            if self.msgs:
                print_warning((
                    '\nWarning in calc_anomaly_type_a: %s'
                    ' Setting all to %s') %
                    (_msg, str(self.anom_type_a_nan_rep)))

            _ = np.isnan(self.vals_tot_anom)
            self.vals_tot_anom[_] = self.anom_type_a_nan_rep
        return

    def calc_anomaly_type_b(
            self,
            strt_time,
            end_time,
            season_months,
            time_fmt='%Y-%m-%d',
            anom_type_b_nan_rep=None,
            fig_out_dir=None,
            n_cpus=1,
            normalize=True):

        assert self._vars_read_flag

        assert isinstance(strt_time, str)
        assert isinstance(end_time, str)
        assert isinstance(time_fmt, str)
        assert isinstance(n_cpus, int)
        assert n_cpus > 0

        assert isinstance(season_months, np.ndarray)
        assert check_nans_finite(season_months)
        assert season_months.ndim == 1
        assert np.all(season_months > 0) and (np.all(season_months < 13))
        assert season_months.shape[0] > 0

        if fig_out_dir is not None:
            assert isinstance(fig_out_dir, (Path, str))

            fig_out_dir = Path(fig_out_dir)
            assert fig_out_dir.parents[0].exists()

            if not fig_out_dir.exists():
                fig_out_dir.mkdir()

        if anom_type_b_nan_rep is not None:
            assert isinstance(anom_type_b_nan_rep, (int, float))

        self.anom_type_b_nan_rep = anom_type_b_nan_rep

        curr_time_idxs = self.get_time_range_idxs(
            strt_time, end_time, season_months, time_fmt)

        self.times = self.times_tot[curr_time_idxs]

        self.mean_arr = np.mean(self.vals_tot_rav[curr_time_idxs], axis=0)
        self.sigma_arr = np.std(self.vals_tot_rav[curr_time_idxs], axis=0)

        self.vals_tot_anom = (
            (self.vals_tot_rav[curr_time_idxs] - self.mean_arr) /
            self.sigma_arr)

        if normalize:
            _anom_min = np.nanmin(self.vals_tot_anom, axis=1)
            _anom_max = np.nanmax(self.vals_tot_anom, axis=1)

            _1 = self.vals_tot_anom - _anom_min[:, None]
            _2 = (_anom_max - _anom_min)[:, None]

            self.vals_tot_anom = _1 / _2

#         for i in range(self.vals_tot_anom.shape[1]):
#             curr_bjs_probs_arr = (
#                 (np.argsort(np.argsort(self.vals_tot_anom[:, i])) + 1) /
#                 (self.vals_tot_anom.shape[0] + 1))
#             self.vals_tot_anom[:, i] = curr_bjs_probs_arr

        assert (
            curr_time_idxs.sum() ==
            self.vals_tot_anom.shape[0] ==
            self.times.shape[0])

        nan_ct = np.sum(np.isnan(self.vals_tot_anom))
        _msg = '%d NaNs out of %d in anomaly of type B.' % (
            nan_ct, self.n_tot_vals)

        if self.anom_type_b_nan_rep is None:
            assert not nan_ct, _msg

        elif nan_ct:
            if self.msgs:
                print_warning((
                    '\nWarning in calc_anomaly_type_b: %s'
                    ' Setting all to %s') %
                    (_msg, str(self.anom_type_b_nan_rep)))

            _ = np.isnan(self.vals_tot_anom)
            self.vals_tot_anom[_] = self.anom_type_b_nan_rep

        if fig_out_dir is not None:
            if self.msgs:
                print('Saving anomaly CDF figs in:', fig_out_dir)
            self._prep_anomaly_mp(self.vals_tot_anom, n_cpus, fig_out_dir)
        return

    def calc_anomaly_type_c(
            self,
            strt_time,
            end_time,
            season_months,
            time_fmt='%Y-%m-%d',
            anom_type_c_nan_rep=None,
            fig_out_dir=None,
            n_cpus=1,
            normalize=True):

        assert self._vars_read_flag

        if anom_type_c_nan_rep is not None:
            assert isinstance(anom_type_c_nan_rep, (int, float))

        assert isinstance(strt_time, str)
        assert isinstance(end_time, str)
        assert isinstance(time_fmt, str)

        assert isinstance(n_cpus, int)
        assert n_cpus > 0

        assert isinstance(season_months, np.ndarray)
        assert check_nans_finite(season_months)
        assert season_months.ndim == 1
        assert season_months.shape[0] > 0
        assert np.all(season_months > 0) and (np.all(season_months < 13))

        if fig_out_dir is not None:
            assert isinstance(fig_out_dir, (Path, str))

            fig_out_dir = Path(fig_out_dir)
            assert fig_out_dir.parents[0].exists()

            if not fig_out_dir.exists():
                fig_out_dir.mkdir()

        self.anom_type_c_nan_rep = anom_type_c_nan_rep

        self.vals_tot_anom = np.full_like(self.vals_tot_rav, np.nan)

        curr_time_idxs = self.get_time_range_idxs(
            strt_time, end_time, season_months, time_fmt)

        self.times = self.times_tot[curr_time_idxs]

        if self.sub_daily_flag:

            unique_hrs = np.unique(self.times_tot.hour)

            if self.msgs:
                print('Unique hours for sub-daily anomaly:', unique_hrs)

            for i in range(12):
                m_idxs = self.times_tot.month == (i + 1)

                if not m_idxs.sum():
                    continue

                for j in range(31):
                    d_idxs = (
                        m_idxs &
                        (self.times_tot.day == (j + 1)) &
                        curr_time_idxs)

                    if not d_idxs.sum():
                        continue

                    for k in unique_hrs:
                        h_idxs = d_idxs & (self.times_tot.hour == k)

                        if not h_idxs.sum():
                            continue

                        curr_vals = self.vals_tot_rav[h_idxs]

                        mean_arr = np.mean(curr_vals, axis=0)
                        sigma_arr = np.std(curr_vals, axis=0)

                        curr_anoms = (curr_vals - mean_arr) / sigma_arr
                        self.vals_tot_anom[h_idxs] = curr_anoms

        else:
            for i in range(12):
                m_idxs = self.times_tot.month == (i + 1)
                for j in range(31):
                    d_idxs = (m_idxs &
                              (self.times_tot.day == (j + 1)) &
                              curr_time_idxs)

                    if not d_idxs.sum():
                        continue

                    curr_vals = self.vals_tot_rav[d_idxs]

                    mean_arr = np.mean(curr_vals, axis=0)
                    sigma_arr = np.std(curr_vals, axis=0)

                    curr_anoms = (curr_vals - mean_arr) / sigma_arr
                    self.vals_tot_anom[d_idxs] = curr_anoms

        self.vals_tot_anom = self.vals_tot_anom[curr_time_idxs]

        if normalize:
            _anom_min = np.nanmin(self.vals_tot_anom, axis=1)
            _anom_max = np.nanmax(self.vals_tot_anom, axis=1)

            _1 = self.vals_tot_anom - _anom_min[:, None]
            _2 = (_anom_max - _anom_min)[:, None]

            self.vals_tot_anom = _1 / _2

        nan_ct = np.sum(np.isnan(self.vals_tot_anom))
        _msg = '%d NaNs out of %d in anomaly of type C.' % (
            nan_ct, self.n_tot_vals)

        if self.anom_type_c_nan_rep is None:
            assert not nan_ct, _msg

        elif nan_ct:
            if self.msgs:
                print_warning((
                    '\nWarning in calc_anomaly_type_c: %s'
                    ' Setting all to %s') %
                    (_msg, str(self.anom_type_c_nan_rep)))

            _ = np.isnan(self.vals_tot_anom)
            self.vals_tot_anom[_] = self.anom_type_c_nan_rep

        if fig_out_dir is not None:
            if self.msgs:
                print('Saving anomaly CDF figs in:', fig_out_dir)
            self._prep_anomaly_mp(self.vals_tot_anom, n_cpus, fig_out_dir)
        return

    def calc_anomaly_type_d(
            self,
            strt_time,
            end_time,
            strt_time_all,
            end_time_all,
            season_months,
            time_fmt='%Y-%m-%d',
            anom_type_d_nan_rep=None,
            eig_cum_sum_ratio=0.95,
            eig_sum_flag=False,
            fig_out_dir=None,
            n_cpus=1,
            normalize=True):

        assert self._vars_read_flag

        if anom_type_d_nan_rep is not None:
            assert isinstance(anom_type_d_nan_rep, (int, float))

        assert isinstance(strt_time, str)
        assert isinstance(end_time, str)

        assert isinstance(strt_time_all, str)
        assert isinstance(end_time_all, str)

        assert isinstance(time_fmt, str)

        assert isinstance(n_cpus, int)
        assert n_cpus > 0

        assert isinstance(eig_cum_sum_ratio, float)
        assert 0 < eig_cum_sum_ratio <= 1

        assert isinstance(season_months, np.ndarray)
        assert check_nans_finite(season_months)
        assert season_months.ndim == 1
        assert np.all(season_months > 0) and (np.all(season_months < 13))
        assert season_months.shape[0] > 0

        if fig_out_dir is not None:
            assert isinstance(fig_out_dir, (Path, str))

            fig_out_dir = Path(fig_out_dir)
            assert fig_out_dir.parents[0].exists()

            if not fig_out_dir.exists():
                fig_out_dir.mkdir()

        self.anom_type_d_nan_rep = anom_type_d_nan_rep

        self.calc_anomaly_type_b(
            strt_time_all,
            end_time_all,
            season_months,
            time_fmt,
            self.anom_type_d_nan_rep,
            normalize=normalize)

        corr_mat = np.corrcoef(self.vals_tot_anom.T)
        eig_val, eig_mat = np.linalg.eig(corr_mat)
        sort_idxs = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sort_idxs]
        eig_mat = eig_mat[:, sort_idxs]
        eig_val_sum = eig_val.sum()
        self.eig_val_cum_sum_arr = np.cumsum(eig_val) / eig_val_sum

        _idxs = self.eig_val_cum_sum_arr <= eig_cum_sum_ratio

        self.n_dims = _idxs.sum()
        assert self.n_dims

        if eig_sum_flag:
            self.n_dims += 1

        curr_time_all_idxs = self.get_time_range_idxs(
            strt_time_all, end_time_all, season_months, time_fmt)

        curr_time_idxs = self.get_time_range_idxs(
            strt_time, end_time, season_months, time_fmt)[curr_time_all_idxs]

        self.times = self.times[curr_time_idxs]
        b_j_s = np.dot(self.vals_tot_anom, eig_mat)[curr_time_idxs]
        self.vals_anom_for_cp_plots = self.vals_tot_anom[curr_time_idxs]

        if eig_sum_flag:
            b_j_s[:, self.n_dims - 1] = (
                (b_j_s[:, self.n_dims - 1:] ** 2).sum(axis=1))

        b_j_s = b_j_s[:, :self.n_dims]

        assert check_nans_finite(b_j_s)

        if normalize:
            self.vals_anom = np.full(
            (curr_time_idxs.sum(), self.n_dims), np.nan)

            for i in range(self.n_dims):
                curr_bjs_arr = b_j_s[:, i]
                curr_bjs_probs_arr = (
                    (np.argsort(np.argsort(curr_bjs_arr)) + 1) /
                    (curr_bjs_arr.shape[0] + 1))

                self.vals_anom[:, i] = curr_bjs_probs_arr
    #             curr_bjs_arr = ((curr_bjs_arr - curr_bjs_arr.min()) /
    #                             (curr_bjs_arr.max() - curr_bjs_arr.min()))
#             self.vals_anom[:, i] = curr_bjs_arr
        else:
            self.vals_anom = b_j_s

        assert check_nans_finite(self.vals_anom)
        assert (
            curr_time_idxs.sum() ==
            self.vals_anom.shape[0] ==
            self.times.shape[0])

        if normalize:
            assert (np.all(self.vals_anom > 0) and np.all(self.vals_anom < 1))

        if fig_out_dir is not None:
            if self.msgs:
                print('Saving anomaly and bjs CDF figs in:', fig_out_dir)
            self._prep_anomaly_bjs_mp(
                self.vals_anom, b_j_s, n_cpus, fig_out_dir)
        return

    def calc_anomaly_type_e(
            self,
            strt_time,
            end_time,
            strt_time_all,
            end_time_all,
            season_months,
            time_fmt='%Y-%m-%d',
            anom_type_e_nan_rep=None,
            eig_cum_sum_ratio=0.95,
            eig_sum_flag=False,
            fig_out_dir=None,
            n_cpus=1,
            normalize=True):

        assert self._vars_read_flag

        if anom_type_e_nan_rep is not None:
            assert isinstance(anom_type_e_nan_rep, (int, float))

        assert isinstance(strt_time, str)
        assert isinstance(end_time, str)

        assert isinstance(strt_time_all, str)
        assert isinstance(end_time_all, str)

        assert isinstance(time_fmt, str)

        assert isinstance(n_cpus, int)
        assert n_cpus > 0

        assert isinstance(eig_cum_sum_ratio, float)
        assert 0 < eig_cum_sum_ratio <= 1

        assert isinstance(season_months, np.ndarray)
        assert check_nans_finite(season_months)
        assert season_months.ndim == 1
        assert np.all(season_months > 0) and (np.all(season_months < 13))
        assert season_months.shape[0] > 0

        if fig_out_dir is not None:
            assert isinstance(fig_out_dir, (Path, str))

            fig_out_dir = Path(fig_out_dir)
            assert fig_out_dir.parents[0].exists()

            if not fig_out_dir.exists():
                fig_out_dir.mkdir()

        self.anom_type_e_nan_rep = anom_type_e_nan_rep

        self.calc_anomaly_type_c(
            strt_time_all,
            end_time_all,
            season_months,
            time_fmt,
            self.anom_type_e_nan_rep,
            fig_out_dir,
            n_cpus,
            normalize=normalize)

        corr_mat = np.corrcoef(self.vals_tot_anom.T)
        eig_val, eig_mat = np.linalg.eig(corr_mat)
        sort_idxs = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sort_idxs]
        eig_mat = eig_mat[:, sort_idxs]
        eig_val_sum = eig_val.sum()

        self.eig_val_cum_sum_arr = np.cumsum(eig_val) / eig_val_sum
        self.eig_vecs_mat = eig_mat

        _idxs = self.eig_val_cum_sum_arr <= eig_cum_sum_ratio
        self.n_dims = _idxs.sum()

        assert self.n_dims

        if eig_sum_flag:
            self.n_dims += 1

        curr_time_all_idxs = self.get_time_range_idxs(
            strt_time_all, end_time_all, season_months, time_fmt)

        curr_time_idxs = self.get_time_range_idxs(
            strt_time, end_time, season_months, time_fmt)[curr_time_all_idxs]

        self.times = self.times[curr_time_idxs]
        b_j_s = np.dot(self.vals_tot_anom, eig_mat)[curr_time_idxs]
        self.vals_anom_for_cp_plots = self.vals_tot_anom[curr_time_idxs]

        if eig_sum_flag:
            b_j_s[:, self.n_dims - 1] = (
                (b_j_s[:, self.n_dims - 1:] ** 2).sum(axis=1))

        b_j_s = b_j_s[:, :self.n_dims]

        assert check_nans_finite(b_j_s)

        if normalize:
            self.vals_anom = np.full(
                (curr_time_idxs.sum(), self.n_dims), np.nan)

            for i in range(self.n_dims):
                curr_bjs_arr = b_j_s[:, i]
                curr_bjs_probs_arr = (
                    (np.argsort(np.argsort(curr_bjs_arr)) + 1) /
                    (curr_bjs_arr.shape[0] + 1))

                self.vals_anom[:, i] = curr_bjs_probs_arr
    #             curr_bjs_arr = ((curr_bjs_arr - curr_bjs_arr.min()) /
    #                             (curr_bjs_arr.max() - curr_bjs_arr.min()))
    #             self.vals_anom[:, i] = curr_bjs_arr
        else:
            self.vals_anom = b_j_s

        assert check_nans_finite(self.vals_anom)
        assert (
            curr_time_idxs.sum() ==
            self.vals_anom.shape[0] ==
            self.times.shape[0])

        if normalize:
            assert (np.all(self.vals_anom > 0) and np.all(self.vals_anom < 1))

        if fig_out_dir is not None:
            if self.msgs:
                print('Saving anomaly and bjs CDF figs in:', fig_out_dir)
            self._prep_anomaly_bjs_mp(
                self.vals_anom, b_j_s, n_cpus, fig_out_dir)
        return

    @staticmethod
    def _plot_anomaly_cdf(dims_idxs, anoms_arr, fig_out_dir):

        plt.figure(figsize=(10, 7))

        top_plot = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        bot_plot = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

        assert (dims_idxs[1] - dims_idxs[0]) == anoms_arr.shape[1]

        for i, dim_idx in enumerate(range(dims_idxs[0], dims_idxs[1], 1)):
            curr_anoms_arr = anoms_arr[:, i].copy()
            curr_anoms_arr.sort()
            curr_anoms_probs_arr = (
                (np.arange(1, curr_anoms_arr.shape[0] + 1)) /
                (curr_anoms_arr.shape[0] + 1))

            top_plot.plot(curr_anoms_arr,
                          curr_anoms_probs_arr,
                          alpha=0.5,
                          marker='o',
                          markersize=3)

            top_plot.set_xticklabels([])
            top_plot.set_ylim(-0.05, 1.05)
            top_plot.set_ylabel('Probability')
            top_plot.grid()
            top_plot.get_xaxis().set_tick_params(width=0)

            bot_plot.hist(
                curr_anoms_arr, bins=20, alpha=0.7)
            bot_plot.set_xlim(*top_plot.get_xlim())
            bot_plot.set_xlabel('Anomaly')
            bot_plot.set_ylabel('Frequency')
            bot_plot.grid()

            top_plot.set_title(
                'Anomaly distribution (D=%d, n=%d)' %
                (dim_idx + 1, curr_anoms_arr.shape[0]))

            plt.savefig(
                str(fig_out_dir / ('anomaly_cdf_%d.png' % (dim_idx + 1))),
                bbox_inches='tight')

            top_plot.cla()
            bot_plot.cla()

        plt.close()
        return

    @staticmethod
    def _plot_anomaly_bjs_cdf(dims_idxs, anoms_arr, bjs_arr, fig_out_dir):

        plt.figure(figsize=(10, 7))

        top_plot = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        bot_plot = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

        assert (dims_idxs[1] - dims_idxs[0]) == anoms_arr.shape[1]

        for i, dim_idx in enumerate(range(dims_idxs[0], dims_idxs[1], 1)):
            # anomaly
            curr_anoms_arr = anoms_arr[:, i].copy()
            curr_anoms_arr.sort()
            curr_anoms_probs_arr = (
                (np.arange(1, curr_anoms_arr.shape[0] + 1)) /
                (curr_anoms_arr.shape[0] + 1))

            top_plot.plot(
                curr_anoms_arr,
                curr_anoms_probs_arr,
                alpha=0.5,
                marker='o',
                markersize=3)

            top_plot.set_xticklabels([])
            top_plot.set_ylim(-0.05, 1.05)
            top_plot.set_ylabel('Probability')
            top_plot.grid()
            top_plot.get_xaxis().set_tick_params(width=0)

            bot_plot.hist(
                curr_anoms_arr, bins=20, alpha=0.7)
            bot_plot.set_xlim(*top_plot.get_xlim())
            bot_plot.set_xlabel('Anomaly')
            bot_plot.set_ylabel('Frequency')
            bot_plot.grid()

            top_plot.set_title(
                'Anomaly distribution (D=%d, n=%d)' %
                (dim_idx + 1, curr_anoms_arr.shape[0]))

            plt.savefig(
                str(fig_out_dir / ('anomaly_cdf_%d.png' % (dim_idx + 1))),
                bbox_inches='tight')

            top_plot.cla()
            bot_plot.cla()

            # bjs
            curr_bjs_arr = bjs_arr[:, i].copy()
            curr_bjs_arr.sort()
            curr_bjs_probs_arr = (
                (np.arange(1, curr_bjs_arr.shape[0] + 1)) /
                (curr_bjs_arr.shape[0] + 1))

            top_plot.plot(
                curr_bjs_arr,
                curr_bjs_probs_arr,
                alpha=0.5,
                marker='o',
                markersize=3)

            top_plot.set_xticklabels([])
            top_plot.set_ylim(-0.05, 1.05)
            top_plot.set_ylabel('Probability')
            top_plot.grid()
            top_plot.get_xaxis().set_tick_params(width=0)

            bot_plot.hist(
                curr_bjs_arr, bins=20, alpha=0.7)
            bot_plot.set_xlim(*top_plot.get_xlim())
            bot_plot.set_xlabel('bj')
            bot_plot.set_ylabel('Frequency')
            bot_plot.grid()

            top_plot.set_title(
                'bjs distribution (D=%d, n=%d)' %
                (dim_idx + 1, curr_bjs_arr.shape[0]))

            plt.savefig(
                str(fig_out_dir / ('bjs_cdf_%d.png' % (dim_idx + 1))),
                bbox_inches='tight')

            top_plot.cla()
            bot_plot.cla()

        plt.close()
        return

    @staticmethod
    def _prep_anomaly_mp(anoms_arr, n_cpus, fig_out_dir):

        _idxs = ret_mp_idxs(anoms_arr.shape[1], n_cpus)
        _idxs_list = [_idxs[i: i + 2] for i in range(n_cpus)]

        _anoms_gen = (
            (anoms_arr[:, _idxs_list[i][0]:_idxs_list[i][1]])
            for i in range(n_cpus))

        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(
                Anomaly._plot_anomaly_cdf,
                _idxs_list,
                _anoms_gen,
                [fig_out_dir] * n_cpus)))

            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in _plot_anomaly_cdf:', msg)
        return

    @staticmethod
    def _prep_anomaly_bjs_mp(anoms_arr, bjs_arr, n_cpus, fig_out_dir):

        assert anoms_arr.shape == bjs_arr.shape

        _idxs = ret_mp_idxs(anoms_arr.shape[1], n_cpus)
        _idxs_list = [_idxs[i: i + 2] for i in range(n_cpus)]

        _anoms_gen = (
            (anoms_arr[:, _idxs_list[i][0]:_idxs_list[i][1]])
            for i in range(n_cpus))

        _bjs_gen = (
            (bjs_arr[:, _idxs_list[i][0]:_idxs_list[i][1]])
            for i in range(n_cpus))

        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(
                Anomaly._plot_anomaly_bjs_cdf,
                _idxs_list,
                _anoms_gen,
                _bjs_gen,
                [fig_out_dir] * n_cpus)))

            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in _plot_anomaly_bjs_cdf:', msg)
        return


if __name__ == '__main__':
    pass
