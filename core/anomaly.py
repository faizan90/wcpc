'''
Created on Dec 29, 2017

@author: Faizan
'''
from itertools import product as iprod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..misc.error_msgs import print_warning
from ..misc.checks import check_nans_finite, check_nats


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
        
        # TODO: implement for other resolutions
        assert self.time_int == 'D'

        # TODO: implement for other types
        assert self.file_type == self.nc_ext

        return

    def _read_nc(self):
        in_ds = xr.open_dataset(str(self.in_ds_path))

        _ = pd.DatetimeIndex(getattr(in_ds, self.time_dim_lab).values)
        self.times_tot = pd.DatetimeIndex(_)

        assert not np.sum(self.times_tot.duplicated())

        assert not check_nats(self.times_tot)

        self.x_coords = getattr(in_ds, self.x_dim_lab).values
        self.y_coords = getattr(in_ds, self.y_dim_lab).values

        assert self.x_coords.shape[0]
        assert self.y_coords.shape[0]

        assert len(self.x_coords.shape) == 1
        assert len(self.y_coords.shape) == 1

        assert check_nans_finite(self.x_coords)
        assert check_nans_finite(self.y_coords)

        self.x_coords_mesh, self.y_coords_mesh = np.meshgrid(self.x_coords,
                                                             self.y_coords)

        self.x_coords_rav = self.x_coords_mesh.ravel()
        self.y_coords_rav = self.y_coords_mesh.ravel()

        self.vals_tot = getattr(in_ds, self.vals_dim_lab).values
        self.n_timesteps_tot = self.vals_tot.shape[0]

        self.vals_tot_rav = self.vals_tot.reshape(self.n_timesteps_tot, -1)
        assert len(self.vals_tot_rav.shape) == 2
        
        self.n_tot_vals = (self.vals_tot_rav.shape[0] * 
                           self.vals_tot_rav.shape[1])

        self.min_vals_tot_rav = np.nanmin(self.vals_tot_rav, axis=1)
        self.max_vals_tot_rav = np.nanmax(self.vals_tot_rav, axis=1)

        _nan_idxs = np.isnan(self.vals_tot_rav)
        nan_ct = np.sum(_nan_idxs)
        _msg = '%d NaN(s) out of %d input values.' % (nan_ct, self.n_tot_vals)

        if nan_ct:
            if self.msgs:
                print_warning(('\nWarning in read_vars: %s'
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

                _rows_seq = np.repeat(np.arange(_back_row, _fore_row + 1),
                                      _fore_row - _back_row)
                _cols_seq = np.tile(np.arange(_back_col, _fore_col + 1),
                                    _fore_col - _back_col)

                _row_col_seq = list(iprod(np.arange(_back_row, _fore_row + 1),
                                          np.arange(_back_col, _fore_col + 1)))
                _row_col_seq = np.array(_row_col_seq, dtype=int)

                self.vals_tot_rav[curr_row, curr_col] = \
                    np.nanmean(self.vals_tot_rav[_row_col_seq[:, 0],
                                                 _row_col_seq[:, 1]])
                    
        return

    def read_vars(self,
                  in_ds_path,
                  x_dim_lab='lon',
                  y_dim_lab='lat',
                  time_dim_lab='time',
                  vals_dim_lab='slp',
                  file_type='nc',
                  time_int='D'):

        self.in_ds_path = Path(in_ds_path)

        self.x_dim_lab = x_dim_lab
        self.y_dim_lab = y_dim_lab
        self.vals_dim_lab = vals_dim_lab
        self.time_dim_lab = time_dim_lab
        self.time_int = time_int

        self.file_type = file_type

        self._verify_input()
        
        if self.file_type == self.nc_ext:
            self._read_nc()

        self._vars_read_flag = True
        return
        
    def calc_anomaly_type_a(self, anom_type_a_nan_rep=None):
        assert self._vars_read_flag
        
        if anom_type_a_nan_rep is not None:
            assert isinstance(anom_type_a_nan_rep, (int, float))
            
        self.anom_type_a_nan_rep = anom_type_a_nan_rep

        _1 = self.vals_tot_rav - self.min_vals_tot_rav[:, None]
        _2 = (self.max_vals_tot_rav - self.min_vals_tot_rav)[:, None]

        self.vals_tot_anom = _1 / _2

        assert len(self.vals_tot_rav.shape) == len(self.vals_tot_anom.shape)

        nan_ct = np.sum(np.isnan(self.vals_tot_anom))
        _msg = '%d NaNs out of %d in anomaly of type A.' % (nan_ct,
                                                            self.n_tot_vals)

        if self.anom_type_a_nan_rep is None:
            assert not nan_ct, _msg
        elif nan_ct:
            if self.msgs:
                print_warning(('\nWarning in calc_anomaly_type_a: %s'
                               ' Setting all to %s') %
                               (_msg, str(self.anom_type_a_nan_rep)))

            _ = np.isnan(self.vals_tot_anom)
            self.vals_tot_anom[_] = self.anom_type_a_nan_rep
        return

    def calc_anomaly_type_b(self, anom_type_b_nan_rep=None):
        assert self._vars_read_flag

        if anom_type_b_nan_rep is not None:
            assert isinstance(anom_type_b_nan_rep, (int, float))

        self.anom_type_b_nan_rep = anom_type_b_nan_rep

        self.mean_arr = np.mean(self.vals_tot_rav, axis=0)
        self.sigma_arr = np.std(self.vals_tot_rav, axis=0)

        self.vals_tot_anom = ((self.vals_tot_rav - self.mean_arr) /
                                  self.sigma_arr)

        _anom_min = np.nanmin(self.vals_tot_anom, axis=1)
        _anom_max = np.nanmax(self.vals_tot_anom, axis=1)

        _1 = self.vals_tot_anom - _anom_min[:, None]
        _2 = (_anom_max - _anom_min)[:, None]

        self.vals_tot_anom = _1 / _2

        assert len(self.vals_tot_rav.shape) == len(self.vals_tot_anom.shape)

        nan_ct = np.sum(np.isnan(self.vals_tot_anom))
        _msg = '%d NaNs out of %d in anomaly of type B.' % (nan_ct,
                                                            self.n_tot_vals)

        if self.anom_type_b_nan_rep is None:
            assert not nan_ct, _msg
        elif nan_ct:
            if self.msgs:
                print_warning(('\nWarning in calc_anomaly_type_b: %s'
                               ' Setting all to %s') %
                               (_msg, str(self.anom_type_b_nan_rep)))

            _ = np.isnan(self.vals_tot_anom)
            self.vals_tot_anom[_] = self.anom_type_b_nan_rep
        return

    def calc_anomaly_type_c(self, anom_type_c_nan_rep=None):
        assert self._vars_read_flag

        if anom_type_c_nan_rep is not None:
            assert isinstance(anom_type_c_nan_rep, (int, float))

        self.anom_type_c_nan_rep = anom_type_c_nan_rep

        self.vals_tot_anom = np.full_like(self.vals_tot_rav, np.nan)

        for i in range(12):
            m_idxs = self.times_tot.month == (i + 1)
            for j in range(31):
                d_idxs = m_idxs & (self.times_tot.day == (j + 1))
                
                if not d_idxs.sum():
                    continue

                curr_vals = self.vals_tot_rav[d_idxs]

                mean_arr = np.mean(curr_vals, axis=0)
                sigma_arr = np.std(curr_vals, axis=0)

                curr_anoms = (curr_vals - mean_arr) / sigma_arr
                self.vals_tot_anom[d_idxs] = curr_anoms

        _anom_min = np.nanmin(self.vals_tot_anom, axis=1)
        _anom_max = np.nanmax(self.vals_tot_anom, axis=1)

        _1 = self.vals_tot_anom - _anom_min[:, None]
        _2 = (_anom_max - _anom_min)[:, None]

        self.vals_tot_anom = _1 / _2

        assert len(self.vals_tot_rav.shape) == len(self.vals_tot_anom.shape)

        nan_ct = np.sum(np.isnan(self.vals_tot_anom))
        _msg = '%d NaNs out of %d in anomaly of type B.' % (nan_ct,
                                                            self.n_tot_vals)

        if self.anom_type_c_nan_rep is None:
            assert not nan_ct, _msg
        elif nan_ct:
            if self.msgs:
                print_warning(('\nWarning in calc_anomaly_type_b: %s'
                               ' Setting all to %s') %
                               (_msg, str(self.anom_type_c_nan_rep)))

            _ = np.isnan(self.vals_tot_anom)
            self.vals_tot_anom[_] = self.anom_type_c_nan_rep
        return
