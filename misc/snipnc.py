'''
Created on Dec 29, 2017

@author: Faizan
'''
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .error_msgs import print_warning


class SnipNC:

    '''
    For given time and area bounds, snip data from a netcdf file

    Designed for daily data recorded at different times over a day
    '''

    def __init__(self):

        self._paths_set_flag = False
        self._coords_set_flag = False
        self._times_set_flag = False
        self._snip_flag = False
        return

    def set_paths(self, in_nc_path, out_nc_path):

        '''Set input and output netcdf paths'''

        self.in_nc_path = Path(in_nc_path)
        self.out_nc_path = Path(out_nc_path)

        self._verify_paths()
        self._paths_set_flag = True
        return

    def set_coords(
            self,
            x_min,
            x_max,
            y_min,
            y_max,
            coords_type='geo',
            x_dim_lab='lon',
            y_dim_lab='lat'):

        '''Set required areal variables'''

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.coords_type = coords_type

        self.x_dim_lab = x_dim_lab
        self.y_dim_lab = y_dim_lab

        self._verify_bounds()
        self._coords_set_flag = True
        return

    def set_times(
            self,
            beg_time_str,
            end_time_str,
            time_fmt_str,
            hrs_list=None,
            time_res_str='H',
            time_dim_lab='time'):

        '''Set required temporal variables'''

        self.beg_time_str = beg_time_str
        self.end_time_str = end_time_str
        self.time_fmt_str = time_fmt_str
        self.time_res_str = time_res_str
        self.hrs_list = hrs_list
        self.time_dim_lab = time_dim_lab

        self._verify_times()
        self._times_set_flag = True
        return

    def _verify_paths(self):

        assert self.in_nc_path.exists()

        assert self.out_nc_path.parent.is_dir()
        assert self.out_nc_path.parent.exists()
        assert self.out_nc_path.suffix
        return

    def _verify_bounds(self):

        assert isinstance(self.x_min, (int, float))
        assert isinstance(self.x_max, (int, float))
        assert isinstance(self.y_min, (int, float))
        assert isinstance(self.y_max, (int, float))

        assert isinstance(self.x_dim_lab, str)
        assert isinstance(self.y_dim_lab, str)

        # TODO: implement projected type
        assert self.coords_type == 'geo', 'coords_type can only be \'geo\'!'

        if self.coords_type == 'geo':
            assert 0 <= self.x_min < 360
            assert 0 <= self.x_max < 360

            assert 0 <= self.y_min <= 180
            assert 0 <= self.y_max <= 180

            assert self.y_min < self.y_max
        return

    def _verify_times(self):

        assert isinstance(self.beg_time_str, str)
        assert isinstance(self.end_time_str, str)
        assert isinstance(self.time_fmt_str, str)
        assert isinstance(self.time_res_str, str)

        (self.beg_time, self.end_time) = pd.to_datetime(
             [self.beg_time_str, self.end_time_str], format=self.time_fmt_str)

        # a buffer of days
        self.beg_time = self.beg_time - pd.Timedelta(days=1)
        self.end_time = self.end_time + pd.Timedelta(days=1)

        # TODO: implement for other resolutions
        assert self.time_res_str == 'H'

        if self.hrs_list is not None:
            assert isinstance(self.hrs_list, list)
            assert all([isinstance(x, int) for x in self.hrs_list])

        assert isinstance(self.time_dim_lab, str)
        return

    def snip(self):

        '''Snip netcdf after setting all inputs and verifying them'''

        assert self._paths_set_flag
        assert self._coords_set_flag
        assert self._times_set_flag

        in_nc_ds = xr.open_dataset(self.in_nc_path)

        assert hasattr(in_nc_ds, self.x_dim_lab)
        assert hasattr(in_nc_ds, self.y_dim_lab)
        assert hasattr(in_nc_ds, self.time_dim_lab)

        time_idxs = pd.DatetimeIndex(
            getattr(in_nc_ds, self.time_dim_lab).values)

        assert time_idxs.shape[0] == getattr(
            in_nc_ds, self.time_dim_lab).shape[0]

        x_coords = getattr(in_nc_ds, self.x_dim_lab).values
        y_coords = getattr(in_nc_ds, self.y_dim_lab).values

        time_idxs_bool = (
            (time_idxs >= self.beg_time) & (time_idxs <= self.end_time))

        if self.hrs_list is not None:
            hours_idxs_bool = np.zeros(time_idxs.shape, dtype=bool)

            for hr in self.hrs_list:
                hours_idxs_bool = hours_idxs_bool | (time_idxs.hour == hr)

            time_idxs_bool = hours_idxs_bool & time_idxs_bool

        _1 = time_idxs.duplicated(keep='last')
        _2 = np.sum(_1)

        if _2:
            print_warning('Warning: Duplicate times exists in the input!')
            print_warning('Duplicate times:')
            print_warning(time_idxs[_1])
            print_warning('These are not included in the output.')
            print('\n')

            time_idxs_bool = np.where(_1, False, time_idxs_bool)

            assert not np.sum(time_idxs[time_idxs_bool].duplicated())

        if self.x_min > self.x_max:
            _1 = x_coords >= self.x_min
            _2 = x_coords <= self.x_max

            x_coords_bool = np.concatenate((x_coords[_1], x_coords[_2]))

        else:
            x_coords_bool = (x_coords >= self.x_min) & (x_coords <= self.x_max)

        y_coords_bool = (y_coords >= self.y_min) & (y_coords <= self.y_max)

        assert np.any(time_idxs_bool)
        assert np.any(x_coords_bool)
        assert np.any(y_coords_bool)

        out_nc_ds = in_nc_ds.loc[
            {self.time_dim_lab: time_idxs_bool,
             self.x_dim_lab: x_coords_bool,
             self.y_dim_lab: y_coords_bool}]

        self.out_nc_ds = out_nc_ds

        self._snip_flag = True
        return

    def save_snip(self):

        assert self._snip_flag

        self.out_nc_ds.to_netcdf(str(self.out_nc_path))
        return
