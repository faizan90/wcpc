'''
Created on Jan 6, 2018

@author: Faizan
'''
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

plt.ioff()

from ..misc.error_msgs import print_warning


class PlotNC:
    def __init__(self, msgs=True):
        assert isinstance(msgs, (int, bool))
        self.msgs = msgs

        self._vars_set_flag = False
        return

    def set_vars(self,
                 path_to_nc,
                 out_dir,
                 var_lab,
                 x_coords_lab,
                 y_coords_lab,
                 time_lab,
                 beg_time,
                 end_time,
                 time_fmt):

        assert isinstance(path_to_nc, (str, Path))
        assert isinstance(out_dir, (str, Path))
        assert isinstance(var_lab, str)
        assert isinstance(x_coords_lab, str)
        assert isinstance(y_coords_lab, str)
        assert isinstance(time_lab, str)
        assert isinstance(beg_time, str)
        assert isinstance(end_time, str)
        assert isinstance(time_fmt, str)

        path_to_nc = Path(path_to_nc)
        assert path_to_nc.exists()

        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir()

        in_ds = xr.open_dataset(str(path_to_nc))
        assert hasattr(in_ds, var_lab)
        assert hasattr(in_ds, x_coords_lab)
        assert hasattr(in_ds, y_coords_lab)
        assert hasattr(in_ds, time_lab)

        self.in_ds = in_ds
        self.path_to_nc = path_to_nc
        self.out_dir = out_dir
        self.var_lab = var_lab
        self.x_coords_lab = x_coords_lab
        self.y_coords_lab = y_coords_lab
        self.time_lab = time_lab
        self.time_fmt = time_fmt

        (self.beg_time,
         self.end_time) = pd.to_datetime([beg_time, end_time],
                                         format=time_fmt)

        self._vars_set_flag = True
        return

    def plot(self, fig_size=(13, 10)):
        assert self._vars_set_flag

        time_idxs = pd.DatetimeIndex(getattr(self.in_ds,
                                             self.time_lab).values)

        time_idxs_bool = ((time_idxs >= self.beg_time) &
                          (time_idxs <= self.end_time))

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

        plot_ds = self.in_ds.loc[{self.time_lab: time_idxs_bool}]

        x_coords_arr = getattr(plot_ds, self.x_coords_lab).values
        y_coords_arr = getattr(plot_ds, self.y_coords_lab).values

        assert x_coords_arr.ndim == 1
        assert y_coords_arr.ndim == 1

        times_arr = pd.DatetimeIndex(getattr(plot_ds, self.time_lab).values)
        n_steps = times_arr.shape[0]
        
        max_n_coords = 10

        if self.msgs:
            print('\nPlotting %d steps...' % np.sum(time_idxs_bool))

        for i in range(n_steps):
            curr_var_ds = plot_ds[{self.time_lab: i}]
            curr_var_vals = getattr(curr_var_ds, self.var_lab).values
            curr_time = getattr(curr_var_ds, self.time_lab).values

            curr_x_coords = getattr(curr_var_ds, self.x_coords_lab).values
            curr_y_coords = getattr(curr_var_ds, self.y_coords_lab).values
            
            if curr_var_vals.shape[0] != curr_y_coords.shape[0]:
                curr_var_vals = curr_var_vals.transpose()

            if curr_y_coords[0] > curr_y_coords[-1]:
                curr_y_coords = curr_y_coords[::-1]
                curr_var_vals = np.flipud(curr_var_vals)

            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            cax = ax.imshow(curr_var_vals, origin='lower', interpolation=None)

            n_x_coords = curr_x_coords.shape[0]
            n_y_coords = curr_y_coords.shape[0]

            x_ticks_pos = np.arange(0, n_x_coords)
            y_ticks_pos = np.arange(0, n_y_coords)
            
            if n_x_coords > max_n_coords:
                x_step_size = n_x_coords // max_n_coords
            else:
                x_step_size = 1

            if n_y_coords > max_n_coords:
                y_step_size = n_y_coords // max_n_coords
            else:
                y_step_size = 1

            ax.set_xticks(x_ticks_pos[::x_step_size])
            ax.set_yticks(y_ticks_pos[::y_step_size])

            ax.set_xticklabels(curr_x_coords[::x_step_size])
            ax.set_yticklabels(curr_y_coords[::y_step_size])

            ax.set_xlabel(self.x_coords_lab)
            ax.set_ylabel(self.y_coords_lab)

            cbar = fig.colorbar(cax, orientation='horizontal')
            cbar.set_label(self.var_lab)

            _ = pd.to_datetime(curr_time)
            out_fig_name = _.strftime('%Y%m%d%H%M%S.png')

            if self.msgs:
                print(_)

            ax.set_title('Variable: \'%s\' at time %s' %
                         (self.var_lab, str(_)))

            out_fig_path = self.out_dir / out_fig_name

            plt.savefig(str(out_fig_path), bbox_inches='tight')
            plt.close()

        return
