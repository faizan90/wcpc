'''
Created on Jan 4, 2018

@author: Faizan-Uni
'''
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import matplotlib.gridspec as gridspec

plt.ioff()

from spinterps import Variogram, OrdinaryKriging

from ..misc.ftns import change_pts_crs
from ..misc.checks import check_nans_finite
from ..alg_dtypes import DT_D_NP, DT_UL_NP
from ..misc.ftns import ret_mp_idxs


class PlotCPs:

    def __init__(self, msgs=True):
        assert isinstance(msgs, (int, bool))
        self.msgs = msgs

        self._epsg_set_flag = False
        self._bck_shp_set_flag = False
        self._coords_arr_set_flag = False
        self._sel_cps_arr_set_flag = False
        self._cp_rules_arr_set_flag = False
        self._anom_arr_set_flag = False
        self._other_prms_set_flag = False
        self._kriged_flag = False
        return

    def set_epsgs(self,
                  anom_epsg,
                  bck_shp_epsg,
                  out_epsg):

        assert isinstance(anom_epsg, int)
        assert isinstance(bck_shp_epsg, int)
        assert isinstance(out_epsg, int)

        assert bck_shp_epsg == out_epsg

        self.anom_epsg = anom_epsg
        self.bck_shp_epsg = bck_shp_epsg
        self.out_epsg = out_epsg

        self._epsg_set_flag = True
        return

    def set_bck_shp(self, bck_shp_path=None):

        if bck_shp_path is not None:
            bck_shp_path = Path(bck_shp_path)
            assert bck_shp_path.exists()

            sf = shp.Reader(str(bck_shp_path))
            self.bck_polys_list = [i.__geo_interface__
                                   for i in sf.iterShapes()]
        else:
            self.bck_polys_list = None

        self.bck_shp_path = bck_shp_path

        self._bck_shp_set_flag = True
        return

    def set_coords_arr(self, x_coords, y_coords):
        assert isinstance(x_coords, np.ndarray)
        assert isinstance(y_coords, np.ndarray)

        assert check_nans_finite(x_coords)
        assert check_nans_finite(y_coords)

        assert x_coords.ndim == 1
        assert y_coords.ndim == 1

        assert x_coords.shape[0] and y_coords.shape[0]

        self.x_coords = np.array(x_coords, dtype=DT_D_NP, order='C')
        self.y_coords = np.array(y_coords, dtype=DT_D_NP, order='C')

        self.n_pts = self.x_coords.shape[0] * self.y_coords.shape[0]

        self._coords_arr_set_flag = True
        return

    def set_sel_cps_arr(self, sel_cps_arr):
        assert isinstance(sel_cps_arr, np.ndarray)
        assert check_nans_finite(sel_cps_arr)
        assert sel_cps_arr.ndim == 1
        assert sel_cps_arr.shape[0]

        self.sel_cps_arr = np.array(sel_cps_arr, dtype=DT_UL_NP, order='C')
        self._sel_cps_arr_set_flag = True
        return

    def set_cp_rules_arr(self, cp_rules_arr):
        assert isinstance(cp_rules_arr, np.ndarray)
        assert check_nans_finite(cp_rules_arr)
        assert cp_rules_arr.ndim == 2
        assert all(cp_rules_arr.shape)

        self.cp_rules_arr = np.array(cp_rules_arr, dtype=DT_UL_NP, order='C')
        self._cp_rules_arr_set_flag = True
        return

    def set_anoms_arr(self, anom_arr):
        assert isinstance(anom_arr, np.ndarray)
        assert check_nans_finite(anom_arr)
        assert anom_arr.ndim == 2
        assert all(anom_arr.shape)

        self.anom_arr = np.array(anom_arr, dtype=DT_D_NP, order='C')
        self._anom_arr_set_flag = True
        return

    def set_other_prms(self, fuzz_nos_arr, n_cps, anom_type, in_coords_type='geo'):
        assert isinstance(fuzz_nos_arr, np.ndarray)
        assert check_nans_finite(fuzz_nos_arr)
        assert fuzz_nos_arr.ndim == 2
        assert fuzz_nos_arr.shape[0]
        assert fuzz_nos_arr.shape[1] == 3

        assert isinstance(n_cps, int)
        assert n_cps > 0

        assert isinstance(in_coords_type, str)
        assert (in_coords_type == 'geo') or (in_coords_type == 'proj')

        assert isinstance(anom_type, str)
        assert len(anom_type) == 1
        assert 'b' <= anom_type <= 'f'

        self.n_fuzz_nos = fuzz_nos_arr.shape[0]
        self.n_cps = n_cps
        self.in_coords_type = in_coords_type
        self.anom_type = anom_type

        self._other_prms_set_flag = True
        return

    def _prep_coords(self):
        if self.in_coords_type == 'geo':
            self.x_coords = np.where(self.x_coords > self.x_coords[-1],
                                     self.x_coords - 360,
                                     self.x_coords)

        _1 = np.linspace(self.x_coords.min(),
                         self.x_coords.max(),
                         self.n_krige_intervals)
        _2 = np.linspace(self.y_coords.min(),
                         self.y_coords.max(),
                         self.n_krige_intervals)

        _1_mesh, _2_mesh = np.meshgrid(_1, _2)

        _1_mesh_rav = _1_mesh.ravel()
        _2_mesh_rav = _2_mesh.ravel()

        self.n_pts_krige = _1_mesh.shape[0] * _1_mesh.shape[1]
        self.krige_pts_shape = _1_mesh.shape

        self.krige_x_coords, self.krige_y_coords = change_pts_crs(
            _1_mesh_rav,
            _2_mesh_rav,
            self.anom_epsg,
            self.out_epsg)

        self.krige_x_coords_mesh = self.krige_x_coords.reshape(_1_mesh.shape)
        self.krige_y_coords_mesh = self.krige_y_coords.reshape(_1_mesh.shape)
        return

    def krige(self, n_krige_intervals):
        assert self._epsg_set_flag
        assert self._bck_shp_set_flag
        assert self._coords_arr_set_flag
        assert self._sel_cps_arr_set_flag
        assert self._cp_rules_arr_set_flag
        assert self._anom_arr_set_flag
        assert self._other_prms_set_flag

        assert isinstance(n_krige_intervals, int)
        assert n_krige_intervals > 0

        assert self.sel_cps_arr.shape[0] == self.anom_arr.shape[0]

        assert self.anom_arr.shape[1] == self.n_pts

        self.n_krige_intervals = n_krige_intervals

        self._prep_coords()

        self.best_cps_mean_anoms = np.empty(shape=(self.n_cps, self.n_pts),
                                           dtype=float)
        self.best_cps_std_anoms = np.empty(shape=(self.n_cps,
                                                  self.y_coords.shape[0],
                                                  self.x_coords.shape[0]),
                                           dtype=float)

        self.best_cps_min_anoms = self.best_cps_std_anoms.copy()
        self.best_cps_max_anoms = self.best_cps_std_anoms.copy()

        (self.x_coords_mesh,
         self.y_coords_mesh) = np.meshgrid(self.x_coords,
                                           self.y_coords)

        (self.x_coords_mesh_rav,
         self.y_coords_mesh_rav) = (self.x_coords_mesh.ravel(),
                                    self.y_coords_mesh.ravel())

        assert self.x_coords_mesh_rav.shape[0] == self.n_pts

        for j in range(self.n_cps):
            _ = self.sel_cps_arr == j
            _ = self.anom_arr[_]

            _1 = _.mean(axis=0)
            _2 = _.std(axis=0)
            _3 = _.min(axis=0)
            _4 = _.max(axis=0)

            self.best_cps_mean_anoms[j] = _1

            self.best_cps_std_anoms[j] = _2.reshape((self.y_coords.shape[0],
                                                     self.x_coords.shape[0]))

            self.best_cps_min_anoms[j] = _3.reshape((self.y_coords.shape[0],
                                                     self.x_coords.shape[0]))

            self.best_cps_max_anoms[j] = _4.reshape((self.y_coords.shape[0],
                                                     self.x_coords.shape[0]))

        self.krige_z_coords = np.zeros((self.n_cps, self.n_pts_krige))
        self.krige_z_coords_mesh = np.zeros((self.n_cps,
                                             *self.krige_pts_shape))

        self.cp_x_coords_list = []
        self.cp_y_coords_list = []
        self.cp_z_coords_list = []
        self.vgs_list = []

        for j in range(self.n_cps):
            if self.msgs:
                print('Kriging CP:', (j))

            curr_mean_cp = self.best_cps_mean_anoms[j]

            if self.anom_type != 'd':
                curr_false_idxs = self.cp_rules_arr[j] == self.n_fuzz_nos
            else:
                curr_false_idxs = (
                    np.zeros(self.best_cps_mean_anoms[j].shape[0], dtype=bool))

            curr_true_idxs = np.logical_not(curr_false_idxs)
            curr_mean_cp[curr_false_idxs] = np.nan

            curr_cp_vals = curr_mean_cp[curr_true_idxs]
            curr_x_coord_vals = self.x_coords_mesh_rav[curr_true_idxs]
            curr_y_coord_vals = self.y_coords_mesh_rav[curr_true_idxs]

            curr_mean_cp = curr_mean_cp.reshape(self.x_coords_mesh.shape)

            #==================================================================
            # Reproject to out_epsg
            #==================================================================

            curr_re_x_coords, curr_re_y_coords = change_pts_crs(
                curr_x_coord_vals,
                curr_y_coord_vals,
                self.anom_epsg,
                self.out_epsg)

            self.cp_x_coords_list.append(curr_re_x_coords)
            self.cp_y_coords_list.append(curr_re_y_coords)
            self.cp_z_coords_list.append(curr_cp_vals)

            #==================================================================
            # Fit Variogram
            #==================================================================
            variogram = Variogram(
                x=curr_re_x_coords,
                y=curr_re_y_coords,
                z=curr_cp_vals,
                perm_r_list=[1],
                fit_vgs=['Sph'])

            variogram.fit()
            fit_vg = variogram.vg_str_list[0]
            assert fit_vg

            self.vgs_list.append(fit_vg)

            if self.msgs:
                print('Fitted variogram is:', fit_vg)

            #==================================================================
            # Krige
            #==================================================================
            ord_krig = OrdinaryKriging(xi=curr_re_x_coords,
                                       yi=curr_re_y_coords,
                                       zi=curr_cp_vals,
                                       xk=self.krige_x_coords,
                                       yk=self.krige_y_coords,
                                       model=fit_vg)
            ord_krig.krige()

            krige_z_coords = ord_krig.zk
            self.krige_z_coords[j] = krige_z_coords

            assert check_nans_finite(krige_z_coords)

            krige_z_coords_mesh = krige_z_coords.reshape(self.krige_pts_shape)
            self.krige_z_coords_mesh[j] = krige_z_coords_mesh

        self._kriged_flag = True
        return

    def plot_kriged_cps(self,
                        cont_levels,
                        out_figs_dir,
                        fig_size=((10, 7)),
                        n_cpus=1):
        assert self._kriged_flag

        assert isinstance(cont_levels, np.ndarray)
        assert check_nans_finite(cont_levels)
        assert cont_levels.ndim == 1
        assert cont_levels.shape[0] > 0

        assert isinstance(out_figs_dir, (Path, str))
        out_figs_dir = Path(out_figs_dir)
        assert out_figs_dir.parents[0].exists()
        if not out_figs_dir.exists():
            out_figs_dir.mkdir()

        assert isinstance(fig_size, (tuple, list))
        assert len(fig_size) == 2
        assert all(fig_size)

        assert isinstance(n_cpus, int)
        assert n_cpus > 0

        n_cpus = max(1, min(n_cpus, int(self.n_cps / 3)))

        _plot_kriged_cps_partial = (
            partial(_plot_kriged_cps,
                    fig_size=fig_size,
                    bck_polys_list=self.bck_polys_list,
                    krige_x_coords_mesh=self.krige_x_coords_mesh,
                    krige_y_coords_mesh=self.krige_y_coords_mesh,
                    anom_type=self.anom_type,
                    cont_levels=cont_levels,
                    out_figs_dir=out_figs_dir,
                    x_coords=self.x_coords,
                    y_coords=self.y_coords,
                    best_cps_min_anoms=self.best_cps_min_anoms,
                    best_cps_max_anoms=self.best_cps_max_anoms,
                    best_cps_std_anoms=self.best_cps_std_anoms,
                    msgs=self.msgs))

        _idxs = ret_mp_idxs(self.n_cps, n_cpus)
        cp_nos_list = [_idxs[i: i + 2] for i in range(n_cpus)]

        _z_gen = (
            (self.krige_z_coords_mesh[cp_nos_list[i][0]:cp_nos_list[i][1]])
            for i in range(n_cpus))
        _x_gen = (self.cp_x_coords_list[cp_nos_list[i][0]:cp_nos_list[i][1]]
                  for i in range(n_cpus))
        _y_gen = (self.cp_y_coords_list[cp_nos_list[i][0]:cp_nos_list[i][1]]
                  for i in range(n_cpus))
        _vgs_gen = (self.vgs_list[cp_nos_list[i][0]:cp_nos_list[i][1]]
                    for i in range(n_cpus))
        _cp_rules_gen = (self.cp_rules_arr[cp_nos_list[i][0]:cp_nos_list[i][1]]
                         for i in range(n_cpus))

        _all_gen = (
            (cp_nos_list[i],
             self.krige_z_coords_mesh[cp_nos_list[i][0]:cp_nos_list[i][1]],
             self.cp_x_coords_list[cp_nos_list[i][0]:cp_nos_list[i][1]],
             self.cp_y_coords_list[cp_nos_list[i][0]:cp_nos_list[i][1]],
             self.vgs_list[cp_nos_list[i][0]:cp_nos_list[i][1]],
             self.cp_rules_arr[cp_nos_list[i][0]:cp_nos_list[i][1]])
             for i in range(n_cpus))

        if n_cpus == 1:
            _map = list(map(_plot_kriged_cps_partial, _all_gen))

        else:
            mp_pool = Pool(processes=n_cpus, maxtasksperchild=1)
            _map = list(mp_pool.imap_unordered(_plot_kriged_cps_partial,
                                               _all_gen))
            mp_pool.terminate()

        if self.msgs:
            print(_map)
        return


def _plot_kriged_cps(args,
                     fig_size,
                     bck_polys_list,
                     krige_x_coords_mesh,
                     krige_y_coords_mesh,
                     anom_type,
                     cont_levels,
                     out_figs_dir,
                     x_coords,
                     y_coords,
                     best_cps_min_anoms,
                     best_cps_max_anoms,
                     best_cps_std_anoms,
                     msgs):

    (cp_nos,
     krige_z_coords_mesh,
     cp_x_coords_list,
     cp_y_coords_list,
     vgs_list,
     cp_rules_arr) = args

    max_n_coords = 10
    n_x_coords = x_coords.shape[0]
    n_y_coords = y_coords.shape[0]

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

    cmap = plt.get_cmap('autumn')

    min_cbar_val = min(best_cps_min_anoms.min(),
                       best_cps_max_anoms.min())

    max_cbar_val = max(best_cps_min_anoms.max(),
                       best_cps_max_anoms.max())

    # CP
    cps_fig = plt.figure(figsize=fig_size)
    cps_ax = cps_fig.gca()

    cp_std_fig = plt.figure(figsize=fig_size)
    cp_std_ax = cp_std_fig.gca()

    for j, cp_no in enumerate(range(cp_nos[0], cp_nos[1])):
        if msgs:
            print('Plotting CP No.', cp_no)

        plt.figure(cps_fig.number)

        if bck_polys_list is not None:
            for poly in bck_polys_list:
                cps_ax.add_patch(PolygonPatch(poly,
                                              alpha=0.5,
                                              fc='#999999',
                                              ec='#999999'))

        cs = cps_ax.contour(krige_x_coords_mesh,
                            krige_y_coords_mesh,
                            krige_z_coords_mesh[j],
                            levels=cont_levels,
                            vmin=0.0,
                            vmax=1.0,
                            linestyles='solid',
                            extend='both')

        if anom_type != 'd':
            cps_ax.scatter(cp_x_coords_list[j], cp_y_coords_list[j])

        cps_ax.set_title('CP no. %d, %s' % (cp_no, vgs_list[j]))

        cps_ax.set_xlim(krige_x_coords_mesh.min(),
                        krige_x_coords_mesh.max())
        cps_ax.set_ylim(krige_y_coords_mesh.min(),
                        krige_y_coords_mesh.max())

        cps_ax.set_xlabel('Eastings')
        cps_ax.set_ylabel('Northings')

        cps_ax.clabel(cs, inline=True, inline_spacing=0.01, fontsize=10)

        plt.savefig(str(out_figs_dir / ('cp_map_%0.2d.png' % (cp_no))),
                    bbox_inches='tight')

        cps_ax.cla()

        # CP Std.
        plt.figure(cp_std_fig.number)
        cax = cp_std_ax.imshow(best_cps_std_anoms[cp_no],
                               origin='upper',
                               interpolation=None,
                               cmap=cmap)

        if anom_type != 'd':
            _cp_rules_str = (
                cp_rules_arr[j].reshape(
                    best_cps_std_anoms[cp_no].shape).astype('|U'))

            txt_x_corrs = np.tile(range(_cp_rules_str.shape[1]),
                                  _cp_rules_str.shape[0])

            txt_y_corrs = np.repeat(range(_cp_rules_str.shape[0]),
                                    _cp_rules_str.shape[1])

            for k in range(txt_x_corrs.shape[0]):
                cp_std_ax.text(txt_x_corrs[k],
                               txt_y_corrs[k],
                               _cp_rules_str[txt_y_corrs[k],
                                             txt_x_corrs[k]],
                               va='center',
                               ha='center',
                               color='black')

        cp_std_ax.set_xticks(x_ticks_pos[::x_step_size])
        cp_std_ax.set_yticks(y_ticks_pos[::y_step_size])

        cp_std_ax.set_xticklabels(x_coords[::x_step_size])
        cp_std_ax.set_yticklabels(y_coords[::y_step_size])

        cp_std_ax.set_xlabel('Eastings')
        cp_std_ax.set_ylabel('Northings')

        cbar = cp_std_fig.colorbar(cax, orientation='horizontal')
        cbar.set_label('Anomaly Std.')

        cp_std_ax.set_title('Std. CP no. %d' % (cp_no))

        plt.savefig(str(out_figs_dir / ('std_cp_map_%0.2d.png' % (cp_no))),
                    bbox_inches='tight')
        cbar.remove()
        cp_std_ax.cla()

        # CP Min. and Max.
        cp_min_max_fig = plt.figure(figsize=fig_size)
        cp_min_max_grid = gridspec.GridSpec(1, 2)
        cp_min_ax = plt.subplot(cp_min_max_grid[0, 0])
        cp_max_ax = plt.subplot(cp_min_max_grid[0, 1])

        plt.figure(cp_min_max_fig.number)

        # Min
        cax_min = cp_min_ax.imshow(best_cps_min_anoms[cp_no],
                                   origin='upper',
                                   interpolation=None,
                                   vmin=min_cbar_val,
                                   vmax=max_cbar_val,
                                   cmap=cmap)

        cp_min_ax.set_xticks(x_ticks_pos[::x_step_size])
        cp_min_ax.set_yticks(y_ticks_pos[::y_step_size])

        cp_min_ax.set_xticklabels(x_coords[::x_step_size])
        cp_min_ax.set_yticklabels(y_coords[::y_step_size])

        cp_min_ax.set_xlabel('Eastings')
        cp_min_ax.set_ylabel('Northings')

        cp_min_ax.set_title('Min. Anomaly CP no. %d' % (cp_no))

        # Max
        cp_max_ax.imshow(best_cps_max_anoms[cp_no],
                         origin='upper',
                         interpolation=None,
                         vmin=min_cbar_val,
                         vmax=max_cbar_val,
                         cmap=cmap)

        cp_max_ax.set_xticks(x_ticks_pos[::x_step_size])
        cp_max_ax.set_yticks(y_ticks_pos[::y_step_size])

        cp_max_ax.set_xticklabels(x_coords[::x_step_size])
        cp_max_ax.set_yticklabels([])

        cp_max_ax.set_xlabel('Eastings')

        cbar = cp_min_max_fig.colorbar(cax_min,
                                       ax=[cp_min_ax, cp_max_ax],
                                       orientation='horizontal')

        cp_max_ax.set_title('Max. Anomaly CP no. %d' % (cp_no))

        plt.savefig(str(out_figs_dir /
                        ('min_max_cp_map_%0.2d.png' % (cp_no))),
                    bbox_inches='tight')

        plt.close()

    plt.close('all')
    return


def plot_iter_cp_pcntgs(n_cps,
                        curr_n_iters_arr,
                        cp_pcntge_arr,
                        out_fig_loc,
                        best_iter_idx,
                        old_new_cp_map_arr=None,
                        fig_size=(17, 10),
                        msgs=True):
    if msgs:
        print('\n\nPlotting CP frequency evolution...')

    assert isinstance(n_cps, int)
    assert n_cps > 0

    assert isinstance(curr_n_iters_arr, np.ndarray)
    assert check_nans_finite(curr_n_iters_arr)
    assert curr_n_iters_arr.ndim == 1

    assert isinstance(cp_pcntge_arr, np.ndarray)
    assert check_nans_finite(cp_pcntge_arr)
    assert cp_pcntge_arr.ndim == 2
    assert all(cp_pcntge_arr.shape)
    assert cp_pcntge_arr.shape[1] == n_cps

    assert isinstance(out_fig_loc, (str, Path))
    out_fig_loc = Path(out_fig_loc)
    assert out_fig_loc.parents[0].exists()

    assert isinstance(best_iter_idx, int)
    assert best_iter_idx >= 0

    if old_new_cp_map_arr is not None:
        assert isinstance(old_new_cp_map_arr, np.ndarray)
        assert check_nans_finite(old_new_cp_map_arr)
        assert old_new_cp_map_arr.ndim == 2
        assert old_new_cp_map_arr.shape[1] == 2
        assert old_new_cp_map_arr.shape[0] == n_cps
    else:
        old_new_cp_map_arr = np.repeat(range(n_cps), 2).reshape(-1, 2)

    out_fig_pre_path = out_fig_loc.parents[0]
    out_fig_name, out_ext = out_fig_loc.name.rsplit('.', 1)

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    best_idx = int(np.where(curr_n_iters_arr == best_iter_idx)[0][0])

    for i in range(n_cps):
        curr_cp_pcntge_arr = cp_pcntge_arr[:, old_new_cp_map_arr[i, 0]]
        ax.plot(curr_n_iters_arr,
                curr_cp_pcntge_arr,
                label='CP %2d' % i,
                color='blue',
                alpha=0.75)
        ax.scatter(best_iter_idx,
                   cp_pcntge_arr[best_idx, old_new_cp_map_arr[i, 0]],
                   color='red',
                   label='Final freq.',
                   alpha=0.9)

        ax.set_xlabel('Iteration No. (-)')
        ax.set_ylabel('Relative CP frequency (-)')
        ax.set_ylim(0.0, 1.0)
        ax.grid()
        ax.legend(loc=0)
        ax.set_title('CP classification - CP frequency evolution')

        plt.savefig(str(out_fig_pre_path /
                        ((out_fig_name + ('_cp_%0.2d.' % i) + out_ext))),
                    bbox_inches='tight')

        ax.cla()

    plt.close()
    return
