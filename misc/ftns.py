'''
Created on Jan 4, 2018

@author: Faizan-Uni
'''
from math import log as mlog

import pyproj
import numpy as np
from scipy.stats import rankdata

# use change_pts_crs as it way faster.
# def change_pt_crs(x, y, in_epsg, out_epsg):
#     """
#     Purpose:
#         To return the coordinates of given points in a different coordinate system.
#
#     Description of arguments:
#         x (int or float, single or list): The horizontal position of the input point
#         y (int or float, single or list): The vertical position of the input point
#         Note: In case of x and y in list form, the output is also in a list form.
#         in_epsg (string or int): The EPSG code of the input coordinate system
#         out_epsg (string or int): The EPSG code of the output coordinate system
#     """
#     in_crs = pyproj.Proj("+init=EPSG:" + str(in_epsg))
#     out_crs = pyproj.Proj("+init=EPSG:" + str(out_epsg))
#     return pyproj.transform(in_crs, out_crs, float(x), float(y))


def change_pts_crs(xs, ys, in_epsg, out_epsg):

    """
    Purpose:
        To return the coordinates of given points in a different coordinate system.

    Description of arguments:
        x (array): The horizontal position of the input point
        y (array): The vertical position of the input point
        in_epsg (string or int): The EPSG code of the input coordinate system
        out_epsg (string or int): The EPSG code of the output coordinate system
    """

    tfmr = pyproj.Transformer.from_crs(
        f'EPSG:{in_epsg}', f'EPSG:{out_epsg}', always_xy=True)

    out_crds = tfmr.transform(xs, ys)

    return out_crds


def ret_mp_idxs(n_vals, n_cpus):
    idxs = np.linspace(0, n_vals, n_cpus + 1, endpoint=True, dtype='int64')

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def KS_two_samp_test(ref_vals, sim_vals, alpha):
    assert not np.any(np.isnan(ref_vals))
    assert not np.any(np.isnan(sim_vals))
    assert not np.any(np.isnan([alpha]))

    assert ref_vals.ndim == 1
    assert sim_vals.ndim == 1

    assert isinstance(alpha, float)
    assert 0 < alpha < 1

    ref_vals = ref_vals.copy()
    sim_vals = sim_vals.copy()

    ref_vals.sort()
    sim_vals.sort()

    n_ref = ref_vals.shape[0]
    n_sim = sim_vals.shape[0]

    ref_probs = rankdata(ref_vals) / (n_ref + 1)
    sim_probs = rankdata(sim_vals) / (n_sim + 1)

    c_alpha = (-0.5 * mlog((1 - alpha) * 0.5)) ** 0.5
    ks_lim = c_alpha * (((n_ref + n_sim) / (n_ref * n_sim)) ** 0.5)

    sim_probs_in_ref = np.interp(sim_vals, ref_vals, ref_probs)
    max_diff = max(np.abs(sim_probs_in_ref - sim_probs))

    if max_diff > ks_lim:
        ret_val = 'reject'
    else:
        ret_val = 'accept'
    return ret_val
