'''
@author: Faizan-Uni-Stuttgart

'''
# some text for git

# some text top pull at uni

import os
import timeit
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wcpc.core.anomaly import Anomaly

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

    out_dir = main_dir / r''

#     in_net_cdf_file = main_dir / r'NCAR_ds010.0_19610101_20151231_dailydata_europe.nc'
#     out_anomaly_pkl = out_dir / 'NCAR_ds010.0_19610101_20151231_dailydata_europe_ate.pkl'
#     out_crds_file = out_dir / 'NCAR_ds010.0_19610101_20151231_dailydata_europe_ate_crds.csv'
#     nc_var_lab = 'slp'

#     in_net_cdf_file = main_dir / r'ncep_1948_2017_level_500_europe.nc'
#     out_anomaly_pkl = out_dir / 'ncep_500_ate_1948_2015.pkl'
#     out_crds_file = out_dir / 'ncep_500_ate_1948_2015_crds.csv'
#     nc_var_lab = 'hgt'

    in_net_cdf_file = main_dir / r'ncep_1948_2017_level_500_6h_europe.nc'
    out_anomaly_pkl = out_dir / 'ncep_atb_1996_2014_level_500_6h.pkl'
#     out_crds_file = out_dir / 'ncep_atb_1996_2014_level_500_6h_crds.csv'

    nc_var_lab = 'hgt'

    strt_time = '1996-01-01'
    end_time = '2014-12-31'

    sep = ';'
    time_fmt = '%Y-%m-%d'
    sub_daily_flag = True
    normalize = True

    os.chdir(main_dir)

    out_dir.mkdir(exist_ok=True)

    anomaly = Anomaly()

    anomaly.read_vars(
        in_net_cdf_file,
        'lon',
        'lat',
        'time',
        nc_var_lab,
        'nc',
        'D',
        sub_daily_flag=sub_daily_flag)

    anomaly.calc_anomaly_type_b(
        strt_time,
        end_time,
        season_months=np.arange(1, 13),
        time_fmt=time_fmt,
        normalize=normalize)

#     with open(out_anomaly_pkl, 'wb') as _pkl_hdl:
#         pickle.dump(anomaly, _pkl_hdl)

#     # for app-dis
#     out_dict = {}
#
#     anomaly_var_df = pd.DataFrame(
#         data=anomaly.vals_tot_anom,
#         index=anomaly.times)
#
#     pcs_arr = anomaly.vals_anom
#     eig_val_cum_sums = anomaly.eig_val_cum_sum_arr
#
#     out_dict['anomaly_var_df'] = anomaly_var_df
#     out_dict['pcs_arr'] = pcs_arr
#     out_dict['eig_val_cum_sums'] = eig_val_cum_sums
#     out_dict['in_anomaly_eig_vecs_mat'] = anomaly.eig_vecs_mat
#
#     print(f'eig_val_cum_sums: {eig_val_cum_sums}')
#
#     # anom rank pca
#     rank_var_df = (
#         anomaly_var_df.rank() / (anomaly_var_df.shape[0] + 1)) - 0.5
#     out_dict['rank_var_df'] = rank_var_df
#     in_rank_corr_mat = np.corrcoef(rank_var_df.values.T)
#     print('in_rank_corr_mat shape:', in_rank_corr_mat.shape)
#     out_dict['in_rank_corr_mat'] = in_rank_corr_mat
#
#     in_rank_eig_vals, in_rank_eig_vecs_mat = np.linalg.eig(in_rank_corr_mat)
#     rank_eig_sort_idxs = np.argsort(in_rank_eig_vals)[::-1]
#     in_rank_eig_vals = in_rank_eig_vals[rank_eig_sort_idxs]
#     in_rank_eig_vecs_mat = in_rank_eig_vecs_mat[:, rank_eig_sort_idxs]
#     print('in_rank_eig_vals shape:', in_rank_eig_vals.shape)
#     print('in_rank_eig_vecs_mat shape:', in_rank_eig_vecs_mat.shape)
#     out_dict['in_rank_eig_vals'] = in_rank_eig_vals
#     out_dict['in_rank_eig_vecs_mat'] = in_rank_eig_vecs_mat
#
#     rank_eig_val_cum_sums = np.cumsum(in_rank_eig_vals) / in_rank_eig_vals.sum()
#     print('rank_eig_val_cum_sums:', rank_eig_val_cum_sums)
#     out_dict['rank_eig_val_cum_sums'] = rank_eig_val_cum_sums
#
#     rank_pcs_arr = np.dot(rank_var_df.values, in_rank_eig_vecs_mat)
#     out_dict['rank_pcs_arr'] = rank_pcs_arr

#     # for app-dis
#     out_dict = {}
#
#     anomaly_var_df = pd.DataFrame(
#         data=anomaly.vals_tot_anom,
#         index=anomaly.times)
#
#     pcs_arr = anomaly.vals_anom
#     eig_val_cum_sums = anomaly.eig_val_cum_sum_arr
#
#     out_dict['anomaly_var_df'] = anomaly_var_df
#     out_dict['pcs_arr'] = pcs_arr
#     out_dict['eig_val_cum_sums'] = eig_val_cum_sums
#     out_dict['in_anomaly_eig_vecs_mat'] = anomaly.eig_vecs_mat
#
#     print(f'eig_val_cum_sums: {eig_val_cum_sums}')
#
#     # anom rank pca
#     rank_var_df = (
#         anomaly_var_df.rank() / (anomaly_var_df.shape[0] + 1)) - 0.5
#     out_dict['rank_var_df'] = rank_var_df
#     in_rank_corr_mat = np.corrcoef(rank_var_df.values.T)
#     print('in_rank_corr_mat shape:', in_rank_corr_mat.shape)
#     out_dict['in_rank_corr_mat'] = in_rank_corr_mat
#
#     in_rank_eig_vals, in_rank_eig_vecs_mat = np.linalg.eig(in_rank_corr_mat)
#     rank_eig_sort_idxs = np.argsort(in_rank_eig_vals)[::-1]
#     in_rank_eig_vals = in_rank_eig_vals[rank_eig_sort_idxs]
#     in_rank_eig_vecs_mat = in_rank_eig_vecs_mat[:, rank_eig_sort_idxs]
#     print('in_rank_eig_vals shape:', in_rank_eig_vals.shape)
#     print('in_rank_eig_vecs_mat shape:', in_rank_eig_vecs_mat.shape)
#     out_dict['in_rank_eig_vals'] = in_rank_eig_vals
#     out_dict['in_rank_eig_vecs_mat'] = in_rank_eig_vecs_mat
#
#     rank_eig_val_cum_sums = np.cumsum(in_rank_eig_vals) / in_rank_eig_vals.sum()
#     print('rank_eig_val_cum_sums:', rank_eig_val_cum_sums)
#     out_dict['rank_eig_val_cum_sums'] = rank_eig_val_cum_sums
#
#     rank_pcs_arr = np.dot(rank_var_df.values, in_rank_eig_vecs_mat)
#     out_dict['rank_pcs_arr'] = rank_pcs_arr
#
#     with open(out_anomaly_pkl, 'wb') as _pkl_hdl:
#         pickle.dump(out_dict, _pkl_hdl)
#
#     crds_df = pd.DataFrame(
#         index=anomaly_var_df.columns, columns=['X', 'Y'], dtype=float)
#
#     crds_df['X'] = anomaly.x_coords_rav
#     crds_df['Y'] = anomaly.y_coords_rav
#
#     crds_df.to_csv(out_crds_file, sep=sep)
#
#     crds_df = pd.DataFrame(
#         index=anomaly_var_df.columns, columns=['X', 'Y'], dtype=float)
#
#     crds_df['X'] = anomaly.x_coords_rav
#     crds_df['Y'] = anomaly.y_coords_rav
#
#     crds_df.to_csv(out_crds_file, sep=sep)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
