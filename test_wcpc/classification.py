'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wcpc.core.anomaly import Anomaly
from wcpc.core.classify import CPClassiA
from wcpc.core.alg_dtypes import DT_D_NP

np.set_printoptions(precision=3,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.3f}'.format})

pd.options.display.precision = 3
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 250

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

    in_net_cdf_file = main_dir / r'wcpc_test.nc'

    in_ppt_df_pkl = main_dir / r'ppt_19800101_20091231.pkl'
    in_wett_nebs_pkl = main_dir / r'ppt_wettness_neborhood.pkl'

    in_cats_ppt_df_pkl = main_dir / r'cats_ppt_19800101_20091231.pkl'

    in_prms_pkl = main_dir / r'wcpc_test_calib.pkl'

    os.chdir(main_dir)

    anomaly = Anomaly()

    anomaly.read_vars(in_net_cdf_file,
                      'lon',
                      'lat',
                      'time',
                      'slp',
                      'nc',
                      'D',
                      0)
    anomaly.calc_anomaly_type_a(0.5)

    vals_tot_anom = anomaly.vals_tot_anom
    
    dates_tot = anomaly.times_tot
    dates_tot = pd.DatetimeIndex(dates_tot.date)
    
    #==========================================================================
    # classification input
    #==========================================================================
    calib_period_srt = '1980-01-01'
    calib_period_end = '1989-12-31'

    time_fmt = '%Y-%m-%d'

    n_cpus = 7

    n_cps = 12
    max_idxs_ct = 10
    no_cp_val = 99
    miss_cp_val = 98
    p_l = 2.0

    o_1_ppt_thresh_arr = np.array([0.11, 10.0], dtype=DT_D_NP)
    o_2_ppt_thresh_arr = np.array([0.11, 10.0], dtype=DT_D_NP)

    o_4_wett_thresh_arr = np.array([0.0, 0.99], dtype=DT_D_NP)

    anneal_temp_ini = 0.026
    temp_red_alpha = 0.99

    max_m_iters = 5000
    max_n_iters = max_m_iters * 10
    max_k_iters = max_m_iters * 5

#     max_m_iters = 5000
#     max_n_iters = max_m_iters * 1000

    fuzz_nos_arr = np.array([[0.0, 0.0, 0.4],
                             [-0.2, 0.2, 0.5],
                             [0.5, 0.8, 1.2],
                             [0.6, 1.0, 1.1]], dtype=DT_D_NP)  # the fifth one is dealt with later

    in_ppt_df = pd.read_pickle(in_ppt_df_pkl)
    in_ppt_df = in_ppt_df.loc[dates_tot]

    in_wettness_df = pd.read_pickle(in_wett_nebs_pkl)
    in_wettness_df = in_wettness_df.loc[dates_tot]

    in_cats_ppt_df = pd.read_pickle(in_cats_ppt_df_pkl)
    in_cats_ppt_df = in_cats_ppt_df.loc[dates_tot]

#     rand_idxs = np.random.randint(0, in_ppt_df.columns.shape[0], 100)
#     rand_ppt_stns = in_ppt_df.columns[rand_idxs]

    rand_ppt_stns = ['P1197', 'P891', 'P1468', 'P232', 'P3015', 'P2497',
                     'P4169', 'P2211', 'P2542', 'P403', 'P3032', 'P1197',
                     'P2261', 'P3527', 'P3987']

#     rand_cats_idxs = np.random.randint(0, in_cats_ppt_df.columns.shape[0], 15)
#     rand_ppt_cats = in_cats_ppt_df.columns[rand_cats_idxs]
    rand_ppt_cats = ['1458', '460', '411', '76123', '420', '422', '2489',
                     '473', '1458', '420', '434', '406', '2477', '420', '3465']

    in_ppt_df = in_ppt_df[rand_ppt_stns]
    in_cats_ppt_df = in_cats_ppt_df[rand_ppt_cats]

    calib_dates = pd.to_datetime([calib_period_srt, calib_period_end],
                                 format=time_fmt)
    idxs_calib = ((dates_tot >= calib_dates[0]) &
                  (dates_tot <= calib_dates[1]))
    assert np.sum(idxs_calib), 'No steps to calibrate!'

    in_ppt_arr_calib = in_ppt_df.loc[idxs_calib].values.copy(order='C')
    slp_anom_calib = vals_tot_anom[idxs_calib, :].copy(order='C')
    in_wet_arr_calib = in_wettness_df.loc[idxs_calib].values.copy(
        order='C')
    in_cats_ppt_arr_calib = in_cats_ppt_df.loc[idxs_calib].values.copy(
        order='C')

    classi = CPClassiA()
    classi.set_stn_ppt(in_ppt_arr_calib)
    classi.set_cat_ppt(in_cats_ppt_arr_calib)
    classi.set_neb_wett(in_wet_arr_calib)
    classi.set_cp_prms(n_cps,
                       max_idxs_ct,
                       no_cp_val,
                       miss_cp_val,
                       p_l,
                       fuzz_nos_arr)

    classi.set_obj_1_on(o_1_ppt_thresh_arr, 1.0)
    classi.set_obj_2_on(o_2_ppt_thresh_arr, 1.0)
    classi.set_obj_3_on(0.1)
    classi.set_obj_4_on(o_4_wett_thresh_arr, 1.0)
    classi.set_obj_5_on(0.1)

    classi.set_cyth_flags(cyth_nonecheck=False,
                          cyth_boundscheck=False,
                          cyth_wraparound=False,
                          cyth_cdivision=True,
                          cyth_language_level=3,
                          cyth_infer_types=None)

    classi.set_anomaly(slp_anom_calib)
    classi.set_sim_anneal_prms(anneal_temp_ini,
                               temp_red_alpha,
                               max_m_iters,
                               max_n_iters,
                               max_k_iters)
    classi.classify('auto')
    #==========================================================================
    # 
    #==========================================================================
    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
