'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wcpc import (Anomaly,
                  CPAssignA,
                  CPClassiA,
                  ThreshPPT,
                  WettnessIndex,
                  CPHistPlot,
                  PlotCPs,
                  DT_D_NP)

np.set_printoptions(precision=8,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.8f}'.format})

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

    backgrnd_shp_file = main_dir / Path(r'world_map_epsg_3035.shp')

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

    valid_period_srt = '1990-01-01'
    valid_period_end = '2009-12-31'

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

    cont_levels = np.linspace(0.0, 1.0, 41)

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

    if not in_prms_pkl.exists():
        classi.classify('auto', force_compile=False)
        with open(in_prms_pkl, 'wb') as in_prms_pkl:
            pickle.dump(classi, in_prms_pkl)
    else:
        with open(in_prms_pkl, 'rb') as prms_hdl:
            classi = pickle.load(prms_hdl)

    cp_rules = classi.cp_rules

    wettness = WettnessIndex()
    wettness.set_ppt_arr(in_ppt_arr_calib)
    wettness.set_cps_arr(classi.calib_dict['best_sel_cps'], n_cps)
    wettness.reorder_cp_rules(cp_rules)
    cp_rules = wettness.cp_rules_sorted
    
    assign_cps = CPAssignA()
    assign_cps.set_anomaly(slp_anom_calib)
    assign_cps.set_cp_prms(n_cps,
                           max_idxs_ct,
                           no_cp_val,
                           miss_cp_val,
                           p_l,
                           fuzz_nos_arr)
    assign_cps.set_cp_rules(cp_rules)

    assign_cps.assign_cps('auto', force_compile=False)
    calib_cps = assign_cps.sel_cps_arr

#     _ = np.where(calib_cps != classi.calib_dict['best_sel_cps'])
#
#     print(calib_cps[_])
#     print(classi.calib_dict['best_sel_cps'][_])
#
#     print(assign_cps.assign_dict['dofs_arr'][_])
#     print(classi.calib_dict['best_dofs_arr'][_])
#
#     assert np.all(calib_cps == classi.calib_dict['best_sel_cps'])

    valid_dates = pd.to_datetime([valid_period_srt, valid_period_end],
                                 format=time_fmt)
    idxs_valid = ((dates_tot >= valid_dates[0]) &
                  (dates_tot <= valid_dates[1]))
    assert np.sum(idxs_valid), 'No steps to validate!'

    in_ppt_arr_valid = in_ppt_df.loc[idxs_valid].values.copy(order='C')
    slp_anom_valid = vals_tot_anom[idxs_valid, :].copy(order='C')
    in_wet_arr_valid = in_wettness_df.loc[idxs_valid].values.copy(
        order='C')
    in_cats_ppt_arr_valid = in_cats_ppt_df.loc[idxs_valid].values.copy(
        order='C')

    assign_cps = CPAssignA()
    assign_cps.set_anomaly(slp_anom_valid)
    assign_cps.set_cp_prms(n_cps,
                           max_idxs_ct,
                           no_cp_val,
                           miss_cp_val,
                           p_l,
                           fuzz_nos_arr)
    assign_cps.set_cp_rules(cp_rules)

    assign_cps.assign_cps('auto')
    valid_cps = assign_cps.sel_cps_arr

    wettness = WettnessIndex()
    wettness.set_ppt_arr(in_ppt_arr_calib)
    wettness.set_cps_arr(calib_cps, n_cps)
    wettness.cmpt_wettness_idx()
    print('Calib wettness:\n', wettness.ppt_cp_wett_arr, '\n\n')

    wettness = WettnessIndex()
    wettness.set_ppt_arr(in_ppt_arr_valid)
    wettness.set_cps_arr(valid_cps, n_cps)
    wettness.cmpt_wettness_idx()
    print('Valid wettness:\n', wettness.ppt_cp_wett_arr, '\n\n')

    thresh = ThreshPPT()
    thresh.set_ppt_arr(in_ppt_arr_calib)
    thresh.set_cps_arr(calib_cps, n_cps)
    thresh.set_ge_vals_arr(o_1_ppt_thresh_arr)
    thresh.cmpt_ge_qual()
    thresh.set_le_vals_arr(o_1_ppt_thresh_arr)
    thresh.cmpt_le_qual()

    print('Calib thresh ge:\n', thresh.cp_ge_qual_arr)
    print(thresh.ppt_ge_pis_arr, '\n\n')

    print('Calib thresh le:\n', thresh.cp_le_qual_arr)
    print(thresh.ppt_le_pis_arr, '\n\n')

    thresh = ThreshPPT()
    thresh.set_ppt_arr(in_ppt_arr_valid)
    thresh.set_cps_arr(valid_cps, n_cps)
    thresh.set_ge_vals_arr(o_1_ppt_thresh_arr)
    thresh.cmpt_ge_qual()
    thresh.set_le_vals_arr(o_1_ppt_thresh_arr)
    thresh.cmpt_le_qual()

    print('valid thresh ge:\n', thresh.cp_ge_qual_arr)
    print(thresh.ppt_ge_pis_arr, '\n\n')

    print('valid thresh le:\n', thresh.cp_le_qual_arr)
    print(thresh.ppt_le_pis_arr)

    hist_plots = CPHistPlot()

    hist_plots.set_values_ser(in_ppt_df.loc[idxs_valid].iloc[:, 0])
    hist_plots.set_sel_cps_ser(pd.Series(index=in_ppt_df.loc[idxs_valid].index,
                                         data=valid_cps))
    hist_plots.set_hist_plot_prms(3,
                                  2,
                                  miss_cp_val,
                                  min_prob=None,
                                  max_prob=0.9,
                                  freq='D')

    hist_plots.cmpt_cp_hists()
    hist_plots.plot_cp_hists('Test fig',
                             'test_suff',
                             r'P:\Synchronize\IWS\2016_DFG_SPATE\data\CP_histograms_test',
                             )

    plot_cps = PlotCPs()
    plot_cps.set_epsgs(4236, 3035, 3035)

    plot_cps.set_bck_shp(backgrnd_shp_file)
    plot_cps.set_coords_arr(anomaly.x_coords, anomaly.y_coords)
    plot_cps.set_sel_cps_arr(calib_cps)
    plot_cps.set_cp_rules_arr(cp_rules)
    plot_cps.set_anoms_arr(slp_anom_calib)
    plot_cps.set_other_prms(fuzz_nos_arr, n_cps, in_coords_type='geo')

    plot_cps.krige(100)

    plot_cps.plot_kriged_cps(cont_levels,
                             r'P:\Synchronize\IWS\2016_DFG_SPATE\data\CP_histograms_test',
                             fig_size=((10, 7)))

    #==========================================================================
    #
    #==========================================================================
    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
