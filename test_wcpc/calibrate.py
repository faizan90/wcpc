'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from datetime import datetime
from pathlib import Path
from psutil import cpu_count

import numpy as np
import pandas as pd

from wcpc import (Anomaly,
                  CPAssignA,
                  CPClassiA,
                  ThreshPPT,
                  WettnessIndex,
                  CPHistPlot,
                  PlotCPs,
                  plot_iter_cp_pcntgs,
                  ObjVals,
                  plot_tri_fuzz_no,
                  RandCPsGen,
                  RandCPsPerfComp,
                  PlotDOFs,
                  PlotFuzzDOFs)

from std_logger import StdFileLoggerCtrl


def run_wcpc(args_dict):
    out_dir = args_dict['out_dir']
    if not out_dir.exists():
        out_dir.mkdir()
#     else:
#         print('\n\n\n', 40 * '#', '\n', 40 * '#', sep='')
#         print('Simulation results exist already!')
#         print(40 * '#', '\n', 40 * '#', '\n\n\n', sep='')
#         return

    with open(str(out_dir / 'args_dict.pkl'), 'wb') as _pkl_hdl:
        pickle.dump(args_dict, _pkl_hdl)

    print(out_dir)

    _out_log_file = out_dir / ('cp_classi_log_%s.log' %
                               datetime.now().strftime('%Y%m%d%H%M%S'))

    log_link = StdFileLoggerCtrl(_out_log_file)

    print('INFO: Classification started at:',
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '\n')

    print('#' * 30)
    print('#Contents of args_dict:')
    for key in args_dict:
        print('#%s:' % str(key), args_dict[key])
    print('#' * 30, '\n')

    _calib_valid_labs = ['calib', 'valid', 'all']

    anomaly_pkl_flag = args_dict['anomaly_pkl_flag']
    in_net_cdf_file = args_dict['in_net_cdf_file']
    x_coords_lab = args_dict['x_coords_lab']
    y_coords_lab = args_dict['y_coords_lab']
    time_lab = args_dict['time_lab']
    anom_var_lab = args_dict['anom_var_lab']
    anom_type = args_dict['anom_type']
    nan_anom_val = args_dict['nan_anom_val']
    eig_cum_sum_ratio = args_dict['eig_cum_sum_ratio']
    take_eig_rest_sq_sum_flag = args_dict['take_eig_rest_sq_sum_flag']
    plot_anomaly_flag = args_dict['plot_anomaly_flag']

    calib_months = args_dict['calib_months']
    valid_months = args_dict['valid_months']
    all_months = args_dict['all_months']

    _months_list = [calib_months, valid_months, all_months]

    n_cpus = args_dict['n_cpus']

    if n_cpus == 'auto':
        n_cpus = cpu_count() - 1
    else:
        assert n_cpus > 0

    valid_period_srt = args_dict['valid_period_srt']
    valid_period_end = args_dict['valid_period_end']
    time_fmt = args_dict['time_fmt']

    calib_period_srt = args_dict['calib_period_srt']
    calib_period_end = args_dict['calib_period_end']

    calib_dates = pd.to_datetime([calib_period_srt, calib_period_end],
                                 format=time_fmt)

    valid_dates = pd.to_datetime([valid_period_srt, valid_period_end],
                                 format=time_fmt)

    _min_time = min(min(calib_dates), min(valid_dates))
    _max_time = max(max(calib_dates), max(valid_dates))

    _ca_va_dates_strs_list = [[calib_period_srt, calib_period_end],
                              [valid_period_srt, valid_period_end],
                              [_min_time.strftime(time_fmt),
                               _max_time.strftime(time_fmt)]]

    ft_ll_steps_thresh = args_dict['ft_ll_steps_thresh']
    ft_ul_steps_thresh = args_dict['ft_ul_steps_thresh']

    #==========================================================================
    # Anomaly Start
    #==========================================================================
    anomaly_prms_pkl = out_dir / 'anomaly.pkl'
    _anoms_list = []  # slp_anom_calib, slp_anom_valid, vals_tot_anom
    _plot_anoms_list = []
#     _old_anoms_list = []
    _sel_cp_times_list = []

    if anomaly_pkl_flag or (not anomaly_prms_pkl.exists()):
        anomaly = Anomaly()

        anomaly.read_vars(in_net_cdf_file,
                          x_coords_lab,
                          y_coords_lab,
                          time_lab,
                          anom_var_lab,
                          'nc',
                          'D')

        with open(anomaly_prms_pkl, 'wb') as _pkl_hdl:
            pickle.dump(anomaly, _pkl_hdl)
    else:
        with open(anomaly_prms_pkl, 'rb') as _pkl_hdl:
            anomaly = pickle.load(_pkl_hdl)

    for i in range(len(_calib_valid_labs)):
        print(_calib_valid_labs[i])

        if plot_anomaly_flag:
            plot_anomaly_dir = (out_dir /
                                ('anomaly_cdfs_%s' % _calib_valid_labs[i]))
        else:
            plot_anomaly_dir = None

        if anom_type == 'b':
            anomaly.calc_anomaly_type_b(_ca_va_dates_strs_list[i][0],
                                        _ca_va_dates_strs_list[i][1],
                                        _months_list[i],
                                        time_fmt,
                                        nan_anom_val,
                                        fig_out_dir=plot_anomaly_dir,
                                        n_cpus=n_cpus)
        elif anom_type == 'c':
            anomaly.calc_anomaly_type_c(_ca_va_dates_strs_list[i][0],
                                        _ca_va_dates_strs_list[i][1],
                                        _months_list[i],
                                        time_fmt,
                                        nan_anom_val,
                                        fig_out_dir=plot_anomaly_dir,
                                        n_cpus=n_cpus)
        elif anom_type == 'd':
            anomaly.calc_anomaly_type_d(_ca_va_dates_strs_list[i][0],
                                        _ca_va_dates_strs_list[i][1],
                                        _ca_va_dates_strs_list[-1][0],
                                        _ca_va_dates_strs_list[-1][1],
                                        _months_list[i],
                                        time_fmt,
                                        nan_anom_val,
                                        eig_cum_sum_ratio=eig_cum_sum_ratio,
                                        eig_sum_flag=take_eig_rest_sq_sum_flag,
                                        fig_out_dir=plot_anomaly_dir,
                                        n_cpus=n_cpus)

        elif anom_type == 'f':
            anomaly.calc_anomaly_type_f(_ca_va_dates_strs_list[i][0],
                                        _ca_va_dates_strs_list[i][1],
                                        _ca_va_dates_strs_list[-1][0],
                                        _ca_va_dates_strs_list[-1][1],
                                        _months_list[i],
                                        ft_ll_steps_thresh,
                                        ft_ul_steps_thresh,
                                        time_fmt,
                                        nan_anom_val,
                                        fig_out_dir=plot_anomaly_dir,
                                        n_cpus=n_cpus)

        else:
            raise ValueError('Incorrect anom_type!')

        if (anom_type == 'd') or (anom_type == 'e'):
            _anoms_list.append(anomaly.vals_anom)
            _plot_anoms_list.append(anomaly.vals_anom_for_cp_plots)
        else:
            _anoms_list.append(anomaly.vals_tot_anom)
            _plot_anoms_list.append(anomaly.vals_tot_anom)

        _sel_cp_times_list.append(anomaly.times)

    dates_tot = _sel_cp_times_list[-1]
    dates_tot = pd.DatetimeIndex(dates_tot.date)
    with open(out_dir / 'calib_valid_all_time_idxs.pkl', 'wb') as _pkl_hdl:
        pickle.dump([pd.DatetimeIndex(_) for _ in _sel_cp_times_list],
                    _pkl_hdl)

#     raise Exception
    #==========================================================================
    # Anomaly End
    #==========================================================================

    #==========================================================================
    # Prep Input Start
    #==========================================================================

    in_ppt_df_pkl = args_dict['in_ppt_df_pkl']
#     in_wett_nebs_pkl = args_dict['in_wett_nebs_pkl']
    in_cats_ppt_df_pkl = args_dict['in_cats_ppt_df_pkl']
    in_cats_dis_df_pkl = args_dict['in_cats_dis_df_pkl']
#     in_lorenz_df_pkl = args_dict['in_lorenz_df_pkl']
#     in_nebs_stns_pkl = args_dict['in_nebs_stns_pkl']

    dis_cat_ext_evts = args_dict['dis_cat_ext_evts']

    in_ppt_df = pd.read_pickle(in_ppt_df_pkl)
    in_ppt_df = in_ppt_df.loc[dates_tot]

#     in_wettness_df = pd.read_pickle(in_wett_nebs_pkl)
#     in_wettness_df = in_wettness_df.loc[dates_tot]

    in_cats_ppt_df = pd.read_pickle(in_cats_ppt_df_pkl)
    in_cats_ppt_df = in_cats_ppt_df.loc[dates_tot]

    in_cats_dis_df = pd.read_pickle(in_cats_dis_df_pkl)
    in_cats_dis_df = in_cats_dis_df.loc[dates_tot]

#     in_lorenz_df = pd.read_pickle(in_lorenz_df_pkl)
#     in_lorenz_df = in_lorenz_df.loc[dates_tot]

    o_2_ppt_thresh_arr = args_dict['o_2_ppt_thresh_arr']

    hist_cat_ser = pd.Series(in_cats_dis_df.loc[:, dis_cat_ext_evts].copy())
    hist_cat_ser.iloc[1:] = hist_cat_ser.values[1:] - hist_cat_ser.values[:1]
    hist_cat_ser.iloc[0] = 0

    hist_cat_ser.loc[hist_cat_ser.values < 0] = 0

    #==========================================================================
    # Normalize cat ppt w.r.t area Start
    #==========================================================================
#     cats_areas_df_path = args_dict['cats_areas_df_path']
#     sep = args_dict['sep']
#     cats_areas_df = pd.read_csv(cats_areas_df_path, sep=sep, index_col=0)
#     cats_areas_df = cats_areas_df.loc[[int(_) for _ in in_cats_ppt_df.columns]]
#
#     _max_area_idx = cats_areas_df.loc[:, 'cumm_area'].idxmax()
#     _max_area = cats_areas_df.loc[_max_area_idx, 'cumm_area']
#     _area_ratios = cats_areas_df.loc[:, 'diff_area'].values / _max_area
#     assert np.isclose(_area_ratios.sum(), 1.0)
#     in_cats_ppt_df = pd.DataFrame((in_cats_ppt_df * _area_ratios).sum(axis=1))
#
#     _cats_maxes = in_cats_ppt_df.max(axis=0).values
#     _max_cat_ppt = _cats_maxes.min()
#     _o_2_max_thresh = o_2_ppt_thresh_arr.max()
#     if np.any(_cats_maxes <  _o_2_max_thresh):
#         print('\n####Rescaling o_2_ppt_thresh_arr####!')
#         _rescale_ratio = 0.8 * (_max_cat_ppt / _o_2_max_thresh)
#         print('old o_2_ppt_thresh_arr:\n', o_2_ppt_thresh_arr)
#         print('recalse ratio:', _rescale_ratio)
#         o_2_ppt_thresh_arr = _rescale_ratio * o_2_ppt_thresh_arr
#         print('new o_2_ppt_thresh_arr:\n', o_2_ppt_thresh_arr)
#         print('\n')
    #==========================================================================
    # Normalize cat ppt w.r.t area End
    #==========================================================================

#     rand_idxs = np.random.randint(0, in_ppt_df.columns.shape[0], 100)
#     rand_ppt_stns = in_ppt_df.columns[rand_idxs]

#     rand_ppt_stns = ['P1197', 'P891', 'P1468', 'P232', 'P3015', 'P2497',
#                      'P4169', 'P2211', 'P2542', 'P403', 'P3032', 'P1197',
#                      'P2261', 'P3527', 'P3987']

#     rand_ppt_stns = ['P3176', 'P5711', 'P4926']

#     rand_cats_idxs = np.random.randint(0, in_cats_ppt_df.columns.shape[0], 15)
#     rand_ppt_cats = in_cats_ppt_df.columns[rand_cats_idxs]
#     rand_ppt_cats = ['1458', '460', '411', '76123', '420', '422', '2489',
#                      '473', '1458', '420', '434', '406', '2477', '420', '3465']
#     rand_ppt_cats = ['427', '4416']

    # when cats
#     rand_lors_stns = ['1458', '460', '411', '76123', '420', '422', '2489',
#                      '473', '1458', '420', '434', '406', '2477', '420', '3465']

#     # when stns
#     rand_lors_stns = ['P1197', 'P891', 'P1468', 'P232', 'P3015', 'P2497',
#                      'P4169', 'P2211', 'P2542', 'P403', 'P3032', 'P1197',
#                      'P2261', 'P3527', 'P3987']

    # when nebs
#     all_lor_stns = []
#     with open(in_nebs_stns_pkl, 'rb') as _hdl:
#         neb_idx = '0'
#         print('Using neiborhood number: %s' % neb_idx)
#         nebs_stn_names_dict = pickle.load(_hdl)
#         rand_lors_stns = nebs_stn_names_dict[neb_idx]

#         [all_lor_stns.extend(_) for _ in nebs_stn_names_dict.values()]

#     in_ppt_df = in_ppt_df[rand_ppt_stns]
#     in_cats_ppt_df = in_cats_ppt_df[rand_ppt_cats]
#     in_cats_dis_df = in_cats_dis_df[rand_ppt_cats]
#     in_lorenz_df = in_lorenz_df[rand_lors_stns]

    idxs_calib = _sel_cp_times_list[0]

#     in_ppt_lorenz_stns_df = in_ppt_df.loc[idxs_calib, all_lor_stns]
#     in_ppt_lorenz_stns_df.to_pickle('ppt_df_lorenz_nebs_calib.pkl')

#     in_ppt_calib_df = in_ppt_df.loc[idxs_calib]
#     in_ppt_calib_df.to_pickle('ppt_df_stns_calib.pkl')

#     in_cats_ppt_calib_df = in_cats_ppt_df.loc[idxs_calib]
#     in_cats_ppt_calib_df.to_pickle('cats_ppt_df_stns_calib.pkl')

    use_dis_diff = False
    if use_dis_diff:
        print('\n####### using discharge differences in obj 2!########')
        in_dis_diff_df = in_cats_dis_df.copy()
        in_dis_diff_df.iloc[1:, :] = (in_dis_diff_df.values[1:, :] -
                                      in_dis_diff_df.values[:1, :])
        in_dis_diff_df.iloc[0, :] = 0
        in_dis_diff_df[in_dis_diff_df < 0] = 0

        in_dis_diff_df = 100 * in_dis_diff_df / in_dis_diff_df.max()
        in_cats_ppt_df = in_dis_diff_df

        in_dis_diff_df.to_pickle(str(out_dir / 'dis_diff_df.pkl'))

#         raise Exception

    in_ppt_arr_calib = in_ppt_df.loc[idxs_calib].values.copy(order='C')
    slp_anom_calib = _anoms_list[0].copy(order='C')
#     in_wet_arr_calib = in_wettness_df.loc[idxs_calib].values.copy(
#         order='C')
    in_cats_ppt_arr_calib = in_cats_ppt_df.loc[idxs_calib].values.copy(
        order='C')
#     in_lorenz_arr_calib = in_lorenz_df.loc[idxs_calib].values.copy(
#         order='C')

    idxs_valid = _sel_cp_times_list[1]

#     if use_dis_diff:
# #         print('\n####### using discharge differences in obj 2!########')
#         in_dis_diff_df = in_cats_dis_df.loc[idxs_valid].copy()
#         in_dis_diff_df.iloc[1:, :] = (in_dis_diff_df.values[1:, :] -
#                                       in_dis_diff_df.values[:1, :])
#         in_dis_diff_df.iloc[0, :] = 0
#         in_dis_diff_df[in_dis_diff_df < 0] = 0
#
#         in_dis_diff_df = 100 * in_dis_diff_df / in_dis_diff_df.max()
#         in_cats_ppt_df.loc[idxs_valid] = in_dis_diff_df

    in_ppt_arr_valid = in_ppt_df.loc[idxs_valid].values.copy(order='C')
#     slp_anom_valid = _anoms_list[1].copy(order='C')
#     in_wet_arr_valid = in_wettness_df.loc[idxs_valid].values.copy(
#         order='C')
    in_cats_ppt_arr_valid = in_cats_ppt_df.loc[idxs_valid].values.copy(
        order='C')

#     in_lorenz_arr_valid = in_lorenz_df.loc[idxs_valid].values.copy(
#         order='C')

    if use_dis_diff:
        _min = np.vstack((in_cats_ppt_arr_calib.max(axis=0),
                          in_cats_ppt_arr_valid.max(axis=0))).min()
        o_2_ppt_thresh_arr = np.linspace(0, _min - 1, 30)

    _idxs_list = [idxs_calib,
                  idxs_valid,
                  np.ones(in_ppt_df.shape[0], dtype=bool)]

    _ppt_arrs_list = [in_ppt_arr_calib,
                      in_ppt_arr_valid,
                      in_ppt_df.values]
    #==========================================================================
    # Prep Input End
    #==========================================================================

    #==========================================================================
    # Classification Start
    #==========================================================================

    cp_classi_pkl_flag = args_dict['cp_classi_pkl_flag']
    cp_classi_pkl = out_dir / 'cp_classi.pkl'
    n_cps = args_dict['n_cps']
    max_idxs_ct = args_dict['max_idxs_ct']
    no_cp_val = args_dict['no_cp_val']
    miss_cp_val = args_dict['miss_cp_val']
    p_l = args_dict['p_l']
    fuzz_nos_arr = args_dict['fuzz_nos_arr']
    out_fuzz_no_fig_path = out_dir / r'fuzz_no.png'

    _min_tot_anom = _anoms_list[0].min()
    _max_tot_anom = _anoms_list[0].max()

    if fuzz_nos_arr[0, 0] > _min_tot_anom:
        print('####Changed the least value of fuzz_nos_arr!####')
        fuzz_nos_arr[0, 0] = (np.sign(_min_tot_anom) *
                              abs(_min_tot_anom * 1.1))

    if fuzz_nos_arr[-1, -1] < _max_tot_anom:
        print('####Changed the highest value of fuzz_nos_arr!####')
        fuzz_nos_arr[-1, -1] = (np.sign(_max_tot_anom) *
                                abs(_max_tot_anom * 1.1))

    # not the best way to do this
    assert fuzz_nos_arr.min() <= fuzz_nos_arr[0, 0]
    assert fuzz_nos_arr.max() >= fuzz_nos_arr[-1, -1]

    plot_tri_fuzz_no(fuzz_nos_arr, out_fuzz_no_fig_path)
#     raise Exception

    obj_1_flag = args_dict['obj_1_flag']
    o_1_ppt_thresh_arr = args_dict['o_1_ppt_thresh_arr']
    obj_1_wt = args_dict['obj_1_wt']

    obj_2_flag = args_dict['obj_2_flag']
    obj_2_wt = args_dict['obj_2_wt']

    obj_3_flag = args_dict['obj_3_flag']
    obj_3_wt = args_dict['obj_3_wt']

    obj_4_flag = args_dict['obj_4_flag']
    o_4_wett_thresh_arr = args_dict['o_4_wett_thresh_arr']
    obj_4_wt = args_dict['obj_4_wt']

    obj_5_flag = args_dict['obj_5_flag']
    obj_5_wt = args_dict['obj_5_wt']

    obj_6_flag = args_dict['obj_6_flag']
    obj_6_wt = args_dict['obj_6_wt']
    obj_6_wett_thresh = args_dict['obj_6_wett_thresh']

    obj_7_flag = args_dict['obj_7_flag']
    obj_7_wt = args_dict['obj_7_wt']

    obj_8_flag = args_dict['obj_8_flag']
    obj_8_wt = args_dict['obj_8_wt']

    lo_freq_pen_wt = args_dict['lo_freq_pen_wt']
    min_freq = args_dict['min_freq']

    anneal_temp_ini = args_dict['anneal_temp_ini']
    temp_red_alpha = args_dict['temp_red_alpha']
    max_m_iters = args_dict['max_m_iters']
    max_n_iters = args_dict['max_n_iters']
    max_k_iters = args_dict['max_k_iters']
    temp_adj_iters = args_dict['temp_adj_iters']
    sort_cps_flag = args_dict['sort_cps_flag']
    plot_cp_freq_iter_flag = args_dict['plot_cp_freq_iter_flag']

    op_mp_memb_flag = args_dict['op_mp_memb_flag']
    op_mp_obj_ftn_flag = args_dict['op_mp_obj_ftn_flag']
    verif_plots_flag = args_dict['verif_plots_flag']
    no_steep_anom_flag = args_dict['no_steep_anom_flag']

    if no_steep_anom_flag and (anom_type == 'd'):
        no_steep_anom_flag = False
        print('\n###no_steep_anom_flag set to False due to anom_type d!###\n')

    out_obj_vals_evo_fig_path = out_dir / 'obj_vals_evolution.png'
    out_cp_freq_evo_fig_path = out_dir / 'cp_freq_evolution.png'

    if cp_classi_pkl_flag or (not cp_classi_pkl.exists()):
        classi = CPClassiA()
        classi.set_stn_ppt(in_ppt_arr_calib)
        classi.set_cat_ppt(in_cats_ppt_arr_calib)
#         classi.set_neb_wett(in_wet_arr_calib)
#         classi.set_lorenz_arr(in_lorenz_arr_calib)
        classi.set_cp_prms(n_cps,
                           max_idxs_ct,
                           no_cp_val,
                           miss_cp_val,
                           p_l,
                           fuzz_nos_arr,
                           lo_freq_pen_wt,
                           min_freq)

        if obj_1_flag:
            classi.set_obj_1_on(o_1_ppt_thresh_arr, obj_1_wt)
        if obj_2_flag:
            classi.set_obj_2_on(o_2_ppt_thresh_arr, obj_2_wt)
        if obj_3_flag:
            classi.set_obj_3_on(obj_3_wt)
        if obj_4_flag:
            classi.set_obj_4_on(o_4_wett_thresh_arr, obj_4_wt)
        if obj_5_flag:
            classi.set_obj_5_on(obj_5_wt)
        if obj_6_flag:
            classi.set_obj_6_on(obj_6_wt, obj_6_wett_thresh)
        if obj_7_flag:
            classi.set_obj_7_on(obj_7_wt)
        if obj_8_flag:
            classi.set_obj_8_on(obj_8_wt)

        classi.set_cyth_flags(cyth_nonecheck=False,
                              cyth_boundscheck=False,
                              cyth_wraparound=False,
                              cyth_cdivision=True,
                              cyth_language_level=3,
                              cyth_infer_types=None)

#         classi.set_cyth_flags(cyth_nonecheck=True,
#                               cyth_boundscheck=True,
#                               cyth_wraparound=True,
#                               cyth_cdivision=True,
#                               cyth_language_level=3,
#                               cyth_infer_types=None)

        classi.set_anomaly(slp_anom_calib,
                           anomaly.vals_tot.shape[1],
                           anomaly.vals_tot.shape[2])

        classi.set_sim_anneal_prms(anneal_temp_ini,
                                   temp_red_alpha,
                                   max_m_iters,
                                   max_n_iters,
                                   max_k_iters,
                                   temp_adj_iters=temp_adj_iters,
                                   min_acc_rate=60,
                                   max_acc_rate=80,)

        classi.op_mp_memb_flag = op_mp_memb_flag
        classi.op_mp_obj_ftn_flag = op_mp_obj_ftn_flag
        classi.no_steep_anom_flag = no_steep_anom_flag

        classi.classify(n_cpus, force_compile=False)
#         classi.classify(n_cpus, force_compile=True)

        with open(cp_classi_pkl, 'wb') as _pkl_hdl:
            pickle.dump(classi, _pkl_hdl)

        classi.plot_iter_obj_vals(out_obj_vals_evo_fig_path)

    else:
        with open(cp_classi_pkl, 'rb') as _pkl_hdl:
            classi = pickle.load(_pkl_hdl)

    if verif_plots_flag:
        classi.plot_verifs(out_dir)

    cp_rules = classi.cp_rules

    if anom_type != 'd':
        for i in range(n_cps):
            print(cp_rules[i, :].reshape(anomaly.vals_tot.shape[1],
                                         anomaly.vals_tot.shape[2]), '\n')

#     raise Exception

    if sort_cps_flag or plot_cp_freq_iter_flag:
        print('Reordering CPs based on Wettness index...')
        wettness = WettnessIndex()

#         if obj_2_flag or obj_5_flag:
#             wettness.set_ppt_arr(in_cats_ppt_arr_calib)
#         else:
#             wettness.set_ppt_arr(in_ppt_arr_calib)

        wettness.set_ppt_arr(in_ppt_arr_calib)

        wettness.set_cps_arr(classi.calib_dict['best_sel_cps'], n_cps)
        wettness.reorder_cp_rules(cp_rules)

        if sort_cps_flag:
            map_ = wettness.old_new_cp_map_arr
            cp_rules = wettness.cp_rules_sorted
        else:
            map_ = None

        print('old_new_cp_map:\n', map_)

        if plot_cp_freq_iter_flag:
            plot_iter_cp_pcntgs(n_cps,
                                classi.curr_n_iters_arr,
                                classi.cp_pcntge_arr,
                                out_cp_freq_evo_fig_path,
                                classi.calib_dict['last_best_accept_n_iter'],
                                map_)

    #==========================================================================
    # Calssification End
    #==========================================================================

    #==========================================================================
    # Assign CPs Start
    #==========================================================================

    cp_assign_pkls = out_dir / 'cp_assign_%s.pkl'
    cp_assign_pkl_flag = args_dict['cp_assign_pkl_flag']

    _sel_cp_rules_list = []
    _dofs_arr_list = []
    _fuzz_dofs_arr_list = []

    for i in range(len(_calib_valid_labs)):
        print(_calib_valid_labs[i])
        _ = Path(str(cp_assign_pkls) % _calib_valid_labs[i])
        if cp_assign_pkl_flag or (not _.exists()):
            assign_cps = CPAssignA()
            assign_cps.set_anomaly(_anoms_list[i],
                                   anomaly.vals_tot.shape[1],
                                   anomaly.vals_tot.shape[2])
            assign_cps.set_cp_prms(n_cps,
                                   max_idxs_ct,
                                   no_cp_val,
                                   miss_cp_val,
                                   p_l,
                                   fuzz_nos_arr,
                                   lo_freq_pen_wt,
                                   min_freq)
            assign_cps.set_cp_rules(cp_rules)
            assign_cps.op_mp_memb_flag = op_mp_memb_flag
            assign_cps.op_mp_obj_ftn_flag = op_mp_obj_ftn_flag

            assign_cps.set_cyth_flags(cyth_nonecheck=False,
                                      cyth_boundscheck=False,
                                      cyth_wraparound=False,
                                      cyth_cdivision=True,
                                      cyth_language_level=3,
                                      cyth_infer_types=None)

#             assign_cps.set_cyth_flags(cyth_nonecheck=True,
#                                       cyth_boundscheck=True,
#                                       cyth_wraparound=True,
#                                       cyth_cdivision=True,
#                                       cyth_language_level=3,
#                                       cyth_infer_types=None)

            assign_cps.assign_cps(n_cpus, force_compile=False)
#             assign_cps.assign_cps(n_cpus, force_compile=True)

            with open(_, 'wb') as _pkl_hdl:
                pickle.dump(assign_cps, _pkl_hdl)
        else:
            with open(_, 'rb') as _pkl_hdl:
                assign_cps = pickle.load(_pkl_hdl)

        _sel_cp_rules_list.append(assign_cps.sel_cps_arr)
        _dofs_arr_list.append(assign_cps.dofs_arr)
        _fuzz_dofs_arr_list.append(assign_cps.fuzz_dofs_arr)

    if verif_plots_flag:
        plot_dofs = PlotDOFs()
        plot_dofs.set_calib_dofs(_dofs_arr_list[0])
        plot_dofs.set_valid_dofs(_dofs_arr_list[1])
        plot_dofs.set_all_dofs(_dofs_arr_list[2])
        plot_dofs.plot_verifs(out_dir)

        plot_fuzz_dofs = PlotFuzzDOFs()
        plot_fuzz_dofs.set_calib_dofs(_fuzz_dofs_arr_list[0])
        plot_fuzz_dofs.set_valid_dofs(_fuzz_dofs_arr_list[1])
        plot_fuzz_dofs.set_all_dofs(_fuzz_dofs_arr_list[2])
        plot_fuzz_dofs.plot_verifs(out_dir)

    #==========================================================================
    # Assign CPs End
    #==========================================================================

    #==========================================================================
    # RandCPs Start
    #==========================================================================
    compare_rand_flag = args_dict['compare_rand_flag']

    rand_cps_gen_pkl = out_dir / 'rand_cps_gen.pkl'
    if compare_rand_flag:
        _1 = Path(str(rand_cps_gen_pkl))
#         if (not rand_cps_gen_pkl.exists()):
        print('\nComparing random CPs...')
        comp_rand_cps_pkls = out_dir / 'comp_rand_cps_%s.pkl'

        n_gens = args_dict['n_rand_cp_gens']
        n_sim_cp_gens = args_dict['n_sim_cp_gens']
        n_nrst_cps = args_dict['n_nrst_cps']

        _rand_cps_comp_list = []

        rand_cps_gen = RandCPsGen()

        rand_cps_gen.set_cyth_flags(cyth_nonecheck=False,
                                    cyth_boundscheck=False,
                                    cyth_wraparound=False,
                                    cyth_cdivision=True,
                                    cyth_language_level=3,
                                    cyth_infer_types=None)

#             rand_cps_gen.set_cyth_flags(cyth_nonecheck=True,
#                                         cyth_boundscheck=True,
#                                         cyth_wraparound=True,
#                                         cyth_cdivision=True,
#                                         cyth_language_level=3,
#                                         cyth_infer_types=None)

        rand_cps_gen.op_mp_memb_flag = op_mp_memb_flag
        rand_cps_gen.op_mp_obj_ftn_flag = op_mp_obj_ftn_flag

        rand_cps_gen.gen_cp_rules(n_cps,
                                  n_gens,
                                  max_idxs_ct,
                                  _anoms_list[0].shape[1],
                                  p_l,
                                  no_cp_val,
                                  fuzz_nos_arr,
                                  _anoms_list[0],
                                  anomaly.vals_tot.shape[1],
                                  anomaly.vals_tot.shape[2],
                                  no_steep_anom_flag,
                                  n_threads=n_cpus,
                                  force_compile=False)

        mult_cp_rules = rand_cps_gen.mult_cp_rules
#         mult_sel_cps = rand_cps_gen.mult_sel_cps

        with open(_1, 'wb') as _pkl_hdl:
            pickle.dump(rand_cps_gen, _pkl_hdl)

        _strt = timeit.default_timer()
        for i in range(len(_calib_valid_labs)):
            print(_calib_valid_labs[i])
            _2 = Path(str(comp_rand_cps_pkls) % _calib_valid_labs[i])
#             if not _2.exists():
            assign_cps = CPAssignA()
            assign_cps.set_anomaly(_anoms_list[i],
                                   anomaly.vals_tot.shape[1],
                                   anomaly.vals_tot.shape[2])
            assign_cps.set_cp_prms(n_cps,
                                   max_idxs_ct,
                                   no_cp_val,
                                   miss_cp_val,
                                   p_l,
                                   fuzz_nos_arr,
                                   lo_freq_pen_wt,
                                   min_freq)
            assign_cps.set_mult_cp_rules(mult_cp_rules)
            assign_cps.op_mp_memb_flag = op_mp_memb_flag
            assign_cps.no_steep_anom_flag = no_steep_anom_flag

            assign_cps.assign_mult_cps(n_cpus, force_compile=False)
#             assign_cps.assign_mult_cps(n_cpus, force_compile=True)

            assign_cps.set_sim_sel_cps_dofs_arr(_dofs_arr_list[i])
            assign_cps.sim_sel_cps(n_sim_cp_gens, n_nrst_cps)

            rand_cps_comp_obj = RandCPsPerfComp()
            rand_cps_comp_obj.set_mult_cp_rules(mult_cp_rules)
            rand_cps_comp_obj.set_mult_sel_cps_arr(assign_cps.mult_sel_cps_arr)

            rand_cps_comp_obj.set_sim_sel_cps_arr(assign_cps.sim_sel_cps_arr)

            rand_cps_comp_obj.set_stn_ppt(
                in_ppt_df.loc[_idxs_list[i]].values)
            rand_cps_comp_obj.set_cat_ppt(
                in_cats_ppt_df.loc[_idxs_list[i]].values)
#             rand_cps_comp_obj.set_neb_wett(
#                 in_wettness_df.loc[_idxs_list[i]].values)
#             rand_cps_comp_obj.set_lorenz_arr(
#                 in_lorenz_df.loc[_idxs_list[i]].values)
            rand_cps_comp_obj.set_cps_arr(
                _sel_cp_rules_list[i], n_cps)

            rand_cps_comp_obj.set_cp_prms(n_cps,
                                          max_idxs_ct,
                                          no_cp_val,
                                          miss_cp_val,
                                          p_l,
                                          fuzz_nos_arr,
                                          lo_freq_pen_wt,
                                          min_freq)

            if obj_1_flag:
                rand_cps_comp_obj.set_obj_1_on(o_1_ppt_thresh_arr,
                                               obj_1_wt)
            if obj_2_flag:
                rand_cps_comp_obj.set_obj_2_on(o_2_ppt_thresh_arr,
                                               obj_2_wt)
            if obj_3_flag:
                rand_cps_comp_obj.set_obj_3_on(obj_3_wt)
            if obj_4_flag:
                rand_cps_comp_obj.set_obj_4_on(o_4_wett_thresh_arr,
                                               obj_4_wt)
            if obj_5_flag:
                rand_cps_comp_obj.set_obj_5_on(obj_5_wt)
            if obj_6_flag:
                rand_cps_comp_obj.set_obj_6_on(obj_6_wt, obj_6_wett_thresh)
            if obj_7_flag:
                rand_cps_comp_obj.set_obj_7_on(obj_7_wt)
            if obj_8_flag:
                rand_cps_comp_obj.set_obj_8_on(obj_8_wt)

            rand_cps_comp_obj.set_cyth_flags(cyth_nonecheck=False,
                                             cyth_boundscheck=False,
                                             cyth_wraparound=False,
                                             cyth_cdivision=True,
                                             cyth_language_level=3,
                                             cyth_infer_types=None)

    #             rand_cps_comp_obj.set_cyth_flags(cyth_nonecheck=True,
    #                                         cyth_boundscheck=True,
    #                                         cyth_wraparound=True,
    #                                         cyth_cdivision=True,
    #                                         cyth_language_level=3,
    #                                         cyth_infer_types=None)

            rand_cps_comp_obj.op_mp_obj_ftn_flag = op_mp_obj_ftn_flag

            rand_cps_comp_obj.cmpt_mult_obj_val(n_cpus, force_compile=False)
#             rand_cps_comp_obj.cmpt_mult_obj_val(n_cpus, force_compile=True)

            rand_cps_comp_obj.cmpt_sim_obj_val(n_cpus, force_compile=False)

            rand_cps_comp_obj.cmpt_mult_wettnesses(_ppt_arrs_list[i])
            rand_cps_comp_obj.cmpt_sim_wettnesses(_ppt_arrs_list[i])

            with open(_2, 'wb') as _pkl_hdl:
                pickle.dump(rand_cps_comp_obj, _pkl_hdl)
#             else:
#                 with open(_2, 'rb') as _pkl_hdl:
#                     _rand_cps_comp_list = pickle.load(_pkl_hdl)

            _rand_cps_comp_list.append(rand_cps_comp_obj)

        _stop = timeit.default_timer()
        print('Total time for rand_comps: %0.2f secs\n' % (_stop - _strt))
    #==========================================================================
    # RandCPs End
    #==========================================================================

    #==========================================================================
    # Obj Vals Start
    #==========================================================================
    _obj_vals_list = []
    print('\nCalculating objective function values...')
    for i in range(len(_calib_valid_labs)):
        print(_calib_valid_labs[i])
        obj_vals_obj = ObjVals()
        obj_vals_obj.set_stn_ppt(in_ppt_df.loc[_idxs_list[i]].values)
        obj_vals_obj.set_cat_ppt(in_cats_ppt_df.loc[_idxs_list[i]].values)
#         obj_vals_obj.set_neb_wett(in_wettness_df.loc[_idxs_list[i]].values)
#         obj_vals_obj.set_lorenz_arr(in_lorenz_df.loc[_idxs_list[i]].values)
        obj_vals_obj.set_cps_arr(_sel_cp_rules_list[i], n_cps)

        obj_vals_obj.set_cp_prms(n_cps,
                                 max_idxs_ct,
                                 no_cp_val,
                                 miss_cp_val,
                                 p_l,
                                 fuzz_nos_arr,
                                 lo_freq_pen_wt,
                                 min_freq)

        if obj_1_flag:
            obj_vals_obj.set_obj_1_on(o_1_ppt_thresh_arr, obj_1_wt)
        if obj_2_flag:
            obj_vals_obj.set_obj_2_on(o_2_ppt_thresh_arr, obj_2_wt)
        if obj_3_flag:
            obj_vals_obj.set_obj_3_on(obj_3_wt)
        if obj_4_flag:
            obj_vals_obj.set_obj_4_on(o_4_wett_thresh_arr, obj_4_wt)
        if obj_5_flag:
            obj_vals_obj.set_obj_5_on(obj_5_wt)
        if obj_6_flag:
            obj_vals_obj.set_obj_6_on(obj_6_wt, obj_6_wett_thresh)
        if obj_7_flag:
            obj_vals_obj.set_obj_7_on(obj_7_wt)
        if obj_8_flag:
            obj_vals_obj.set_obj_8_on(obj_8_wt)

        obj_vals_obj.set_cyth_flags(cyth_nonecheck=False,
                                    cyth_boundscheck=False,
                                    cyth_wraparound=False,
                                    cyth_cdivision=True,
                                    cyth_language_level=3,
                                    cyth_infer_types=None)

    #     obj_vals_obj.set_cyth_flags(cyth_nonecheck=True,
    #                                 cyth_boundscheck=True,
    #                                 cyth_wraparound=True,
    #                                 cyth_cdivision=True,
    #                                 cyth_language_level=3,
    #                                 cyth_infer_types=None)

        obj_vals_obj.op_mp_memb_flag = op_mp_memb_flag
        obj_vals_obj.op_mp_obj_ftn_flag = op_mp_obj_ftn_flag

        obj_vals_obj.cmpt_obj_val(n_cpus, force_compile=False)
#         obj_vals_obj.cmpt_obj_val(n_cpus, force_compile=True)

        _obj_vals_list.append(obj_vals_obj.obj_val)

        if compare_rand_flag:
            _rand_cps_comp_list[i].compare_mult_obj_vals(
                _obj_vals_list[i],
                (out_dir / ('comp_mult_obj_vals_%s.png' % _calib_valid_labs[i])))
            _rand_cps_comp_list[i].compare_sim_obj_vals(
                _obj_vals_list[i],
                (out_dir / ('comp_sim_obj_vals_%s.png' % _calib_valid_labs[i])))
    #==========================================================================
    # Obj Vals End
    #==========================================================================

#     if obj_2_flag or obj_5_flag:
#         _ppt_arrs_list = [in_cats_ppt_arr_calib,
#                           in_cats_ppt_arr_valid,
#                           in_cats_ppt_df.values]
#     else:
#         _ppt_arrs_list = [in_ppt_arr_calib,
#                           in_ppt_arr_valid,
#                           in_ppt_df.values]

    #==========================================================================
    # Wettness Start
    #==========================================================================
    wettness_idx_pkl_flag = args_dict['wettness_idx_pkl_flag']

    if wettness_idx_pkl_flag:
        wettness_idx_pkls = out_dir / 'wettness_idxs_%s.pkl'
        # wett_label = args_dict['wett_label']
        _mean_wett_arrs = []

        for i in range(len(_calib_valid_labs)):
            print(_calib_valid_labs[i])
            _ = Path(str(wettness_idx_pkls) % _calib_valid_labs[i])
            if not _.exists():
                wettness = WettnessIndex()
                wettness.set_ppt_arr(_ppt_arrs_list[i])
                wettness.set_cps_arr(_sel_cp_rules_list[i], n_cps)
                wettness.cmpt_wettness_idx()

                with open(_, 'wb') as _pkl_hdl:
                    pickle.dump(wettness, _pkl_hdl)
            else:
                with open(_, 'rb') as _pkl_hdl:
                    wettness = pickle.load(_pkl_hdl)

            _mean_wett_arr = wettness.mean_cp_wett_arr
            _mean_wett_arrs.append(_mean_wett_arr)

            print('Mean %s wettness:\n' % _calib_valid_labs[i],
                  _mean_wett_arr,
                  '\n\n')

#             wettness.plot_wettness(wett_label,
#                                    out_dir / ('wettness_idxs_%s.png' %
#                                               _calib_valid_labs[i]),
#                                    obj_val=_obj_vals_list[i])

            if compare_rand_flag:
                _rand_cps_comp_list[i].compare_mult_wettnesses(
                    _mean_wett_arrs[i],
                    (out_dir / ('comp_mult_wett_%s.png' % _calib_valid_labs[i])))

                _rand_cps_comp_list[i].compare_sim_wettnesses(
                    _mean_wett_arrs[i],
                    (out_dir / ('comp_sim_wett_%s.png' % _calib_valid_labs[i])))

        wettness.plot_wettness_list(_mean_wett_arrs,
                                    n_cps,
                                    _calib_valid_labs,
                                    (out_dir / 'wettness_idxs.png'),
                                    _obj_vals_list)

    #==========================================================================
    # Wettness End
    #==========================================================================

    #==========================================================================
    # Thresh Start
    #==========================================================================
    thresh_ppt_pkl_flag = args_dict['thresh_ppt_pkl_flag']
    if thresh_ppt_pkl_flag:
        thresh_ppt_pkls = out_dir / 'thresh_ppt_%s.pkl'
        for i in range(len(_calib_valid_labs)):
            _ = Path(str(thresh_ppt_pkls) % _calib_valid_labs[i])
            if not _.exists():
                thresh = ThreshPPT()
                thresh.set_ppt_arr(_ppt_arrs_list[i])
                thresh.set_cps_arr(_sel_cp_rules_list[i], n_cps)
                thresh.set_ge_vals_arr(o_1_ppt_thresh_arr)
                thresh.cmpt_ge_qual()

                with open(_, 'wb') as _pkl_hdl:
                    pickle.dump(thresh, _pkl_hdl)

            else:
                with open(_, 'rb') as _pkl_hdl:
                    thresh = pickle.load(_pkl_hdl)

            print('%s thresh ge:\n' % _calib_valid_labs[i],
                  thresh.cp_ge_qual_arr)
            print(thresh.ppt_ge_pis_arr, '\n\n')

    #==========================================================================
    # Thresh End
    #==========================================================================

    #==========================================================================
    # CPHistPlot Start
    #==========================================================================
    cp_hist_plot_flag = args_dict['cp_hist_plot_flag']
    if cp_hist_plot_flag:
        n_prev_hi_prob_steps = args_dict['n_prev_hi_prob_steps']
        n_post_hi_prob_steps = args_dict['n_post_hi_prob_steps']
        hi_prob = args_dict['hi_prob']
        ext_evts_n_cens_time = args_dict['ext_evts_n_cens_time']
        for i in range(len(_calib_valid_labs)):
            print('\n\nPlotting %s Freq. Hist.' % _calib_valid_labs[i])
            hist_plots = CPHistPlot()

            hist_plots.set_values_ser(hist_cat_ser.loc[_sel_cp_times_list[i]])
            hist_plots.set_sel_cps_ser(pd.Series(index=_sel_cp_times_list[i],
                                                 data=_sel_cp_rules_list[i]))
            hist_plots.set_hist_plot_prms(n_prev_hi_prob_steps,
                                          n_post_hi_prob_steps,
                                          miss_cp_val,
                                          _months_list[i],
                                          min_prob=None,
                                          max_prob=hi_prob,
                                          n_cens_time=ext_evts_n_cens_time,
                                          freq='D')

            hist_plots.cmpt_cp_hists()

            hist_plots.plot_cp_hists('%s fig' % _calib_valid_labs[i],
                                     _calib_valid_labs[i],
                                     out_dir / 'prev_post_histograms')

    #==========================================================================
    # CPHistPLot End
    #==========================================================================

    #==========================================================================
    # Plot CPs Start
    #==========================================================================
    plot_cps_flag = args_dict['plot_cps_flag']
    if plot_cps_flag:
        anom_epsg = args_dict['anom_epsg']
        out_cp_epsg = args_dict['out_cp_epsg']
        out_bck_shp_epsg = args_dict['out_bck_shp_epsg']
        backgrnd_shp_file = args_dict['backgrnd_shp_file']
        cont_levels = args_dict['cont_levels']
        n_1d_krige_pts = args_dict['n_1d_krige_pts']

        for i in range(len(_calib_valid_labs)):
            print('\n\n', _calib_valid_labs[i])
            plot_cps = PlotCPs()
            plot_cps.set_epsgs(anom_epsg, out_cp_epsg, out_bck_shp_epsg)

            plot_cps.set_bck_shp(backgrnd_shp_file)
            plot_cps.set_coords_arr(anomaly.x_coords, anomaly.y_coords)
            plot_cps.set_sel_cps_arr(_sel_cp_rules_list[i])
            plot_cps.set_cp_rules_arr(cp_rules)
            plot_cps.set_anoms_arr(_plot_anoms_list[i])
            plot_cps.set_other_prms(fuzz_nos_arr,
                                    n_cps,
                                    anom_type=anom_type,
                                    in_coords_type='geo')

            plot_cps.krige(n_1d_krige_pts)

            plot_cps.plot_kriged_cps(cont_levels,
                                     out_dir / ('cp_plots_%s' %
                                                _calib_valid_labs[i]),
                                     fig_size=((15, 10)),
                                     n_cpus=n_cpus)

#             break
    #==========================================================================
    # Plot CPs End
    #==========================================================================

    print('INFO: Classification ended at:',
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log_link.stop()
    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
