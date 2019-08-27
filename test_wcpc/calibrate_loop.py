'''
Created on Jan 6, 2018

@author: Faizan
'''
import re
import os
import timeit
import time
from pathlib import Path
from copy import deepcopy
import itertools

import numpy as np
import pandas as pd

from wcpc import (DT_D_NP)
from calibrate import run_wcpc

np.set_printoptions(precision=3,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.3f}'.format})

pd.options.display.precision = 3
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 250

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

if __name__ == '__main__':

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'Q:\CP_Classification_Results\mulde')

    in_net_cdf_file = main_dir / r'NCAR_ds010.0_19800101_20091231_dailydata_europe.nc'

    in_ppt_df_pkl = main_dir / r'ppt_19500101_20151231_point.pkl'

    in_wett_nebs_pkl = main_dir / r''  # obj 4, 6?

    in_nebs_stns_pkl = main_dir / r''

    in_cats_ppt_df_pkl = main_dir / r'ppt_19500101_20151231_areal.pkl'
    in_cats_dis_df_pkl = main_dir / r'dis_19500101_20151231.pkl'

    in_lorenz_df_pkl = main_dir / r''

    # fix the EPSG as well
    backgrnd_shp_file = main_dir / r'world_map_epsg_3035.shp'

    cats_areas_df_path = main_dir / r''

    main_out_dir = main_dir / 'seasonal_areal_ppt_cps'

    suff = ''
    x_coords_lab = 'lon'
    y_coords_lab = 'lat'
    time_lab = 'time'
    anom_var_lab = 'slp'
    anom_type = 'b'
    nan_anom_val = 0.5

    # number of days in between should be even for anom_type = f
    calib_period_list = [['1980-01-01', '1989-12-31']]
    valid_period_list = [['1990-01-01', '2009-12-30']]
    n_cps_list = [4, 8, 12]
    max_idxs_ct_list = [10]
    p_l_list = [1.5]
    temp_red_alpha_list = [0.99]

    max_n_iters_list = [2000000]
    max_k_iters_list = [10000]
    max_m_iters_list = [1000]

    n_reps = 1

    anneal_temp_ini = 0.05
    temp_adj_iters = 10000

    n_cpus = 'auto'

    no_cp_val = 99
    miss_cp_val = 98

    obj_1_wt = 1.0
    obj_2_wt = 1.0
    obj_3_wt = 0.1
    obj_4_wt = 1.0
    obj_5_wt = 0.1
    obj_6_wt = 1.0
    obj_7_wt = 1.0
    obj_8_wt = 0.1

    summer_months = np.arange(4, 10, 1)
    winter_months = np.array([1, 2, 3, 10, 11, 12])
    all_seasons_months = np.arange(1, 13)

    # have to use all months for anom_type = f
    calib_months, calib_season_lab = summer_months, 'summ'
#     calib_months, calib_season_lab = winter_months, 'wint'
#     calib_months, calib_season_lab = all_seasons_months, 'all'

    valid_months, valid_season_lab = summer_months, 'summ'
#     valid_months, valid_season_lab = winter_months, 'wint'
#     valid_months, valid_season_lab = all_seasons_months, 'all'

    obj_6_wett_thresh = 0.0
    lo_freq_pen_wt = 10
    min_freqs_list = [0.0 / _ for _ in n_cps_list]
    eig_cum_sum_ratio = 0.95
#     take_eig_rest_sq_sum_flag = True
    take_eig_rest_sq_sum_flag = False
    no_steep_anom_flag = True
    no_steep_anom_flag = False
    dis_cat_ext_evts = '560021'
    ext_evts_n_cens_time = 10
    sep = ';'

    ft_ll_steps_thresh = 2
    ft_ul_steps_thresh = None

    obj_1_flag = True
    obj_2_flag = True
    obj_3_flag = True
    obj_4_flag = True
    obj_5_flag = True
    obj_6_flag = True
    obj_7_flag = True
    obj_8_flag = True

    obj_1_flag = False
#     obj_2_flag = False
    obj_3_flag = False
    obj_4_flag = False
    obj_5_flag = False
    obj_6_flag = False
    obj_7_flag = False
    obj_8_flag = False

    wett_label = 'test'

    time_fmt = '%Y-%m-%d'

    o_1_ppt_thresh_arr = np.array([1e-5, 1., 3., 5.],
                                  dtype=DT_D_NP)
#     o_2_ppt_thresh_arr = np.array([0., 1., 3., 5.],
#                                   dtype=DT_D_NP)

    o_2_ppt_thresh_arr = np.array([1e-5, 1., 3, 5.],
                                  dtype=DT_D_NP)

    o_4_wett_thresh_arr = np.linspace(0.0, 0.99, 10,
                                      endpoint=True,
                                      dtype=DT_D_NP)

#     max_m_iters = 15000
#     max_n_iters = max_m_iters * 1000

#     max_k_iters = min(1000000, max_m_iters * 5)

    fuzz_nos_arr = np.array([[-0.1, 0.0, 0.4],
                             [-0.2, 0.2, 0.5],
                             [0.5, 0.8, 1.2],
                             [0.6, 1.0, 1.1]], dtype=DT_D_NP)

#     fuzz_nos_arr = np.array([[-1e6, 0.2, 0.6],
#                              [0.4, 0.8, 1e6]], dtype=DT_D_NP)

#     fuzz_nos_arr = np.array([[-0.1, 0.0, 0.2],
#                              [-0.2, 0.2, 0.5],
#                              [0.2, 0.5, 0.8],
#                              [0.5, 0.8, 1.2],
#                              [0.8, 1.0, 1.1]], dtype=float)

    n_prev_hi_prob_steps = 3
    n_post_hi_prob_steps = 2
    hi_prob = 0.99
    cont_levels = np.linspace(0.0, 1.0, 41)
    n_1d_krige_pts = 50

    n_rand_cp_gens = 100
    n_sim_cp_gens = 100
    n_nrst_cps = 2

    anom_epsg = 4236
    out_cp_epsg = 3035
    out_bck_shp_epsg = 3035

    anomaly_pkl_flag = True
    cp_classi_pkl_flag = True
    cp_assign_pkl_flag = True
    thresh_ppt_pkl_flag = True
    wettness_idx_pkl_flag = True
    cp_hist_plot_flag = True
    plot_cps_flag = True
    sort_cps_flag = True
    plot_cp_freq_iter_flag = True
    compare_rand_flag = True
    op_mp_memb_flag = True
    op_mp_obj_ftn_flag = True
    plot_anomaly_flag = True
    verif_plots_flag = True

#     anomaly_pkl_flag = False
#     cp_classi_pkl_flag = False
#     cp_assign_pkl_flag = False
#     thresh_ppt_pkl_flag = False
#     wettness_idx_pkl_flag = False
#     cp_hist_plot_flag = False
#     plot_cps_flag = False
#     sort_cps_flag = False
#     plot_cp_freq_iter_flag = False
#     compare_rand_flag = False
#     op_mp_memb_flag = False
#     op_mp_obj_ftn_flag = False
    plot_anomaly_flag = False
#     verif_plots_flag = False

    os.chdir(main_dir)

    if not main_out_dir.exists():
        main_out_dir.mkdir()

    assert len(calib_period_list) == len(valid_period_list)
    assert (len(max_m_iters_list) ==
            len(max_n_iters_list) ==
            len(max_k_iters_list))

    all_months = np.unique(np.concatenate((calib_months, valid_months)))

    args_dict = {}

    args_dict['in_net_cdf_file'] = in_net_cdf_file
    args_dict['in_ppt_df_pkl'] = in_ppt_df_pkl
    args_dict['in_wett_nebs_pkl'] = in_wett_nebs_pkl
    args_dict['in_cats_ppt_df_pkl'] = in_cats_ppt_df_pkl
    args_dict['in_cats_dis_df_pkl'] = in_cats_dis_df_pkl
    args_dict['in_lorenz_df_pkl'] = in_lorenz_df_pkl
    args_dict['in_nebs_stns_pkl'] = in_nebs_stns_pkl

    args_dict['cats_areas_df_path'] = cats_areas_df_path
    args_dict['backgrnd_shp_file'] = backgrnd_shp_file

    args_dict['x_coords_lab'] = x_coords_lab
    args_dict['y_coords_lab'] = y_coords_lab
    args_dict['time_lab'] = time_lab
    args_dict['anom_var_lab'] = anom_var_lab
    args_dict['anom_type'] = anom_type
    args_dict['nan_anom_val'] = nan_anom_val

    args_dict['time_fmt'] = time_fmt
    args_dict['wett_label'] = wett_label

    args_dict['obj_1_flag'] = obj_1_flag
    args_dict['obj_2_flag'] = obj_2_flag
    args_dict['obj_3_flag'] = obj_3_flag
    args_dict['obj_4_flag'] = obj_4_flag
    args_dict['obj_5_flag'] = obj_5_flag
    args_dict['obj_6_flag'] = obj_6_flag
    args_dict['obj_7_flag'] = obj_7_flag
    args_dict['obj_8_flag'] = obj_8_flag

    args_dict['n_cpus'] = n_cpus
    args_dict['no_cp_val'] = no_cp_val
    args_dict['miss_cp_val'] = miss_cp_val
    args_dict['n_rand_cp_gens'] = n_rand_cp_gens
    args_dict['n_sim_cp_gens'] = n_sim_cp_gens
    args_dict['n_nrst_cps'] = n_nrst_cps
    args_dict['sep'] = sep

    args_dict['ft_ll_steps_thresh'] = ft_ll_steps_thresh
    args_dict['ft_ul_steps_thresh'] = ft_ul_steps_thresh

    args_dict['obj_1_wt'] = obj_1_wt
    args_dict['obj_2_wt'] = obj_2_wt
    args_dict['obj_3_wt'] = obj_3_wt
    args_dict['obj_4_wt'] = obj_4_wt
    args_dict['obj_5_wt'] = obj_5_wt
    args_dict['obj_6_wt'] = obj_6_wt
    args_dict['obj_7_wt'] = obj_7_wt
    args_dict['obj_8_wt'] = obj_8_wt

    args_dict['calib_months'] = calib_months
    args_dict['valid_months'] = valid_months
    args_dict['all_months'] = all_months

    args_dict['obj_6_wett_thresh'] = obj_6_wett_thresh
    args_dict['lo_freq_pen_wt'] = lo_freq_pen_wt

    args_dict['eig_cum_sum_ratio'] = eig_cum_sum_ratio
    args_dict['take_eig_rest_sq_sum_flag'] = take_eig_rest_sq_sum_flag
    args_dict['no_steep_anom_flag'] = no_steep_anom_flag

    args_dict['dis_cat_ext_evts'] = dis_cat_ext_evts
    args_dict['ext_evts_n_cens_time'] = ext_evts_n_cens_time

    args_dict['o_1_ppt_thresh_arr'] = o_1_ppt_thresh_arr
    args_dict['o_2_ppt_thresh_arr'] = o_2_ppt_thresh_arr
    args_dict['o_4_wett_thresh_arr'] = o_4_wett_thresh_arr

    args_dict['anneal_temp_ini'] = anneal_temp_ini
    args_dict['temp_adj_iters'] = temp_adj_iters

    args_dict['fuzz_nos_arr'] = fuzz_nos_arr

    args_dict['n_prev_hi_prob_steps'] = n_prev_hi_prob_steps
    args_dict['n_post_hi_prob_steps'] = n_post_hi_prob_steps
    args_dict['hi_prob'] = hi_prob
    args_dict['cont_levels'] = cont_levels
    args_dict['anom_epsg'] = anom_epsg
    args_dict['out_cp_epsg'] = out_cp_epsg
    args_dict['out_bck_shp_epsg'] = out_bck_shp_epsg
    args_dict['n_1d_krige_pts'] = n_1d_krige_pts

    args_dict['anomaly_pkl_flag'] = anomaly_pkl_flag
    args_dict['cp_classi_pkl_flag'] = cp_classi_pkl_flag
    args_dict['cp_assign_pkl_flag'] = cp_assign_pkl_flag
    args_dict['thresh_ppt_pkl_flag'] = thresh_ppt_pkl_flag
    args_dict['wettness_idx_pkl_flag'] = wettness_idx_pkl_flag
    args_dict['cp_hist_plot_flag'] = cp_hist_plot_flag
    args_dict['plot_cps_flag'] = plot_cps_flag
    args_dict['sort_cps_flag'] = sort_cps_flag
    args_dict['plot_cp_freq_iter_flag'] = plot_cp_freq_iter_flag
    args_dict['compare_rand_flag'] = compare_rand_flag
    args_dict['plot_anomaly_flag'] = plot_anomaly_flag

    args_dict['op_mp_memb_flag'] = op_mp_memb_flag
    args_dict['op_mp_obj_ftn_flag'] = op_mp_obj_ftn_flag

    args_dict['verif_plots_flag'] = verif_plots_flag

    n_reps_list = list(range(n_reps))

    _iter_lists = [calib_period_list,
                   n_cps_list,
                   max_idxs_ct_list,
                   p_l_list,
                   temp_red_alpha_list,
                   n_reps_list]

    args_dict_list = []
    _sorted_list = []

    prods_iter = itertools.product(*[range(len(calib_period_list)),
                                     n_cps_list,
                                     max_idxs_ct_list,
                                     p_l_list,
                                     temp_red_alpha_list,
                                     range(len(max_m_iters_list)),
                                     n_reps_list])

    for prod in prods_iter:
        _dict = deepcopy(args_dict)
        _dict['calib_period_srt'] = calib_period_list[prod[0]][0]
        _dict['calib_period_end'] = calib_period_list[prod[0]][1]
        _dict['valid_period_srt'] = valid_period_list[prod[0]][0]
        _dict['valid_period_end'] = valid_period_list[prod[0]][1]
        _dict['n_cps'] = prod[1]
        _dict['max_idxs_ct'] = prod[2]
        _dict['p_l'] = prod[3]
        _dict['temp_red_alpha'] = prod[4]

        _dict['max_m_iters'] = max_m_iters_list[prod[5]]
        _dict['max_n_iters'] = max_n_iters_list[prod[5]]
        _dict['max_k_iters'] = max_k_iters_list[prod[5]]
        _dict['min_freq'] = min_freqs_list[n_cps_list.index(_dict['n_cps'])]

        _out_dir = ''

        _out_dir += 'ob'
        if obj_1_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_2_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_3_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_4_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_5_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_6_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_7_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        if obj_8_flag:
            _out_dir += '1'
        else:
            _out_dir += '0'

        _out_dir += '_c%s' % re.sub('\W', '', _dict['calib_period_srt'])
        _out_dir += '_%s' % re.sub('\W', '', _dict['calib_period_end'])

        _out_dir += '_v%s' % re.sub('\W', '', _dict['valid_period_srt'])
        _out_dir += '_%s' % re.sub('\W', '', _dict['valid_period_end'])

        _out_dir += '_cps%0.2d' % _dict['n_cps']
        _out_dir += '_idxsct%0.2d' % _dict['max_idxs_ct']

        _out_dir += '_pl%0.3d' % int(10 * _dict['p_l'])

        _out_dir += '_trda%0.4d' % int(1000 * _dict['temp_red_alpha'])

        _out_dir += '_mxn%0.8d' % _dict['max_n_iters']
        _out_dir += '_mxm%0.8d' % _dict['max_m_iters']

        _out_dir += '_at%s' % anom_type
        if anom_type == 'd':
            _out_dir += '_ecs%0.4d' % int(10000 * eig_cum_sum_ratio)

        _out_dir += '_%s_%s' % (calib_season_lab, valid_season_lab)

        _out_dir += '_rp%0.2d' % prod[-1]

        if suff:
            _out_dir += '_%s' % suff

        _sorted_list.append(_out_dir)

        print(_out_dir)

        _out_dir = main_out_dir / _out_dir

        _dict['out_dir'] = _out_dir

        args_dict_list.append(_dict)

    print('\n\n', 30 * '#', sep='')
    print('%d optimization(s) to run...' % len(args_dict_list))
    print(30 * '#', '\n\n')

    for i in range(len(args_dict_list)):
        run_wcpc(args_dict_list[i])

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
