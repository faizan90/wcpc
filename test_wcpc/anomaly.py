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

    out_dir = main_dir / r'moving_window_volumes_test_01/ncep_pp'

#     in_net_cdf_file = main_dir / r'NCAR_ds010.0_19610101_20151231_dailydata_europe.nc'
#     out_anomaly_pkl = out_dir / 'NCAR_ds010.0_19610101_20151231_dailydata_europe_ate.pkl'
#     nc_var_lab = 'slp'

    in_net_cdf_file = main_dir / r'ncep_1948_2017_level_500_europe.nc'
    out_anomaly_pkl = out_dir / 'ncep_500_ate_1948_2015.pkl'
    nc_var_lab = 'hgt'

    strt_time = '1961-01-01'
    end_time = '2015-12-31'

    time_fmt = '%Y-%m-%d'
    sub_daily_flag = False
    normalize = False

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

    anomaly.calc_anomaly_type_e(
        strt_time,
        end_time,
        strt_time,
        end_time,
        season_months=np.arange(1, 13),
        time_fmt=time_fmt,
        eig_cum_sum_ratio=1.0,
        eig_sum_flag=False,
        normalize=normalize)

#     with open(out_anomaly_pkl, 'wb') as _pkl_hdl:
#         pickle.dump(anomaly, _pkl_hdl)

    # for app-dis
    out_dict = {}

    anomaly_var_df = pd.DataFrame(
        data=anomaly.vals_tot_anom,
        index=anomaly.times)

    pcs_arr = anomaly.vals_anom
    eig_val_cum_sums = anomaly.eig_val_cum_sum_arr

    out_dict['anomaly_var_df'] = anomaly_var_df
    out_dict['pcs_arr'] = pcs_arr
    out_dict['eig_val_cum_sums'] = eig_val_cum_sums

    print(f'eig_val_cum_sums: {eig_val_cum_sums}')

    with open(out_anomaly_pkl, 'wb') as _pkl_hdl:
        pickle.dump(out_dict, _pkl_hdl)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
