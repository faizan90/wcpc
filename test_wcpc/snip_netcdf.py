'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wcpc.misc.snipnc import SnipNC


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data')

    in_net_cdf_file = main_dir / r'NCAR_ds010.0_18990101_20171030_dailydata.nc'
    out_net_cdf_file = main_dir / r'NCAR_ds010.0_19000101_20151231_dailydata_europe.nc'

    lon_min = 345  # the westward it has upto 355 degrees max
    lat_min = 35
    lon_max = 40  # eastward
    lat_max = 65

    stt_time = '1900-01-01'
    end_time = '2015-12-31'
    time_fmt = '%Y-%m-%d'

    # list of integers, multiple because NCAR has variable sample times
    hours_list = [12, 13]  # should be one per day for CP classification

#
#     in_xr_ds = xr.open_dataset(in_net_cdf_file)
#
#     stt_time, end_time = pd.to_datetime([stt_time, end_time], format=time_fmt)
#
#     idx = pd.DatetimeIndex(in_xr_ds.time.values)
#     lats = in_xr_ds.lat.values
#     lons = in_xr_ds.lon.values
#
#     idxs_bool = np.zeros(idx.shape, dtype=bool)
#     hours_bool = idxs_bool.copy()
#
#     idxs_bool = ((idx >= stt_time) & (idx <= end_time))
#
#     for hour in hours_list:
#         hours_bool = hours_bool | (idx.hour == hour)
#
#     idxs_bool = (hours_bool & idxs_bool)
#
#     lats_bool = (lats >= lat_min) & (lats <= lat_max)
#
#     if lon_min > lon_max:
#         _1 = lons >= lon_min
#         _2 = lons <= lon_max
#         lons_vals_re = np.concatenate((lons[_1], lons[_2]))
#
#     else:
#         lons_vals_re = (lons >= lon_min) & (lons <= lon_max)
#
#     out_xr_ds = in_xr_ds.loc[dict(time=idxs_bool,
#                                   lat=lats_bool,
#                                   lon=lons_vals_re)]
#
#     print('out_xr_ds sizes:', out_xr_ds.sizes)
#     out_xr_ds.to_netcdf(out_net_cdf_file)

    snip_nc = SnipNC()
    snip_nc.set_paths(in_net_cdf_file, out_net_cdf_file)
#     snip_nc.set_coords(330, 45, 35, 65, 'geo', 'lon', 'lat')
    snip_nc.set_coords(lon_min, lon_max, lat_min, lat_max, 'geo', 'lon', 'lat')
    snip_nc.set_times(stt_time, end_time, time_fmt, hours_list, 'H', 'time')
#     snip_nc.set_times('1948-01-01', '2017-12-31', '%Y-%m-%d', [0, 6, 12, 18], 'H', 'time')
    snip_nc.snip()
    snip_nc.save_snip()
    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
