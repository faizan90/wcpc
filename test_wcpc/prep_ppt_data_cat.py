'''
Created on Nov 13, 2017

@author: Faizan-Uni

path: P:/Synchronize/IWS/2016_DFG_SPATE/scripts_p3/07_sel_ppt_temp_stns.py

'''

import os
import timeit
import time
from pathlib import Path

import h5py
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


def main():

    main_dir = Path(r'Q:\CP_Classification_Results\mulde')
    os.chdir(main_dir)

    in_ppt_file = Path(r'ppt_1950_to_2015_daily_1km_mulde.h5')

    extract_h5_ds_path = (
        'mulde_precipitation_kriging_1950-01-01_to_2015-12-31_1km_all/EDK')

    out_ppt_file = Path('ppt_19500101_20151231_areal.pkl')

    out_ppt_coords_file = Path('ppt_19500101_20151231_areal_coords.pkl')

    stt_time = '1950-01-01'  # same as input
    end_time = '2015-12-31'
    time_fmt = '%Y-%m-%d'

    with h5py.File(in_ppt_file, mode='r', driver=None) as h5_hdl:
        extr_ds = h5_hdl[extract_h5_ds_path]

        columns = list(extr_ds.keys())

        in_ppt_df = pd.DataFrame(
            columns=columns,
            dtype=float,
            index=pd.to_datetime(
                h5_hdl['time/time_strs'][...], format='%Y%m%dT%H%M%S'))

        in_ppt_coords_df = pd.DataFrame(
            index=columns, columns=['X', 'Y'], dtype=float)

        for column in columns:
            in_ppt_df.loc[:, column] = extr_ds[column][...].mean(axis=1)

            in_ppt_coords_df.loc[column, :] = (
                h5_hdl[f'x_cen_crds/{column}'][...].mean(),
                h5_hdl[f'y_cen_crds/{column}'][...].mean())

    stt_time, end_time = pd.to_datetime([stt_time, end_time], format=time_fmt)

    print('in_ppt_df shape - original:', in_ppt_df.shape)

    in_ppt_df = in_ppt_df.loc[stt_time:end_time]

    print('in_ppt_df shape - time slice:', in_ppt_df.shape)

    in_ppt_df.dropna(how='any', axis=1, inplace=True)

    print('in_ppt_df shape - dropna:', in_ppt_df.shape)

    in_ppt_df.to_pickle(out_ppt_file)
    in_ppt_coords_df.to_pickle(out_ppt_coords_file)
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
