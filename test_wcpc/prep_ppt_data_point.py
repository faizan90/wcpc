'''
Created on Nov 13, 2017

@author: Faizan-Uni

path: P:/Synchronize/IWS/2016_DFG_SPATE/scripts_p3/07_sel_ppt_temp_stns.py

'''

import os
import timeit
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


def main():

    main_dir = Path(r'Q:\CP_Classification_Results\mulde')
    os.chdir(main_dir)

    in_ppt_file = Path(r'ppt_infilled_1950_2015_daily_point.csv')

    in_ppt_coords_file = Path(
        r'ppt_infilled_1950_2015_daily_point_coords.csv')

    out_ppt_file = Path('ppt_19500101_20151231_point.pkl')

    out_ppt_coords_file = Path('ppt_19500101_20151231_point_coords.pkl')

    stt_time = '1950-01-01'  # same as input
    end_time = '2015-12-31'
    time_fmt = '%Y-%m-%d'
    sep = ';'

    in_ppt_df = pd.read_csv(in_ppt_file, sep=sep, index_col=0)
    in_ppt_df.index = pd.to_datetime(in_ppt_df.index, format=time_fmt)

    in_ppt_coords_df = pd.read_csv(in_ppt_coords_file, sep=';', index_col=0)

    stt_time, end_time = pd.to_datetime([stt_time, end_time], format=time_fmt)

    print('in_ppt_df shape - original:', in_ppt_df.shape)

    in_ppt_df = in_ppt_df.loc[stt_time:end_time]

    print('in_ppt_df shape - time slice:', in_ppt_df.shape)

    in_ppt_df.dropna(how='any', axis=1, inplace=True)

    print('in_ppt_df shape - dropna:', in_ppt_df.shape)

    _ = ~in_ppt_coords_df.index.duplicated(keep='last')
    in_ppt_coords_df = in_ppt_coords_df[_]

    in_ppt_coords_df = in_ppt_coords_df.loc[in_ppt_df.columns]

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
