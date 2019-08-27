'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

from wcpc.plot.var2d import PlotNC

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

    in_net_cdf_file = main_dir / r'ncep_1980_2009_level_500_europe.nc'
    out_dir = main_dir / r'test_ncep_plots_europe_500'

    stt_time = '1980-01-01'
    end_time = '1980-12-31'
    time_fmt = '%Y-%m-%d'

    os.chdir(main_dir)

    plot_nc = PlotNC()

    plot_nc.set_vars(in_net_cdf_file,
                     out_dir,
                     'hgt',
                     'lon',
                     'lat',
                     'time',
                     stt_time,
                     end_time,
                     time_fmt)

    plot_nc.plot()

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
