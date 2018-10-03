'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

from wcpc.misc.snipnc import SnipNC

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

#     in_net_cdf_file = main_dir / r'NCAR_ds010.0_18990101_20171030_dailydata.nc'
#     out_net_cdf_file = main_dir / r'NCAR_ds010.0_19610101_20151231_dailydata_europe.nc'

    in_net_cdf_file = main_dir / r'ncep_1948_2017_level_500_6h.nc'
    out_net_cdf_file = main_dir / r'ncep_1948_2017_level_500_6h_europe.nc'

    os.chdir(main_dir)

    snip_nc = SnipNC()
    snip_nc.set_paths(in_net_cdf_file, out_net_cdf_file)
    snip_nc.set_coords(330, 45, 35, 65, 'geo', 'lon', 'lat')
#     snip_nc.set_times('1961-01-01', '2015-12-31', '%Y-%m-%d', [12, 13], 'H', 'time')
    snip_nc.set_times('1948-01-01', '2017-12-31', '%Y-%m-%d', [0, 6, 12, 18], 'H', 'time')
    snip_nc.snip()
    snip_nc.save_snip()

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
