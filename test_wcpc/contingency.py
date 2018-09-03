'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from pathlib import Path

import numpy as np

from wcpc import (ContingencyTablePlot)

np.set_printoptions(precision=8,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.8f}'.format})


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

    cp_assign_pkls_1 = main_dir / 'ncep_anom_b_d_tests\ob01000000_c19800101_19941231_v19950101_20091231_cps10_idxsct20_pl020_trda0990_mxn01000000_mxm00002000_atb_wint_summ_rp00_test_seasons/cp_assign_all.pkl'
    cp_assign_pkls_2 = main_dir / 'ncep_anom_b_d_tests\ob01000000_c19800101_19941231_v19950101_20091231_cps10_idxsct20_pl020_trda0990_mxn01000000_mxm00002000_atb_summ_wint_rp00_test_seasons/cp_assign_all.pkl'

    lab_1 = 'wint_summ_all'
    lab_2 = 'summ_wint_all'

    os.chdir(main_dir)

    with open(cp_assign_pkls_1, 'rb') as _pkl_hdl:
        assign_cps_1 = pickle.load(_pkl_hdl)

    with open(cp_assign_pkls_2, 'rb') as _pkl_hdl:
        assign_cps_2 = pickle.load(_pkl_hdl)

    cont_plot = ContingencyTablePlot()

    cont_plot.set_sel_cps_arr(assign_cps_1.sel_cps_arr,
                              assign_cps_2.sel_cps_arr,
                              assign_cps_1.n_cps,
                              assign_cps_2.n_cps,
                              assign_cps_1.no_cp_val,
                              assign_cps_2.no_cp_val,
                              assign_cps_1.miss_cp_val,
                              assign_cps_2.miss_cp_val)

    cont_plot.cmpt_table()
    print(cont_plot.cont_table_str_1_arr)
    print('\n')
    print(cont_plot.cont_table_str_2_arr)

    print('\n')
    cont_plot.plot_cont_table(main_dir,
                              lab_1=lab_1,
                              lab_2=lab_2)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
