'''
@author: Faizan-Uni-Stuttgart

Apr 19, 2022

3:10:26 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

from wcpc import CPMarkovGen, plot_cp_markov_arrays

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Colleagues_Students\Bardossy\cp_classi\classi_ab_ncar_de_sp7_01\ob10000000_c20050101_20191231_v20050101_20191231_cps08')

    os.chdir(main_dir)

    in_cp_ts_file = Path(r'cp_ts_de_8_cps.csv')

    sep = ';'

    cp_col = 'cp'

    sim_ts_len = None

    n_sims = 1

    out_dir = Path('markov_cp_time_sers')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    cp_ts_ser = pd.read_csv(
        in_cp_ts_file, sep=sep, index_col=0)[cp_col]

    cp_ts = cp_ts_ser.values.astype(np.int32)

    plot_cp_markov_arrays(cp_ts, out_dir, 'ref')

    if sim_ts_len is None:
        sim_ts_len = cp_ts.size
        cp_ts_ser_sim = cp_ts_ser.copy()

        cp_ts_ser.to_csv(out_dir / f'cp_ts__ref.csv', sep=sep)

    else:
        cp_ts_ser_sim = None

        np.savetxt(out_dir / f'cp_ts__ref.csv', cp_ts, delimiter=sep)

    mk_cls = CPMarkovGen()

    mk_cls.set_ref_cp_ts(cp_ts)

    mk_cls.prepare()

    for i in range(n_sims):
        mk_cls.simulate(sim_ts_len)

        cp_ts_sim = mk_cls.get_simulated_series()

        if cp_ts_ser_sim is not None:
            cp_ts_ser_sim[:] = cp_ts_sim

            cp_ts_ser_sim.to_csv(out_dir / f'cp_ts__sim_{i:02d}.csv', sep=sep)

        else:
            np.savetxt(
                out_dir / f'cp_ts__sim_{i:02d}.csv', cp_ts_sim, delimiter=sep)

        plot_cp_markov_arrays(cp_ts_sim, out_dir, f'sim_{i:02d}')
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
