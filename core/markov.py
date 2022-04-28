'''
@author: Faizan-Uni-Stuttgart

Apr 19, 2022

2:16:12 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np

DEBUG_FLAG = False


class CPMarkovGen:

    def __init__(self):

        self._cp_ts = None

        self._unq_cps = None
        self._unq_cp_idxs_dict = None

        self._cond_cp_freqs = None
        self._cond_cp_rel_freqs = None
        self._cond_cp_cumm_rel_freqs = None

        self._cp_ts_sim = None

        self._set_ref_flag = False
        self._set_prepare_flag = False
        self._set_sim_cmplt_flag = False
        return

    def set_ref_cp_ts(self, cp_ts):

        assert isinstance(cp_ts, np.ndarray), type(cp_ts)
        assert cp_ts.ndim == 1, cp_ts.ndim
        assert np.all(np.isfinite(cp_ts))
        assert cp_ts.dtype == np.int32, cp_ts.dtype
        assert cp_ts.size >= 2

        self._cp_ts = cp_ts

        self._set_ref_flag = True
        return

    def prepare(self):

        assert self._set_ref_flag, 'Call set_ref_cp_ts first!'

        (self._unq_cps,
         self._unq_cp_idxs_dict,
         self._cond_cp_freqs,
         self._cond_cp_rel_freqs,
         self._cond_cp_cumm_rel_freqs) = CPMarkovGen.get_cp_markov_arrays(
             self._cp_ts)

        self._set_prepare_flag = True
        return

    def simulate(self, n_time_steps):

        assert self._set_prepare_flag, 'Call prepare first!'

        cp_ts_sim = np.empty(n_time_steps, dtype=np.uint64)

        cp_pre = np.random.choice(self._unq_cps)
        cp_pre_idx = self._unq_cp_idxs_dict[cp_pre]
        for i in range(n_time_steps):

            rand_prob = np.random.rand()

            cp_curr_idx = np.searchsorted(
                self._cond_cp_cumm_rel_freqs[cp_pre_idx,:],
                rand_prob) - 1

            assert 0 <= cp_curr_idx < self._unq_cps.size

            cp_pre = self._unq_cps[cp_curr_idx]
            cp_pre_idx = self._unq_cp_idxs_dict[cp_pre]

            cp_ts_sim[i] = cp_pre

        self._cp_ts_sim = cp_ts_sim

        self._set_sim_cmplt_flag = True
        return

    def get_simulated_series(self):

        '''
        Can only call this function once per simulation.
        '''

        assert self._set_sim_cmplt_flag, 'Call simulate first!'

        self._set_sim_cmplt_flag = False

        return self._cp_ts_sim

    @staticmethod
    def get_cp_markov_arrays(cp_ts):

        unq_cps = np.unique(cp_ts)

        unq_cp_idxs_dict = {unq_cp:i for i, unq_cp in enumerate(unq_cps)}

        cond_freqs = np.zeros((unq_cps.size, unq_cps.size), dtype=np.uint64)

        for i, cp_i in enumerate(unq_cps):
            idxs_i = np.where(cp_ts == cp_i)[0]

            idxs_j = idxs_i + 1

            for idx_j in idxs_j:
                if idx_j == cp_ts.size:
                    continue

                cp_j = cp_ts[idx_j]

                j = unq_cp_idxs_dict[cp_j]

                cond_freqs[i, j] += 1

        cond_rel_freqs = cond_freqs / cond_freqs.sum(axis=1).reshape(-1, 1)

        assert np.all(np.isclose(cond_rel_freqs.sum(axis=1), 1.0)), (
            cond_rel_freqs.sum(axis=1))

        cond_cumm_rel_freqs = cond_rel_freqs.cumsum(axis=1)

        cond_cumm_rel_freqs = np.concatenate(
            (np.zeros((unq_cps.size, 1)), cond_cumm_rel_freqs), axis=1)

        assert np.all(np.isclose(cond_cumm_rel_freqs[:, -1], 1.0)), (
            cond_rel_freqs[:, -1])

        return (
            unq_cps,
            unq_cp_idxs_dict,
            cond_freqs,
            cond_rel_freqs,
            cond_cumm_rel_freqs)


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    #==========================================================================

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
