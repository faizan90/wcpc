'''
@author: Faizan-Uni-Stuttgart

Apr 19, 2022

3:20:15 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from ..core.markov import CPMarkovGen

DEBUG_FLAG = False

mk_props_ftn = CPMarkovGen.get_cp_markov_arrays


def plot_cp_markov_arrays(cp_ts, out_dir, sim_label):

    (unq_cps,
     _,
     cond_freqs,
     cond_rel_freqs,
     cond_cumm_rel_freqs) = mk_props_ftn(cp_ts)

    cond_cumm_rel_freqs = cond_cumm_rel_freqs[:, 1:]

    n_cps = unq_cps.size

    tick_font_size = 6

    ax_mat = plt.subplots(1, 1, figsize=(1 * n_cps, 1 * n_cps))[1]

    if True:
        # Absolute frequency.
        ax_mat.matshow(
            cond_freqs,
            cmap=plt.get_cmap('Blues'),
            vmin=0.0,
            vmax=cond_freqs.max() * 1.1,
            origin='upper')

        for s in zip(
            np.repeat(list(range(n_cps)), n_cps),
            np.tile(list(range(n_cps)), n_cps)):

            ax_mat.text(
                s[0],
                s[1],
                ('%d' % cond_freqs.T[s[0], s[1]]).rstrip('0'),
                va='center',
                ha='center',
                fontsize=tick_font_size,
                rotation=45)

        ax_mat.set_xticks(list(range(0, n_cps)))
        ax_mat.set_xticklabels(unq_cps)

        ax_mat.set_yticks(list(range(0, n_cps)))
        ax_mat.set_yticklabels(unq_cps)

        ax_mat.spines['left'].set_position(('outward', 10))
        ax_mat.spines['right'].set_position(('outward', 10))
        ax_mat.spines['top'].set_position(('outward', 10))
        ax_mat.spines['bottom'].set_position(('outward', 10))

        ax_mat.set_xlabel('Absolute CP frequency (next)', size=tick_font_size)
        ax_mat.set_ylabel('Absolute CP frequency (present)', size=tick_font_size)

        ax_mat.tick_params(
            labelleft=True,
            labelbottom=True,
            labeltop=True,
            labelright=True)

        plt.setp(ax_mat.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(ax_mat.get_yticklabels(), size=tick_font_size)

        plt.savefig(
            str(out_dir / f'abs_freqs__{sim_label}.png'),
            dpi=150,
            bbox_inches='tight')

        plt.cla()

    if True:
        # Relative frequency.
        ax_mat.matshow(
            cond_rel_freqs,
            cmap=plt.get_cmap('Blues'),
            vmin=0.0,
            vmax=cond_rel_freqs.max() * 1.1,
            origin='upper')

        for s in zip(
            np.repeat(list(range(n_cps)), n_cps),
            np.tile(list(range(n_cps)), n_cps)):

            ax_mat.text(
                s[0],
                s[1],
                ('%0.2f' % cond_rel_freqs.T[s[0], s[1]]).rstrip('0'),
                va='center',
                ha='center',
                fontsize=tick_font_size,
                rotation=45)

        ax_mat.set_xticks(list(range(0, n_cps)))
        ax_mat.set_xticklabels(unq_cps)

        ax_mat.set_yticks(list(range(0, n_cps)))
        ax_mat.set_yticklabels(unq_cps)

        ax_mat.spines['left'].set_position(('outward', 10))
        ax_mat.spines['right'].set_position(('outward', 10))
        ax_mat.spines['top'].set_position(('outward', 10))
        ax_mat.spines['bottom'].set_position(('outward', 10))

        ax_mat.set_xlabel('Relative CP frequency (next)', size=tick_font_size)
        ax_mat.set_ylabel('Relative CP frequency (present)', size=tick_font_size)

        ax_mat.tick_params(
            labelleft=True,
            labelbottom=True,
            labeltop=True,
            labelright=True)

        plt.setp(ax_mat.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(ax_mat.get_yticklabels(), size=tick_font_size)

        plt.savefig(
            str(out_dir / f'rel_freqs__{sim_label}.png'),
            dpi=150,
            bbox_inches='tight')

        plt.cla()

    if True:
        # Cummulative relative frequency.
        ax_mat.matshow(
            cond_cumm_rel_freqs,
            cmap=plt.get_cmap('Blues'),
            vmin=0.0,
            vmax=cond_cumm_rel_freqs.max() * 1.1,
            origin='upper')

        for s in zip(
            np.repeat(list(range(n_cps)), n_cps),
            np.tile(list(range(n_cps)), n_cps)):

            ax_mat.text(
                s[0],
                s[1],
                ('%0.2f' % cond_cumm_rel_freqs.T[s[0], s[1]]).rstrip('0'),
                va='center',
                ha='center',
                fontsize=tick_font_size,
                rotation=45)

        ax_mat.set_xticks(list(range(0, n_cps)))
        ax_mat.set_xticklabels(unq_cps)

        ax_mat.set_yticks(list(range(0, n_cps)))
        ax_mat.set_yticklabels(unq_cps)

        ax_mat.spines['left'].set_position(('outward', 10))
        ax_mat.spines['right'].set_position(('outward', 10))
        ax_mat.spines['top'].set_position(('outward', 10))
        ax_mat.spines['bottom'].set_position(('outward', 10))

        ax_mat.set_xlabel(
            'Cumm. relative CP frequency (next)', size=tick_font_size)

        ax_mat.set_ylabel(
            'Cumm. relative CP frequency (present)', size=tick_font_size)

        ax_mat.tick_params(
            labelleft=True,
            labelbottom=True,
            labeltop=True,
            labelright=True)

        plt.setp(ax_mat.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(ax_mat.get_yticklabels(), size=tick_font_size)

        plt.savefig(
            str(out_dir / f'cumm_rel_freqs__{sim_label}.png'),
            dpi=150,
            bbox_inches='tight')

        plt.cla()

    plt.close()
    return


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
