'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()


def plot_tri_fuzz_no(in_tri_fuzz_nos_arr, out_fig_path, fig_size=(15, 7)):
    assert isinstance(in_tri_fuzz_nos_arr, np.ndarray)

    assert in_tri_fuzz_nos_arr.ndim == 2
    assert in_tri_fuzz_nos_arr.shape[0]
    assert in_tri_fuzz_nos_arr.shape[1] == 3

    assert np.all(np.ediff1d(in_tri_fuzz_nos_arr[:, 1]) > 0)

    out_fig_path = Path(out_fig_path)
    assert out_fig_path.parents[0].exists()

    x_min = in_tri_fuzz_nos_arr[0, 1] - 0.5
    x_max = in_tri_fuzz_nos_arr[-1, 1] + 0.5

    membs_arr = np.array([0, 1, 0])

    n_fuzz_nos = in_tri_fuzz_nos_arr.shape[0]
    plt.figure(figsize=fig_size)
    for i in range(n_fuzz_nos):
        plt.fill_between(in_tri_fuzz_nos_arr[i, :],
                         0,
                         membs_arr,
                         alpha=0.3,
                         label=('Fuzz. no.: %d' % i))

    plt.xlim(x_min, x_max)
    plt.xlabel('Fuzzy value')
    plt.ylabel('Membership')
    plt.title('CP Classification - Fuzzy Numbers')
    plt.legend()
    plt.grid()
    plt.savefig(str(out_fig_path), bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program
    
    main_dir = Path(os.getcwd())

    fuzz_nos_arr = np.array([[0.0, 0.0, 0.2],
                             [-0.2, 0.2, 0.5],
                             [0.2, 0.5, 0.8],
                             [0.5, 0.8, 1.2],
                             [0.8, 1.0, 1.0]], dtype=float)

    os.chdir(main_dir)
    
    plot_tri_fuzz_no(fuzz_nos_arr, '', fig_size=(15, 7))

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
