'''
Created on Dec 29, 2017

@author: Faizan
'''
import os

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

from .core.anomaly import Anomaly
from .core.assigncp import CPAssignA
from .core.classify import CPClassiA

from .qual.thresh import ThreshPPT
from .qual.wettness import WettnessIndex
from .qual.contingency import ContingencyTablePlot
from .qual.objvals import ObjVals
from .qual.randcps import RandCPsGen, RandCPsPerfComp

from .plot.cpfreq import CPHistPlot
from .plot.cps import PlotCPs, plot_iter_cp_pcntgs
from .plot.var2d import PlotNC
from .plot.fuzznos import plot_tri_fuzz_no

from .misc.snipnc import SnipNC

from .alg_dtypes import DT_D_NP
