# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:53 2019

@author: AmP
"""

# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from matplotlib2tikz import save as tikz_save


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval_pathPlanner as ev
from Src import load
from Src import plot_fun_pathPlanner as pf
from Src import save as my_save

import utils as uti

# %%

modes = [
        'straight_1',
        'straight_2',
        'straight_3',
#        'curve_1',
#        'curve_2',
#        'curve_3',
        ]


colors = {
        'straight_1': 'blue',
        'straight_2': 'red',
        'straight_3': 'orange',
        'curve_1': 'blue',
        'curve_2': 'red',
        'curve_3': 'orange',
        }


settings = {
    'curve_1': {'nexps': [0, 1, 2, 3, 4],
                'startidx': 4},
    'curve_2': {'nexps': [0, 1, 2, 3, 4],
                'startidx': 2},
    'curve_3': {'nexps': [0, 1, 2, 3, 4],
                'startidx': 4},
    'straight_1': {'nexps': [0, 1, 2, 3, 4],
                   'startidx': 4},
    'straight_2': {'nexps': [0, 1],
                   'startidx': 0},
    'straight_3': {'nexps': [0, 1, 2, 3, 4],
                   'startidx': 0},  # 8
        }


version = 'v40'


# %%

n_steps = {}
MU_P = {}
SIG_P = {}

MU_DX = {}
SIG_DX = {}
MU_DX_SIM = {}
SIG_DX_SIM = {}
for idx in range(6):
    MU_P[idx] = {}
    SIG_P[idx] = {}
    MU_DX[idx] = {}
    SIG_DX[idx] = {}
    MU_DX_SIM[idx] = {}
    SIG_DX_SIM[idx] = {}

MU_EPS = {}
SIG_EPS = {}

MU_DEPS = {}
SIG_DEPS = {}

MU_DEPS_SIM = {}
SIG_DEPS_SIM = {}






#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


for mode in modes:
    n_steps[mode] = {}
    # %% ### Load Data

    dirpath = mode + '/'

    sets = load.get_csv_set(dirpath)
    db = ev.load_data_pathPlanner(dirpath, sets, version=version)

    print('find idxes....')
    POSE_IDX = uti.ieee_find_poses_idx(db, neighbors=10)
    n_steps[mode] = [len(idx)-1 for idx in POSE_IDX]

    # %% ### Track of feet:
#    print('plot track....')
#    pf.plot_track(db, POSE_IDX, 'L', mode, save_as_tikz=False)

    # %%
    predict_poses = 4
    start_idx = settings[mode]['startidx']
    nexps = settings[mode]['nexps']

    print('calc predictions errors....')

    (ALPERR, PERR, EPSERR, alpsig, psig, epsig, gait_predicted, DXm, DXsig,
     depsm, depssig, deps_sim_m, deps_sim_sig, DXsim_m, DXsim_sig) = \
        uti.calc_errors(db, POSE_IDX, version, mode=mode,
                        nexps=nexps, predict_poses=predict_poses,
                        start_idx=start_idx)

# %%
    for idx in PERR:
        MU_P[idx][mode] = PERR[idx][1:]
        SIG_P[idx][mode] = psig[idx][1:]
        MU_DX[idx][mode] = DXm[idx][1:]
        SIG_DX[idx][mode] = DXsig[idx][1:]
        MU_DX_SIM[idx][mode] = DXsim_m[idx][1:]
        SIG_DX_SIM[idx][mode] = DXsim_sig[idx][1:]

    MU_EPS[mode] = EPSERR[1:]
    SIG_EPS[mode] = epsig[1:]

    MU_DEPS[mode] = depsm[1:]
    SIG_DEPS[mode] = depssig[1:]
    MU_DEPS_SIM[mode] = deps_sim_m[1:]
    SIG_DEPS_SIM[mode] = deps_sim_sig[1:]

# %%

    gait_predicted.save_as_tikz(mode+'_gait')


# %%
#    plt.figure('PredictionErrors-ALP')
#    for idx in range(len(ALPERR)):
#        plt.plot(ALPERR[idx], label='marker {}'.format(idx))
#    plt.xlabel('Step count')
#    plt.ylabel('Prediction Error of Angle |a_m - a_p|')

# %% barplot

if 0:  # old plots / relative plots
    # POS
    for idx in [1]:
        mu = MU_P[idx]
        sig = SIG_P[idx]
        N = max([len(v) for v in mu.values()])
        ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                         sig, num='error-p')
    #    ax.set_ylim((0, 330))
        ax.set_ylabel('$|{p}_m - {p}_p|/\\ell_{{n}}$ (\%)')
        ax.set_xlabel('Pose Count')
        ax.grid(True, axis='y')
    
        kwargs = {'extra_axis_parameters': {'anchor=origin',
                                            'axis line style={draw=none}',
                                            'xtick style={draw=none}'}}
        my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_position_error_bar.tex',
                                 **kwargs)
    
    # %% EPS
    
    mu = MU_EPS
    sig = SIG_EPS
    N = max([len(v) for v in mu.values()])
    ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                     sig, num='error-eps')
    ax.set_ylabel('$|{eps}_m - {eps}_p|$ (deg)')
    ax.set_xlabel('Pose Count')
    ax.grid(True, axis='y')
    
    kwargs = {'extra_axis_parameters': {'anchor=origin',
                                        'axis line style={draw=none}',
                                        'xtick style={draw=none}'}}
    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_epsilon_error_bar.tex',
                             **kwargs)


# %% ABSOLUTE PLOTS
# %% DEPS

kwargs = {'extra_axis_parameters': {'anchor=origin',
                                    'axis line style={draw=none}',
                                    'xtick style={draw=none}',
                                    'height=6cm',
                                    'width=10cm'}}


mu = MU_DEPS
sig = SIG_DEPS
N = max([len(v) for v in mu.values()])
ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                 sig, num='error-deps')

ax.set_ylabel('${eps}_m - {eps}_0$ (deg)')
ax.set_xlabel('pose count')
ax.grid(True, axis='y')
ax.set_ylim((0, 170))

my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_deps.tex',
                         **kwargs)


# %% DEPS_SIM

mu = MU_DEPS_SIM
sig = None
N = max([len(v) for v in mu.values()])
ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                 sig, num='error-deps_sim')

ax.set_ylabel('${eps}_p - {eps}_0$ (deg)')
ax.set_xlabel('pose count')
ax.grid(True, axis='y')
ax.set_ylim((0, 170))

my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_deps_sim.tex',
                         **kwargs)


# %% DX

for idx in [1]:
    mu = MU_DX[idx]
    sig = SIG_DX[idx]
    N = max([len(v) for v in mu.values()])
    ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                     sig, num='dx')
    ax.set_ylim((0, 500))
    ax.set_ylabel('$Delta x_m/l_n$ (%)')
    ax.set_xlabel('pose count')
    ax.grid(True, axis='y')

    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_dx.tex',
                             **kwargs)


# %% DX_SIM

for idx in [1]:
    mu = MU_DX_SIM[idx]
    sig = None
    N = max([len(v) for v in mu.values()])
    ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                     sig, num='dx_sim')
    ax.set_ylim((0, 500))
    ax.set_ylabel('$Delta x_p/l_n$ (%)')
    ax.set_xlabel('pose count')
    ax.grid(True, axis='y')

    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_dx_sim.tex',
                             **kwargs)


# %%
plt.show()