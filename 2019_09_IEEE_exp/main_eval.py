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
    'straight_1': {'nexps': [0, 1, 2, 3],
                   'startidx': 4},
    'straight_2': {'nexps': [0, 1],
                   'startidx': 0},
    'straight_3': {'nexps': [0, 1, 2],
                   'startidx': 8},
        }


version = 'v40'


# %%

n_steps = {}
MU_P = {}
SIG_P = {}
for idx in range(6):
    MU_P[idx] = {}
    SIG_P[idx] = {}

MU_EPS = {}
SIG_EPS = {}

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
    
    ALPERR, PERR, EPSERR, alpsig, psig, epsig, gait_predicted = \
        uti.calc_errors(db, POSE_IDX, version, mode=mode,
                        nexps=nexps, predict_poses=predict_poses,
                        start_idx=start_idx)

# %%
    for idx in PERR: 
        MU_P[idx][mode] = PERR[idx][1:]
        SIG_P[idx][mode] = psig[idx][1:]
    
    MU_EPS[mode] = EPSERR[1:]
    SIG_EPS[mode] = epsig[1:]

    # %%
#    col = pf.get_marker_color()
#    plt.figure('PredictionErrorsP')
#    markers = [1]
#    for idx in markers:
#        plt.plot(PERR[idx], label='marker {}'.format(idx), color=col[idx])
#        mu, sig = PERR[idx], psig[idx]
#        plt.fill_between(range(len(mu)), mu+sig, mu-sig,
#                         facecolor=col[idx], alpha=0.5)
#    plt.ylabel('Prediction Error of Position |p_m  p_p|', color=col[idx])
#    plt.gca().tick_params('y', colors=col[idx])
#
#
#    ax = plt.gca().twinx()
#    ax.plot(EPSERR, label='eps', color='purple')
#    ax.fill_between(range(len(EPSERR)), EPSERR+epsig, EPSERR-epsig,
#                         facecolor='purple', alpha=0.5)
#    ax.set_ylabel('Prediction Error of EPS |e_m  e_p|', color='purple')
#    ax.tick_params('y', colors='purple')
#
#    plt.xlabel('Step count')
#    ax.set_xticks([int(x) for x in range(predict_poses+1)])
#    plt.grid()
#
#    plt.savefig('Out/PredictionERROR_'+str(mode)+'_startIDX_'+str(start_idx)+'.png',
#                dpi=300)
#    
##    fdir = path.dirname(path.abspath(__file__))
#    my_save.save_plt_as_tikz('/Out/' + mode + '_error.tex')
#    tikz_save(mode + '_error.tex')

# %%
#    col = pf.get_marker_color()
#    markers = [1]
#    for idx in markers:
#        mu = PERR[idx][1:]
#        sig = psig[idx][1:]
#        fig, ax = plt.subplots()
#        ax.bar(range(len(mu)), mu,
#               yerr=sig,
#               align='edge',
#               width=-.4,
#               alpha=0.5,
#               ecolor='black', color=col[idx],
#               capsize=10)
#        ax.tick_params('y', colors=col[idx])
#        ax.set_ylabel('$|\\bm{p}_m - \\bm{p}_p|/\\ell_{\\textnormal{n}}$ (\\%)',
#                      {'color': col[idx]})
#        ax.set_ylim((-.5, 4))
#        ax.set_xticks(range(len(mu)))
#        ax.set_xticklabels([' ' for i in range(len(mu))])
##        ax.yaxis.grid(True)
#    # eps
#    mu = EPSERR[1:]
#    sig = epsig[1:]
#    ax = ax.twinx()
#    ax.bar(range(len(mu)), -mu,
#           yerr=-sig,
#           align='edge',
#           width=.4,
#           alpha=0.5,
#           ecolor='black', color='purple',
#           capsize=10)
#    ax.tick_params('y', colors='purple')
#    ax.yaxis.set_label_position("right")
#    ax.yaxis.tick_right()
#    ax.set_ylabel('$\\varepsilon_m - \\varepsilon_p$ ($^\circ$)',
#                  {'color': 'purple'})
#    ax.set_xlabel('Pose Count')
#    ax.set_ylim((-10, 80))
#    ax.set_xticks(range(len(mu)))
#    ax.set_xticklabels([int(i+1) for i in range(len(mu))])
#    ax.yaxis.grid(True)
#    
#    kwargs = {'extra_axis_parameters':
#                  {'anchor=origin', 'axis line style={draw=none}',
#                   'xtick style={draw=none}', 'ytick pos=right',
#                   'ytick pos=left',
#                   'ytick style={color=color0}',
#                   'yticklabel style={color=color0}',
#                   'ylabel style={color=color0}'
#                   }
#            }
#    my_save.save_plt_as_tikz('/Out/' + mode + '_error_bar.tex',
#                             **kwargs)




# %%

    gait_predicted.save_as_tikz(mode+'_gait')


# %%
    plt.figure('PredictionErrors-ALP')
    for idx in range(len(ALPERR)):
        plt.plot(ALPERR[idx], label='marker {}'.format(idx))
    plt.xlabel('Step count')
    plt.ylabel('Prediction Error of Angle |a_m - a_p|')

# %% barplot

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

    kwargs = {'extra_axis_parameters':
                  {'anchor=origin', 'axis line style={draw=none}',
                   'xtick style={draw=none}'
                   }
            }
    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_position_error_bar.tex',
                             **kwargs)


# %% EPS

mu = MU_EPS
sig = SIG_EPS
N = max([len(v) for v in mu.values()])
ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
                 sig, num='error-eps')
#ax.set_ylim((-5, 80))
ax.set_ylabel('$|{eps}_m - {eps}_p|$ (deg)')
ax.set_xlabel('Pose Count')
ax.grid(True, axis='y')

kwargs = {'extra_axis_parameters':
              {'anchor=origin', 'axis line style={draw=none}',
               'xtick style={draw=none}'
               }
        }
my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_epsilon_error_bar.tex',
                         **kwargs)


# %%
plt.show()