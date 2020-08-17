#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:52:20 2020

@author: ls
"""
from tikzplotlib import save as tikz_save

import utils as uti


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from Src import calibration
from Src import kin_model
from Src import roboter_repr as robrepr

# %%

modes = [
#        'straight_1',
#        'straight_2',
#        'straight_3',
        'curve_1',
        'curve_2',
        'curve_3',
        ]


version = 'vS12'


# %%

DX_SIM = {}
DY_SIM = {}
DEPS_SIM = {}



n_cyc = 1
exp_idx = 0
eps0 = 0
pos0 = (0, 0)

len_leg, len_tor = calibration.get_len(version)
ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

REF = {
    'straight_1':  [[[90, 0, -90, 90, 0], [1, 0, 0, 1]],
                    [[0, 90, 90, 0, 90], [0, 1, 1, 0]]],
    'straight_2':  [[[86, 4, -110, 83, 4], [1, 0, 0, 1]],
                    [[4, 86, 110, 4, 83], [0, 1, 1, 0]]],
    'straight_3':  [[[0, 18, -85, 10, 22], [1, 0, 0, 1]],
                    [[18, 0, 85, 22, 10], [0, 1, 1, 0]]],
    'curve_1':  [[[97, 28, -98, 116, 17], [1, 0, 0, 1]],
                 [[79, 0, -84, 67, 0], [0, 1, 1, 0]]],
    'curve_2':  [[[104, 48, -114, 124, 27], [1, 0, 0, 1]],
                 [[72, 0, -70, 55, 0], [0, 1, 1, 0]]],
    'curve_3':  [[[164, 124, -152, 221, 62], [1, 0, 0, 1]],
                 [[0, 0, -24, 0, 0], [0, 1, 1, 0]]],
        }

weight = {
    'straight_1':  [.1, 10, 10],
    'straight_2':  [.1, 1, 10],
    'straight_3':  [.1, .1, 10],
    'curve_1':  [.1, 10, 10],
    'curve_2':  [.1, 1, 10],
    'curve_3':  [.1, .1, 10],
        }


for mode_idx, mode in enumerate(modes):
    # %% SIM
    ref = REF[mode]
    w = weight[mode]
    
    x, fpos, fix = kin_model.set_initial_pose(ref[0][0], eps0 , pos0,
                                           len_leg=len_leg, len_tor=len_tor)
    gait = robrepr.GeckoBotGait()
    for n in range(n_cyc):
        for r in ref:
            x, fpos, fix, constraint, cost = kin_model.predict_next_pose(
                    r, x, fpos, f=w, len_leg=len_leg, len_tor=len_tor)
            gait.append_pose(robrepr.GeckoBotPose(x, fpos, fix,
                                    len_leg=len_leg, len_tor=len_tor))
    # half cycle
    x, fpos, fix, constraint, cost = kin_model.predict_next_pose(
            ref[0], x, fpos, f=w, len_leg=len_leg, len_tor=len_tor)
    gait.append_pose(robrepr.GeckoBotPose(x, fpos, fix,
                            len_leg=len_leg, len_tor=len_tor))
    gait.plot_gait(mode)
    gait.save_as_tikz('Out/gaits/'+ mode + '.tex')
    DEPS_SIM[mode_idx] = x[-1] - eps0
    DX_SIM[mode_idx] = fpos[0][1] - pos0[0]  # only front torso
    DY_SIM[mode_idx] = fpos[1][1] - pos0[1]

# %% ABSOLUTE PLOTS
# %% DEPS

kwargs = {'extra_axis_parameters': 
            {'anchor=origin',
             'axis line style={draw=none}',
             'xtick style={draw=none}',
             'height=6cm',
             'width=10cm',
#             'legend style={at={(.1,.9), anchor=north west}}',
#             },
#        'extra_axis_code: {
'''\\pgfplotsset{
    legend image with color/.style={
    legend image code/.code={%
        \\node[anchor=center, rectangle, fill=#1] at (0.3cm,0cm) {};
    }
},
}
\\addlegendimage{legend image with color=blue};
\\addlegendentry{$w_\\varphi = 10$};
\\addlegendimage{legend image with color=red};
\\addlegendentry{$w_\\varphi = 1$};
\\addlegendimage{legend image with color=color0};
\\addlegendentry{$w_\\varphi = 0.1$};''',
             },
#          'extra_tikzpicture_parameters':
#            {'legend image with color/.style={legend image code/.code={\node[anchor=center, rectangle, fill=#1] at (0.3cm,0cm) {}}};',
#            },
        }

colors = {
        '$w_{\\varphi}=10$': 'blue',
        '$w_{\\varphi}=1$': 'red',
        '$w_{\\varphi}=0.1$': 'orange',
        }


def mapped(modes):
    mapped = []
    for mode in modes:
        if mode[-1] == '1':
            val = '10'
        elif mode[-1] == '2':
            val = '1'
        elif mode[-1] == '3':
            val = '0.1'
        mapped.append('$w_{\\varphi}=%s$' % val)
    return mapped


mu = {mode: [abs(DEPS_SIM[mode_idx]), abs(DX_SIM[mode_idx])/ell_n[2]*100, abs(DY_SIM[mode_idx])/ell_n[2]*100] for mode_idx, mode in enumerate(mapped(modes))}
#sig = {mode: [DEPSSIG[mode_idx], DXSIG[mode_idx]/ell_n[2]*100, DYSIG[mode_idx]/ell_n[2]*100] for mode_idx, mode in enumerate(mapped(modes))}

labels = ['$\Delta \epsilon ~ (^\circ)$', '$\Delta x / \ell_n ~ (\%)$', '$\Delta y / \ell_n ~ (\%)$']

ax = uti.barplot(mu, mapped(modes), labels, colors, num='error-deps')

ax.set_ylabel('')
ax.set_xlabel('')
ax.grid(True, axis='y')
#ax.set_ylim((0, 290))  # straight
ax.set_ylim((0, 130))  # curve



tikz_save('Out/sim_' + mode[:-2] + '.tex', standalone=True, **kwargs)
#