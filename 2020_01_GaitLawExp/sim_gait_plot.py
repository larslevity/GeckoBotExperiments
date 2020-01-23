# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:52:26 2020

@author: AmP
"""


import matplotlib.pyplot as plt
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from Src import save as my_save
import plotfun_GaitLawExp as pf
import gait_law_utils as uti
from Src import calibration
from Src import inverse_kinematics
from Src import roboter_repr
from Src import kin_model as model

Q1 = np.array([50, 60, 70, 80, 90])
Q2 = np.array([-.5, -.3, -.1, .1, .3, .5])

RESULT_DX = np.zeros((len(Q2), len(Q1)))
RESULT_DY = np.zeros((len(Q2), len(Q1)))
RESULT_DEPS = np.zeros((len(Q2), len(Q1)))
RESULT_STRESS = np.zeros((len(Q2), len(Q1)))
X_idx = np.zeros((len(Q2), len(Q1)))
Y_idx = np.zeros((len(Q2), len(Q1)))
GAITS = []

n_cyc = 1
and_half = True  # doc:True, #IROS:FALSE
sc = 10  # scale factor
dx, dy = 2.8*sc, (4.5)*sc
version = 'vS11'

len_leg, len_tor = calibration.get_len(version)
ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]


weight = [89, 10, 5.9]   # [f_l, f_o, f_a]
eps = 90


def cut(x):
    return x if x > 0.001 else 0.001


def alpha2(x1, x2, f, c1=1):
    alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
             cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1),
             x1 + x2*abs(x1),
             cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
             cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1)
             ]
    return alpha


alpha = alpha2


for q1_idx, q1 in enumerate(Q1):
    q1str = str(q1)
    for q2_idx, q2 in enumerate(Q2):
        q2str = str(q2).replace('.', '').replace('00', '0')
        X_idx[q2_idx][q1_idx] = q2_idx*dx
        Y_idx[q2_idx][q1_idx] = q1_idx*dy

        f1 = [0, 1, 1, 0]
        f2 = [1, 0, 0, 1]
#        if q2 < 0:
        if 1:
            ref2 = [[alpha(-q1, q2, f2), f2],
                    [alpha(q1, q2, f1), f1]
                    ]
#        else:
#            ref2 = [[alpha(q1, q2, f1), f1],
#                    [alpha(-q1, q2, f2), f2]
#                    ]
        ref2 = ref2*n_cyc
        if and_half:
            ref2 += [ref2[0]]

        # get start positions
        init_pose = roboter_repr.GeckoBotPose(
                *model.set_initial_pose(ref2[0][0], eps,
                                        (q2_idx*dx, q1_idx*dy),
                                        len_leg=len_leg,
                                        len_tor=len_tor))
        gait_ = roboter_repr.predict_gait(ref2, init_pose, weight,
                                          (len_leg, len_tor))
        alp_ = gait_.poses[-1].alp
        ell_ = gait_.poses[-1].ell

        init_pose = roboter_repr.GeckoBotPose(
                *model.set_initial_pose(alp_, eps,
                                        (q2_idx*dx, q1_idx*dy),
                                        ell=ell_),
                len_leg=len_leg,
                len_tor=len_tor)

        # actually simulation
        gait = roboter_repr.predict_gait(ref2, init_pose, weight,
                                         (len_leg, len_tor))

        (dxx, dyy), deps = gait.get_travel_distance()
        RESULT_DX[q2_idx][q1_idx] = dxx
        RESULT_DY[q2_idx][q1_idx] = dyy
        RESULT_DEPS[q2_idx][q1_idx] = deps
        print('(x2, x1):', round(q2, 2), round(q1, 1), ':',
              round(deps, 2))

        GAITS.append(gait)


# %% EPS / GAIT
print('create figure: EPS/GAIT')

fig = plt.figure('GeckoBotGait')
levels = np.arange(-65, 66, 5)
if len(Q1) > 1:
    contour = plt.contourf(X_idx, Y_idx, RESULT_DEPS*n_cyc, alpha=1,
                           cmap='RdBu_r', levels=levels)
# surf = plt.contour(X_idx, Y_idx, DEPS, levels=levels, colors='k')
# plt.clabel(surf, levels, inline=True, fmt='%2.0f')


gait_tex = ''

for gait in GAITS:
    gait.plot_orientation(length=.5*sc)
    gait_tex = gait_tex + '\n%%%%%%%\n' + gait.get_tikz_repr(linewidth='.7mm')

for q1_idx, q1 in enumerate(Q1):
    for q2_idx, q2 in enumerate(Q2):
        dx = RESULT_DX[q2_idx][q1_idx]
        dy = RESULT_DY[q2_idx][q1_idx]
        start = GAITS[q1_idx*len(Q2)+q2_idx].poses[0].get_m1_pos()
        plt.arrow(start[0], start[1], dx, dy, facecolor='red',
                  length_includes_head=1,
                  width=.9,
                  head_width=3)


for xidx, x in enumerate(list(RESULT_DEPS)):
    for yidx, deps in enumerate(list(x)):
        plt.text(X_idx[xidx][yidx], Y_idx[xidx][yidx]-2.2*sc,
                 '$'+str(round(deps*n_cyc, 1))+'^\\circ$',
                 ha="center", va="bottom",
                 fontsize=25,
                 bbox=dict(boxstyle="square",
                           ec=(.5, 1., 0.5),
                           fc=(.8, 1., 0.8),
                           ))


plt.xticks(X_idx.T[0], [round(x, 2) for x in Q2])
plt.yticks(Y_idx[0], [round(x, 1) for x in Q1])
plt.ylabel('step length $q_1$ $(^\\circ)$')
plt.xlabel('steering $q_2$ (1)')
plt.axis('scaled')
plt.ylim((Y_idx[0][0]-30, Y_idx[0][-1]+30))
plt.xlim((-15, 155))

plt.grid()
ax = fig.gca()
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


my_save.save_plt_as_tikz('Out/simulation/gait_'+str(weight)+'.tex',
                         additional_tex_code=gait_tex,
                         scale=.7,
                         scope='scale=.1, opacity=.8')

fig.set_size_inches(10.5, 8)
fig.savefig('Out/simulation/gait_'+str(weight)+'.png', transparent=True,
            dpi=300, bbox_inches='tight')
