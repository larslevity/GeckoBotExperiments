# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:25:49 2019

@author: ls
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval as ev
from Src import save
from Src import load
from Src import kin_model
from Src import predict_pose as pp

# %% ### Load Data

col = ev.get_marker_color()

version = 'v40'
ptrn = 'adj_ptrn'
incl = '76'

 

sets = load.get_csv_set(version+'/'+ptrn+'/incl_'+incl+'/')
#sets = [sets[4]]
#print sets
db, cyc = ev.load_data(version+'/'+ptrn+'/incl_'+incl+'/', sets)

# correction of jump epsilon
for exp in range(len(db)):
    db[exp]['eps'] = ev.shift_jump(db[exp]['eps'], 180)

# correction of epsilon
rotate = 5
for exp in range(len(db)):
    eps0 = db[exp]['eps'][cyc[exp][1]]
    for marker in range(6):
        X = db[exp]['x{}'.format(marker)]
        Y = db[exp]['y{}'.format(marker)]
        X, Y = ev.rotate_xy(X, Y, -eps0+rotate)
        db[exp]['x{}'.format(marker)] = X
        db[exp]['y{}'.format(marker)] = Y
    db[exp]['eps'] = ev.add_offset(db[exp]['eps'], -eps0+rotate)


# %% ### eps during cycle
plt.figure('Epsilon corrected')
for exp in range(len(db)):
    for idx in range(6):
        eps = db[exp]['eps'][cyc[exp][1]:cyc[exp][-1]]
        # hack to remove high freq noise
        for idx in range(1, len(eps)):
            if abs(eps[idx] - eps[idx-1]) > 30:
                eps[idx] = eps[idx-1]
        t = db[exp]['time'][cyc[exp][1]:cyc[exp][-1]]
        plt.plot(t, eps, ':', color='mediumpurple')

eps, sige = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'eps')
t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
eps = np.array(ev.downsample(list(eps)))
t = ev.downsample(t)
sige = np.array(ev.downsample(sige))

plt.plot(t, eps, '-', color='mediumpurple', linewidth=2)
plt.fill_between(t, eps+sige, eps-sige,
                 facecolor='mediumpurple', alpha=0.5)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('orientation angle epsilon (deg)')
# save.save_as_tikz('pics/'+version+'_'+ptrn+'_'+incl+'.tex'.format(idx))


# %%  check for quality:
fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'), num='Quality Check')
for exp in range(len(db)):
    for idx in range(6):
        x = db[exp]['x{}'.format(idx)][cyc[exp][1]:cyc[exp][-1]]
        y = db[exp]['y{}'.format(idx)][cyc[exp][1]:cyc[exp][-1]]
        plt.plot(x, y, color=col[idx])

for idx in range(6):
    x, sigx = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'x{}'.format(idx))
    y, sigy = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'y{}'.format(idx))
    plt.plot(x, y, linewidth=4, color=col[idx])
    plt.plot(x[0], y[0], 'o', markersize=20, color=col[idx])
#    for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
#        el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
#                         facecolor=col[idx], alpha=.3)
#        ax.add_artist(el)
ax.grid()
ax.set_xlabel('x position (cm)')
ax.set_ylabel('y position (cm)')

plt.show()
