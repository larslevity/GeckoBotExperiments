# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:29:37 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval as ev
from Src import save
from Src import load
from Src import kin_model
from Src import predict_pose as pp
from Src import plot_fun as pf


incl = [0, 28, 48, 63, 76, 84]

p_vS11 = {0: [.81, .76, .73, .78, .73, .76],
          28: [.78, .74, .91, .91, .71, .79],
          48: [.67, .69, .96, .99, .71, .70],
          63: [.64, .64, 1.1, 1.1, .68, .68],
          76: [.67, .67, 1.1, 1.1, .66, .68],
          84: [.69, .69, .96, .99, .71, .65]}


p_v40 = {0: [.81, .81, .79, .84, .78, .76],
         28: [.69, .72, .95, .94, .76, .74],
         48: [.62, .65, .98, .97, .73, .71],
         63: [.60, .63, .99, .99, .68, .66],
         76: [.60, .63, .90, .90, .68, .66],
         84: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}


col = ev.get_actuator_color()
plt.figure(0)
for axis in range(6):
    p = [p_vS11[inc][axis] for inc in incl]
    plt.plot(incl, p, color=col[axis])

plt.grid()
plt.xlabel(r'inclination angle $\delta$ ($^\circ$)')
plt.ylabel(r'$p_{\mathrm{ref}}$ (bar)')

kwargs = {'extra_axis_parameters': {'xtick={0, 28, 48, 63, 76, 84}'}}
save.save_as_tikz('tikz/vS11-incl-p.tex', **kwargs)
print('MEAN of p vS11: ', np.mean(p_vS11[0]))


plt.figure(1)
for axis in range(6):
    p = [p_v40[inc][axis] for inc in incl]
    plt.plot(incl, p, color=col[axis])

plt.grid()
plt.xlabel(r'inclination angle $\delta$ ($^\circ$)')
plt.ylabel(r'$p_{\mathrm{ref}}$ (bar)')
kwargs = {'extra_axis_parameters': {'xtick={0, 28, 48, 63, 76, 84}'}}
save.save_as_tikz('tikz/v40-incl-p.tex', **kwargs)
print('MEAN of p v40: ', np.mean(p_v40[0]))
