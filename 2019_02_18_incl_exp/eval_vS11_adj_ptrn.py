# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:25:49 2019

@author: ls
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


version = 'vS11'
ptrn = 'adj_ptrn'
incls = ['00', '28', '48', '63', '76', '84']
#incls = ['48']
Ts = 0.03
VOLUME = {'v40': 0.01105,
          'vS11': .00376}


DEBUG = False

INCL, VELX, VELY, ENERGY = [], [], [], []
SIGVELX, SIGVELY, SIGENERGY = [], [], []


for incl in incls:
    INCL.append(int(incl[:2]))

    # %% ### load data

    dirpath = version+'/'+ptrn+'/incl_'+incl+'/'

    sets = load.get_csv_set(dirpath)
    db, cyc = ev.load_data(dirpath, sets)

    prop = pf.calc_prop(db, cyc)
    db = pf.epsilon_correction(db, cyc)

    # %% ### eps during cycle
    TIME = pf.plot_eps(db, cyc, incl, prop, dirpath)

    # %% ### Track of feet:
#    for exp in [9]:
#        for idx in range(6):
#            plt.plot(db[exp]['x%i' % idx][cyc[exp][2]:],
#                     db[exp]['y%i' % idx][cyc[exp][2]:], color=ev.get_marker_color()[idx])
    DIST = pf.plot_track(db, cyc, incl, prop, dirpath)

    # %% ### Alpha during cycle:
    pf.plot_alpha(db, cyc, incl, prop, dirpath)

    # %% Velocity

    VELX, SIGVELX, VELY, SIGVELY = \
        pf.plot_velocity(db, cyc, incl, prop, dirpath, Ts, DIST, TIME,
                         VELX, VELY, SIGVELX, SIGVELY)

    # %% ### pressure during cycle:
    ENERGY = pf.plot_pressure(db, cyc, incl, prop, dirpath, ptrn, VOLUME,
                              version, DIST, ENERGY)

# %% VEL and ENERGY for incl

pf.plot_vel_incl(VELX, VELY, SIGVELX, SIGVELY, INCL, ENERGY, version, ptrn)


plt.show()
