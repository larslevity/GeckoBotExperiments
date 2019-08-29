# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:26:12 2019

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
from Src import plot_fun_pathPlanner as pf


version = 'vS11'
runs = ['180', 'L', 'R', 'SL', 'SR']
#runs = ['180']


for run in runs:

    # %% ### Load Data

    dirpath = run + '/'

    sets = load.get_csv_set(dirpath)
    db = ev.load_data_pathPlanner(dirpath, sets)
    # %%
    prop = pf.calc_prop(db)


    # %% ### eps during cycle
#    TIME = pf.plot_eps(db, cyc, run, prop, dirpath)

    # %% ### Track of feet:
    pf.plot_track(db, run, prop, dirpath)

    # %% ### Alpha during cycle:
#    ALPHA, SIGALPHA, timestamps, alp_dfx_0, alp_dfx_1 = \
#        pf.plot_alpha(db, cyc, run, prop, dirpath)
#    ALP[incl] = ALPHA
#    SIGA[incl] = SIGALPHA
#    TIMEA[incl] = timestamps
#    ALP_dfx_0[incl] = alp_dfx_0
#    ALP_dfx_1[incl] = alp_dfx_1

    # %% Velocity

#    VELX, SIGVELX, VELY, SIGVELY = \
#        pf.plot_velocity(db, cyc, incl, prop, dirpath, Ts, DIST, TIME,
#                         VELX, VELY, SIGVELX, SIGVELY)

    # %% ### pressure during cycle:
#    ENERGY = pf.plot_pressure(db, cyc, incl, prop, dirpath, ptrn, VOLUME,
#                              version, DIST, ENERGY)

# %% VEL and ENERGY for incl

#pf.plot_vel_incl(VELX, VELY, SIGVELX, SIGVELY, INCL, ENERGY, version, ptrn)
#
#
## %% Plot Alpha INCL
#
#pf.plot_incl_alp_dfx(TIMEA, incls, ALP, ALP_dfx_0, ALP_dfx_1, version, ptrn)


plt.show()