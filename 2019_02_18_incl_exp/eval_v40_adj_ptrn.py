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
from Src import plot_fun as pf


version = 'v40'
ptrn = 'adj_ptrn'
incls = ['00', '28', '48', '63', '76']
Ts = 0.03
VOLUME = {'v40': 0.01105,
          'vS11': .00376}


DEBUG = False

INCL, VELX, VELY, ENERGY, ALP, TIMEA, ALP_dfx_0 = [], [], [], [], {}, {}, {}
ALP_dfx_1 = {}
SIGVELX, SIGVELY, SIGENERGY, SIGA = [], [], [], {}


for incl in incls:
    INCL.append(int(incl))
    # %% ### Load Data

    dirpath = version+'/'+ptrn+'/incl_'+incl+'/'

    sets = load.get_csv_set(dirpath)
    db, cyc = ev.load_data(dirpath, sets)

    prop = pf.calc_prop(db, cyc)
    db = pf.epsilon_correction(db, cyc)

    # %% ### eps during cycle
    TIME = pf.plot_eps(db, cyc, incl, prop, dirpath)

    # %% ### Track of feet:
    DIST = pf.plot_track(db, cyc, incl, prop, dirpath)

    # %% ### Alpha during cycle:
    ALPHA, SIGALPHA, timestamps, alp_dfx_0, alp_dfx_1 = \
        pf.plot_alpha(db, cyc, incl, prop, dirpath)
    ALP[incl] = ALPHA
    SIGA[incl] = SIGALPHA
    TIMEA[incl] = timestamps
    ALP_dfx_0[incl] = alp_dfx_0
    ALP_dfx_1[incl] = alp_dfx_1

    # %% Velocity

    # %% Velocity

    VELX, SIGVELX, VELY, SIGVELY = \
        pf.plot_velocity(db, cyc, incl, prop, dirpath, Ts, DIST, TIME,
                         VELX, VELY, SIGVELX, SIGVELY)

    # %% ### pressure during cycle:
    ENERGY = pf.plot_pressure(db, cyc, incl, prop, dirpath, ptrn, VOLUME,
                              version, DIST, ENERGY)

# %% VEL and ENERGY for incl

pf.plot_vel_incl(VELX, VELY, SIGVELX, SIGVELY, INCL, ENERGY, version, ptrn)

# %% Plot Alpha INCL

pf.plot_incl_alp_dfx(TIMEA, incls, ALP, ALP_dfx_0, ALP_dfx_1, version, ptrn)


plt.show()
