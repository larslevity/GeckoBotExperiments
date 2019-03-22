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
from Src import plot_fun as pf


version = 'vS11'
ptrn = 'adj_ptrn_new'
incls = ['00', '28', '48', '63', '76', '83']
#incls = ['48']
Ts = 0.03
VOLUME = {'v40': 0.01105,
          'vS11': .00376}



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