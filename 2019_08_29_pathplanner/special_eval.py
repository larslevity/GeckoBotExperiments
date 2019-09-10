# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:00:45 2019

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


modes = ['without_eps_correction']

runs = ['special3_x1_60', 'special3_x1_60_px_shift_RPi']
runs = ['special3_x1_60_px_shift_RPi']
#runs = ['180']


for run in runs:

    # %% ### Load Data

    dirpath = run + '/'

    sets = load.get_csv_set(dirpath)
    db = ev.load_data_pathPlanner(dirpath, sets)
    # %%
    prop = pf.calc_prop(db)

    # %% ### Track of feet:
    pf.plot_track(db, run, prop, dirpath)



plt.show()