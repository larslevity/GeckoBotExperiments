# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval_pathPlanner as ev
from Src import save
from Src import load
from Src import kin_model
from Src import predict_pose as pp
from Src import plot_fun_pathPlanner as pf


modes = ['without_eps_correction']

runs = ['L', '180', 'R', 'RFL']
#runs = ['R']


for mode in modes:
    for run in runs:
        # %% ### Load Data

        dirpath = mode + '/' + run + '/'

        sets = load.get_csv_set(dirpath)
        db = ev.load_data_pathPlanner(dirpath, sets)
        # %%
        prop = pf.calc_prop(db)

        # %% ### Track of feet:
        pf.plot_track(db, run, prop, mode)

plt.show()