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
runs = ['n1_x1_50', 'n2_x1_50', 'n2_x1_90', 'n3_x1_90']
runs = ['improved_clb']
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