# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:04:03 2019

@author: AmP
"""

import numpy as np

from Src import load


def load_data_pathPlanner(exp_name, exp_idx=['00'], Ts=.03):
    dataBase = []
#    xscale = 145./1000  # 1000px -> 145cm
#    xshift = -22  # cm
#    yshift = -63  # cm
    xscale = 112./1000  # after changing resolution of RPi
    xshift = -12  # cm
    yshift = -45  # cm

    for exp in exp_idx:
        data = load.read_csv(exp_name+"{}.csv".format(exp))

        try:
            start_idx = data['f0'].index(1)  # upper left foot attached 1sttime
        except ValueError:  # no left foot is fixed
            start_idx = 0
        start_time = data['time'][start_idx]
        data['time'] = \
            [round(data_time - start_time, 3) for data_time in data['time']]
        for key in data:
            if key[0] in ['x', 'y']:
                shift = xshift if key[0] == 'x' else yshift
                data[key] = [i*xscale + shift for i in data[key]]

        dataBase.append(data)

    return dataBase
