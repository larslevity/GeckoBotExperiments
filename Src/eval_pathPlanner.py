# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:04:03 2019

@author: AmP
"""

import numpy as np

from Src import load


def load_data_pathPlanner(path, sets):
    dataBase = []
#    xscale = 145./1000  # 1000px -> 145cm
#    xshift = -22  # cm
#    yshift = -63  # cm
    xscale = 112./1000  # after changing resolution of RPi
    xshift = -12  # cm
    yshift = -45  # cm

    for exp in sets:
        data = load.read_csv(path+"{}.csv".format(exp))

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


def find_poses_idx(db, r3_init=.44, neighbors=5):
    IDX = []
    failed = 0
    for exp_idx in range(len(db)):
        pose_idx = []
        start_idx = db[exp_idx]['r3'].index(r3_init)
        for idx in range(start_idx, len(db[exp_idx]['r3'])-1, 1):
            if db[exp_idx]['r3'][idx] != db[exp_idx]['r3'][idx+1]:
                if not pose_idx:  # empty list
                    pose_idx.append(idx)
                else:
                    for jdx in range(idx, idx-neighbors, -1):  # look the last neigbors
                        if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                            # check
                            dr = db[exp_idx]['r2'][idx] - db[exp_idx]['r2'][jdx]
                            if abs(dr) > .1:
                                failed +=1
                                pose_idx.append(idx)  # append ori
                                break
                            else:
                                pose_idx.append(jdx)
                                break
                        elif jdx == idx-neighbors+1:
                            failed += 1
                            pose_idx.append(idx)  # append ori
        #last#
        idx = len(db[exp_idx]['r3'])-1
        for jdx in range(idx, idx-100, -1):  # look the last neigbors
            if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                # check
                dr = db[exp_idx]['r2'][idx] - db[exp_idx]['r2'][jdx]
                if abs(dr) > .1:
                    failed +=1
                    pose_idx.append(idx)  # append ori
                    break
                else:
                    pose_idx.append(jdx)
                    break
        IDX.append(pose_idx)
        print('failed:', failed)
    return IDX





