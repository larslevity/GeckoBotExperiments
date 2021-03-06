# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:07:46 2020

@author: AmP
"""

import matplotlib.pyplot as plt
import numpy as np


import load


versions = ['vS10']

clb = {}

for version in versions:
    clb[version] = {}
    dirpath = version + '/'
    sets = load.get_csv_set(dirpath)
    db = load.load_data(dirpath, sets)

    alpha = {}
    pressure = {}
    reference = {}
    reference_ = {}

    for exp in range(len(db)):
        for axis in range(6):
            alpha[axis] = np.array(db[exp]['aIMG{}'.format(axis)])
#            pressure[axis] = np.array(db[exp]['pr{}'.format(axis)])
            reference_[axis] = np.array(db[exp]['pr{}'.format(axis)])
            # remove all nans:
#            pressure[axis] = pressure[axis][~np.isnan(alpha[axis])]
            reference[axis] = reference_[axis][~np.isnan(alpha[axis])]
            alpha[axis] = alpha[axis][~np.isnan(alpha[axis])]

    # %% Pressure-alpha relation
    for axis in range(6):
        deg = 5

        # only vals with ref!=0
        idx = reference[axis] != 0

        # only vals with ref[i] > ref[i+1]
        for iidx in range(len(idx))[:-1]:
            if reference[axis][iidx] <= reference[axis][iidx+1]:
                idx[iidx] = False
        # only vals with alpha[i] > 0
        for iidx in enumerate(idx):
            if alpha[axis][iidx] < 0:
                idx[iidx] = False

        # only vals with p[i] > 0.2
        for iidx in enumerate(idx):
            if reference[axis][iidx] < 0.2:
                idx[iidx] = False

        alp_f = list(alpha[axis][idx]) + [0, 140]
        p_f = list(reference[axis][idx]) + [0, 1.2]

        if axis == 1:
            alp_f += [0, 0, 0, 0, 100, 100, 100]
            p_f += [0, 0, 0, 0, 1, 1, 1]
            p_f += [0]*100
            alp_f += [0]*100

        coef = np.polyfit(alp_f, p_f, deg)
        clb[version][axis] = list(coef)

        coef_s = ['%1.3e' % c for c in coef]
        print('Actuator %s:\t%s' % (axis, coef_s))
        poly = np.poly1d(coef)
        alp = np.linspace(-20, 150, 100)

        plt.figure('CLB'+version+'_'+str(axis))
        plt.plot(alpha[axis], reference[axis], ':',
                 label='measurements')
        plt.plot(alp_f, p_f, 'o', label='used measurements')
        plt.plot(alp, poly(alp), '-x', label='fitted')

        plt.grid()
        plt.xlim((-20, 150))
        plt.ylim((-.1, 1.3))
        plt.xlabel(r'bending angle $\alpha$ ($^\circ$)')
        plt.ylabel(r'pressure $p$ (bar)')
        plt.legend(loc='lower right')
#        save.save_plt_as_tikz('pics/'+version+'_clb_'+str(axis)+'.tex')

#        plt.figure(1)
#        plt.plot(pressure[axis], alpha[axis])


plt.show()

print(clb)
