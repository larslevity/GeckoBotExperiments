# -*- coding: utf-8 -*-
"""
Created on Thu Feb 07 15:18:02 2019

@author: AmP
"""
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import eval as ev
import kin_model


###############################################################################
# ################## SHIFT  in POSITION #######################################
###############################################################################

# exp data qualilty:

#          0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22
# big      x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x
# big      x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x

sets = ['{}'.format(idx).zfill(3) for idx in range(23)]
db, cyc = ev.load_data('2019_02_07_big_initial_cycle_exp/', sets)


# check for quality:
for exp in range(len(db)):
    for idx in range(6):
        x = db[exp]['x{}'.format(idx)][cyc[exp][1]:cyc[exp][2]]
        y = db[exp]['y{}'.format(idx)][cyc[exp][1]:cyc[exp][2]]
        plt.plot(x, y)
plt.show()


if 1:
    for exp in range(23):
#        t1 = ev.rm_offset(db[13]['time'][cyc[13][0]:])
#        r1 = db[13]['r1'][cyc[13][0]:]
        t2 = ev.rm_offset(db[exp]['time'][cyc[exp][0]:])
        r2 = db[exp]['r1'][cyc[exp][0]:]

#        plt.plot(t1, r1)
#        plt.plot(t2, r2)
#        plt.show()
#
#        plt.plot(r1)
#        plt.plot(r2)
#        plt.show()
#
#        plt.plot(t1, '.')
#        plt.plot(t2, '.')
#        plt.show()

#        plt.plot(np.diff(t1))
        plt.plot(np.diff(t2))
    plt.show()


if 1:
    print('TRACK FEET')
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'))
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    skip_first = 1
    skip_last = 1

    positions = [{}, {}]
    alpha = {}

    for axis in [0,1,2,3,4,5]:
        x, sigx = ev.calc_mean_of_axis(db, cyc, 'x{}'.format(axis), [1])
        y, sigy = ev.calc_mean_of_axis(db, cyc, 'y{}'.format(axis), [1])
        a, siga = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [1])

        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)

        alpha[axis] = a
        positions[0][axis] = x
        positions[1][axis] = y

        for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
            el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
                             facecolor=col[axis], alpha=.3)
            ax.add_artist(el)

    eps, sige = ev.calc_mean_of_axis(db, cyc, 'eps', [1])

    # ####### plot gecko in first, mid and end position:
    for jdx, idx in enumerate([0, len(eps)-1]):
        pos = ([positions[0][axis][idx] for axis in range(6)],
               [-positions[1][axis][idx] for axis in range(6)])  # mind minus
        alp = [alpha[axis][idx] for axis in range(6)]
        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
        eps_ = 3   # cheat

        pose, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
        pose = (pose[0], [-val for val in pose[1]])         # flip again
        plt.plot(pose[0], pose[1], '.', color=col[jdx])
        print 'idx: ', ell, [a_ - a__ for a_, a__ in zip(alp_, alp__)]

    plt.show()

if 0:
    plt.figure()
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    skip_first = 1
    skip_last = 1

    positions = [{}, {}]
    alpha = {}

    for axis in [0]:
        x, sigx = ev.calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc, 'x{}'.format(axis), (skip_first, skip_last))
        y, sigy = ev.calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc, 'y{}'.format(axis), (skip_first, skip_last))
        a, siga = ev.calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc, 'aIMG{}'.format(axis), (skip_first, skip_last))

        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)
#        plt.figure(1)
#        plt.plot(a, color=col[axis])
#        plt.fill_between(range(len(a)), a+siga, a-siga, facecolor=col[axis], alpha=0.5)

        alpha[axis] = a
        positions[0][axis] = x
        positions[1][axis] = y
    eps, sige = ev.calc_mean_of_axis_for_all_exp_and_cycles(
            db, cyc, 'eps', (skip_first, skip_last))

#    # ####### plot gecko in first, mid and end position:
#    for jdx, idx in enumerate([0, len(eps)/2, len(eps)-1]):
#        pos = ([positions[0][axis][idx] for axis in range(6)],
#               [-positions[1][axis][idx] for axis in range(6)])  # mind minus
#        alp = [alpha[axis][idx] for axis in range(6)]
#        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
#        eps_ = 3   # cheat
#
#        pose, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
#        pose = (pose[0], [-val for val in pose[1]])         # flip again
#        plt.plot(pose[0], pose[1], '.', color=col[jdx])
#        print 'idx: ', ell, [a_ - a__ for a_, a__ in zip(alp_, alp__)]

    plt.show()


