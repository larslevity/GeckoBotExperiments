# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 18:22:04 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import eval as ev
import kin_model
import predict_pose as pp

###############################################################################
# ################## SHIFT  in POSITION #######################################
###############################################################################

# exp data qualilty:

#          0  1  2  3
# big      x  x  x  x
# big      x  x  x  x

sets = ['{}'.format(idx).zfill(3) for idx in range(4)]
db, cyc = ev.load_data('2019_02_11_slow_bigbot/', sets)

# correction of epsilon
for exp in range(len(db)):
    db[exp]['eps'] = ev.shift_jump(db[exp]['eps'], 180)

# correction of epsilon
rotate = 5
for exp in range(len(db)):
    eps0 = db[exp]['eps'][cyc[exp][1]]
    for marker in range(6):
        X = db[exp]['x{}'.format(marker)]
        Y = db[exp]['y{}'.format(marker)]
        X, Y = ev.rotate_xy(X, Y, -eps0+rotate)
        db[exp]['x{}'.format(marker)] = X
        db[exp]['y{}'.format(marker)] = Y
    db[exp]['eps'] = ev.add_offset(db[exp]['eps'], -eps0+rotate)


# check for quality:

for exp in range(len(db)):
    for idx in range(6):
        x = db[exp]['x{}'.format(idx)][cyc[exp][1]:cyc[exp][2]]
        y = db[exp]['y{}'.format(idx)][cyc[exp][1]:cyc[exp][2]]
        plt.figure('raw data XY')
        plt.plot(x, y)
    eps = db[exp]['eps'][cyc[exp][1]:cyc[exp][2]]
    t = db[exp]['time'][cyc[exp][1]:cyc[exp][2]]
    plt.figure('raw data Epsilon')
    plt.plot(t, eps)

# ### eps during cycle
eps, sige = ev.calc_mean_of_axis(db, cyc, 'eps', [1])
t, sigt = ev.calc_mean_of_axis(db, cyc, 'time', [1])
# hack
for idx in range(1, len(t)):
    if abs(eps[idx] - eps[idx-1]) > 1:
        eps[idx] = eps[idx-1]
    if abs(sige[idx] - sige[idx-1]) > 1:
        sige[idx] = sige[idx-1]

plt.figure('Epsilon corrected')
plt.plot(ev.rm_offset(t), eps, '-', color='mediumpurple')
plt.fill_between(ev.rm_offset(t), eps+sige, eps-sige,
                 facecolor='mediumpurple', alpha=0.5)

plt.grid()
plt.xlabel('time (s)')
plt.ylabel('orientation angle epsilon (deg)')
plt.ylim((-11, 10))




if 1:
    print('TRACK FEET')
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'),
                           num='Track of feet')
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    skip_first = 1
    skip_last = 1

    positions = [{}, {}]
    alpha = {}
    SIGXY = [{}, {}]

    for axis in [0, 1, 2, 3, 4, 5]:
        x, sigx = ev.calc_mean_of_axis(db, cyc, 'x{}'.format(axis), [1])
        y, sigy = ev.calc_mean_of_axis(db, cyc, 'y{}'.format(axis), [1])

        plt.plot(x, y, color=col[axis])
        for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
            el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
                             facecolor=col[axis], alpha=.3)
            ax.add_artist(el)

        a, siga = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [1])
        alpha[axis] = a
        positions[0][axis] = x
        positions[1][axis] = y
        SIGXY[0][axis] = sigx
        SIGXY[1][axis] = sigy

    # ####### plot eps inclination
    for x1, y1, x4, y4, i in zip(positions[0][1], positions[1][1],
                                 positions[0][4], positions[1][4],
                                 range(len(positions[0][4]))):
        if np.mod(i, 30) == 0:
            plt.plot([x1, x4], [y1, y4], '--k')

    # ####### plot gecko in first, mid and end position:
    for idx in [0, len(eps)/2, len(eps)-1]:
        pos = ([positions[0][axis][idx] for axis in range(6)],
               [positions[1][axis][idx] for axis in range(6)])
        alp = [alpha[axis][idx] for axis in range(6)]
        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
        eps_ = eps[idx]
        print(alp_)

        pose, marks, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
        plt.plot(pose[0], pose[1], '.', color='gray', markersize=1)
        for jdx in range(6):
            plt.plot(marks[0][jdx], marks[1][jdx], 'o', markersize=5, color=col[jdx])
        print 'idx: ', ell, [a_ - a__ for a_, a__ in zip(alp_, alp__)]

    # #### compare to model:
#    plt.figure(3)
#    plt.axis('equal')
#    init_pose = [(90, 1, -90, 90, 1), 0, (0, 0)]
#    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 1, 0]]
#           for gam in range(-90, 90, 10)]
#    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 1]]
#            for gam in range(-90, 90, 10)[::-1]]  # reverse
#    ref = ref + ref2
#
#    x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
#                                             len_leg=13, len_tor=14)
#
#    pp.plot_gait(*pp.start_end(*data))
#
#    markers = pp.marker_history(marks)
#    for idx, marker in enumerate(markers):
#        x, y = marker
#        plt.plot(x, y, color=col[idx])

    # ## withot stretching
    init_pose = [(90, 1, -90, 90, 1), 0, (-5, 0)]
    step = 45
    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 0, 0]]
           for gam in range(-90, 91, step)]
    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 0]]
            for gam in range(-90, 90, step)[::-1]]  # revers
    ref = ref + ref2

    x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
                                             len_leg=13, len_tor=14,
                                             dev_ang=.1)
#    pp.plot_gait(*pp.start_mid_end(*data))

    markers = pp.marker_history(marks)
    for idx, marker in enumerate(markers):
        x, y = marker
        plt.plot(x, y, '-', color=col[idx])



    # ############################################# ANIMATION ################
    # ################# plain track
    fig_ani, ax_ani = plt.subplots(num='Track of feet Animation',
                                   subplot_kw=dict(aspect='equal'))
    for axis in range(6):
        x, y = positions[0][axis], positions[1][axis]
        sigx, sigy = SIGXY[0][axis], SIGXY[1][axis]

        plt.plot(x, y, color=col[axis])
        for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
            el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
                             facecolor=col[axis], alpha=.3)
            ax_ani.add_artist(el)
    # collect animation data
    data_xy = []
    data_marks = []
    for idx in range(0, len(eps), 5):
        # ####### plot gecko guess:
        pos = ([positions[0][axis][idx] for axis in range(6)],
               [positions[1][axis][idx] for axis in range(6)])
        alp = [alpha[axis][idx] for axis in range(6)]
        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
        eps_ = eps[idx]
    
        pose, marks, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
        data_xy.append(pose)
        data_marks.append(marks)

    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    line_ani = pp.animate_gait(fig_ani, data_xy, data_marks)  # _ = --> important
    pp.save_animation(line_ani, name='track_of_feet.mp4', conv='avconv')
    

plt.show()
