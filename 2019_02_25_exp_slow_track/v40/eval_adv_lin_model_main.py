#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:32:00 2019

@author: ls
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import eval as ev
import save
import kin_model
import predict_pose as pp

###############################################################################
# ################## SHIFT  in POSITION #######################################
###############################################################################
#%%
# exp data qualilty:

#          0  1  2  3
# big      x  x  x  x
# big      x  x  x  x

sets = ['{}'.format(idx).zfill(3) for idx in range(4)]
db, cyc = ev.load_data('2019_02_11_slow_bigbot/', sets)

# correction of jump epsilon
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

#%% ### eps during cycle
eps, sige = ev.calc_mean_of_axis(db, cyc, 'eps', [1])
t, sigt = ev.calc_mean_of_axis(db, cyc, 'time', [1])
# hack to remove high freq noise
for idx in range(1, len(t)):
    if abs(eps[idx] - eps[idx-1]) > 1:
        eps[idx] = eps[idx-1]
    if abs(sige[idx] - sige[idx-1]) > 1:
        sige[idx] = sige[idx-1]

eps = np.array(ev.downsample(list(eps)))
t = ev.downsample(t)
sige = np.array(ev.downsample(sige))
plt.figure('Epsilon corrected')
plt.plot(ev.rm_offset(t), eps, '-', color='mediumpurple')
plt.fill_between(ev.rm_offset(t), eps+sige, eps-sige,
                 facecolor='mediumpurple', alpha=0.5)

plt.grid()
plt.xlabel('time (s)')
plt.ylabel('orientation angle epsilon (deg)')
plt.ylim((-11, 10))

#save.save_as_tikz('pics/track/eps.tex')



#%% ### alpha during cycle
refleft = [90, 0, 0, 90, 90, 0, 0, 90, 90]
refright = [0, 90, 90, 0, 0, 90, 90, 0, 0]
reftorso = [-90, 90, 90, -90, -90, 90, 90, -90, -90]
reftime = np.array([0, .38, .5, .88, 1, 1.38, 1.5, 1.88, 2])*17  # cyctime

fig, ax = plt.subplots(nrows=3, ncols=1, num='Alpha during cycle', sharex=True)

t1, sigt = ev.calc_mean_of_axis(db, cyc, 'time', [1])
t2, sigt2 = ev.calc_mean_of_axis(db, cyc, 'time', [2])
t = np.concatenate((t1, t2), axis=None)
t = ev.rm_offset(t)

ALP = {}


col = ['red', 'darkred', 'orange', 'green', 'blue', 'darkblue']
for axis in [0, 1, 4, 5, 2]:
    alp1, sigalp1 = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [1])
    alp2, sigalp2 = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [2])
    alp = np.array(list(alp1)+list(alp2))
    sigalp = np.array(list(sigalp1)+list(sigalp2))

    # downsample for tikz
    prop = .1
    alp = np.array(ev.downsample(list(alp), prop))
    t_s = ev.downsample(t, prop)
    sigalp = np.array(ev.downsample(sigalp, prop))
    ALP[axis] = alp

    if axis in [0, 4]:
        axidx = 0
    elif axis in [1, 5]:
        axidx = 1
    else:
        axidx = 2
    ax[axidx].plot(t_s, alp, '-', color=col[axis])
    ax[axidx].fill_between(t_s, alp+sigalp, alp-sigalp, facecolor=col[axis], alpha=0.5)

ax[0].plot(reftime, refleft, ':', color='k')
ax[1].plot(reftime, refright, ':', color='k')
ax[2].plot(reftime, reftorso, ':', color='k')

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('left legs (deg)')
ax[1].set_ylabel('right legs (deg)')
ax[2].set_ylabel('torso (deg)')

ax[2].set_xlabel('time (s)')

save.save_as_tikz('pics/kin_model/alpha_analysis.tex')


# %% Left right Leg - Torso Relation

# find idx of first half cycle
for endidx, val in enumerate(t):
    if val > 8.5:
        print endidx
        break
fig, ax = plt.subplots(nrows=2, ncols=1, num='torso - leg relation', sharex=True)
idx = endidx
idx = len(t)-1
ax[0].plot(ALP[2][:idx], ALP[0][:idx], color=col[0])
ax[0].plot(ALP[2][:idx], ALP[4][:idx], color=col[4])

ax[1].plot(ALP[2][:idx], ALP[1][:idx], color=col[1])
ax[1].plot(ALP[2][:idx], ALP[5][:idx], color=col[5])

ax[0].grid()
ax[1].grid()

ax[0].set_ylabel('left legs (deg)')
ax[1].set_ylabel('right legs (deg)')
ax[1].set_xlabel('torso angle (deg)')

save.save_as_tikz('pics/kin_model/torso-lr-leg-relation.tex')


# %% Front Rear Leg Relation

# find idx of first half cycle
for endidx, val in enumerate(t_s):
    if val > 8.5:
        print endidx
        break
fig, ax = plt.subplots(nrows=1, ncols=1, num='rear leg relation', sharex=True)
idx = endidx
idx = len(t)-1

ax.plot(ALP[2], ALP[1], color=col[1])
ax.plot(-ALP[2], ALP[0], color=col[0])
ax.plot(ALP[2], -ALP[4]+105, color=col[4])
ax.plot(-ALP[2], -ALP[5]+105, color=col[5])

ax.grid()
ax.set_ylabel('legs (deg)')
ax.set_xlabel('torso (deg)')

# save.save_as_tikz('pics/kin_model/leg-relation.tex')

# %% Hystersis Analysis Front Leg

# find idx of first half cycle
for endidx, val in enumerate(t_s):
    if val > 8.5:
        print endidx
        break
fig, ax = plt.subplots(nrows=1, ncols=1, num='front leg hystersis')
idx = endidx

ax.plot(ALP[2][:idx], ALP[1][:idx], color=col[1])
ax.plot(ALP[2][idx:2*idx], ALP[1][idx:2*idx], ':', color=col[1])

dx = ALP[2][10] - ALP[2][0]
dy = ALP[1][10] - ALP[1][0]
ax.arrow(ALP[2][0], ALP[1][0]+10, dx, dy, head_width=5, color=col[1])

dx = ALP[2][idx+13] - ALP[2][idx+10]
dy = ALP[1][idx+13] - ALP[1][idx+10]
ax.arrow(ALP[2][idx+10], ALP[1][idx+10]-10, dx, dy, linestyle=':',
         head_width=5, color=col[1])


# front not fixed
N = 2
gam = ALP[2][:idx]
f_nfx = np.polyfit(ALP[2][:idx], ALP[1][:idx], N)
p_f_nfx = np.poly1d(f_nfx)
ax.plot(gam, p_f_nfx(gam), ':o')

# a1_nfx(gam) = 57.6 + .5*gam + .00082*gam**2


# front fixed
N = 3
gam = ALP[2][idx:2*idx]
f_fx = np.polyfit(gam, ALP[1][idx:2*idx], N)
p_f_fx = np.poly1d(f_fx)
ax.plot(gam, p_f_fx(gam), ':o')

# a1_fix(gam) = 37.8 + 0.052*gam + 0.0031*gam**2 + 0.00006*gam**3

ax.grid()
ax.set_ylabel('front leg (deg)')
ax.set_xlabel('torso (deg)')

save.save_as_tikz('pics/kin_model/front-leg-hysteresis.tex')

# %% Hystersis Analysis Rear Leg

# find idx of first half cycle
for endidx, val in enumerate(t_s):
    if val > 8.5:
        print endidx
        break
fig, ax = plt.subplots(nrows=1, ncols=1, num='rear leg hystersis')
idx = endidx

LEG = 4

ax.plot(ALP[2][:idx], ALP[LEG][:idx], color=col[LEG])
ax.plot(ALP[2][idx:2*idx], ALP[LEG][idx:2*idx], ':', color=col[LEG])

dx = ALP[2][10] - ALP[2][0]
dy = ALP[LEG][10] - ALP[LEG][0]
ax.arrow(ALP[2][0], ALP[LEG][0]-10, dx, dy, head_width=5, color=col[LEG])

dx = ALP[2][idx+13] - ALP[2][idx+10]
dy = ALP[LEG][idx+13] - ALP[LEG][idx+10]
ax.arrow(ALP[2][idx+10], ALP[LEG][idx+10]+10, dx, dy, linestyle=':',
         head_width=5, color=col[LEG])


# rear not fixed
N = 2
gam = range(-90, 90)
f_nfx = np.polyfit(ALP[2][:idx], ALP[LEG][:idx], N)
p_f_nfx = np.poly1d(f_nfx)
ax.plot(gam, p_f_nfx(gam), ':o')
print(f_nfx)
# a4_nfx(gam) = 37.3 - .47*gam + .002*gam**2

# rear fixed
N = 3
f_fx = np.polyfit(ALP[2][idx:2*idx], ALP[LEG][idx:2*idx], N)
p_f_fx = np.poly1d(f_fx)
gam = range(-90, 90)
ax.plot(gam, p_f_fx(gam), ':o')
print(f_fx)
# a4_fix(gam) = 59.5 - 0.077*gam - 0.00058*gam**2 - 0.000055*gam**3

ax.grid()
ax.set_ylabel('front leg (deg)')
ax.set_xlabel('torso (deg)')

save.save_as_tikz('pics/kin_model/rear-leg-hysteresis.tex')

# %% Kin Model definition


def front_foot_fix(gam):  # a1
    return 37.8 + 0.052*gam + 0.0031*gam**2 + 0.00006*gam**3


def front_foot_nfx(gam):  # a1
    return 57.6 + .5*gam + .00082*gam**2


def rear_foot_fix(gam):  # a4
    return 59.5 - 0.077*gam - 0.00058*gam**2 - 0.000055*gam**3


def rear_foot_nfx(gam):  # a4
    return 37.3 - .47*gam + .002*gam**2

# %% simulate model


col = ['red', 'orange', 'darkred', 'blue', 'green', 'magenta']

## Old model
#init_pose = [(90, 1, -90, 90, 1), 0, (-5, 0)]
#step = 45
#ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 0, 0]]
#       for gam in range(-90, 91, step)]
#ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 0]]
#        for gam in range(-90, 90, step)[::-1]]  # revers
#ref = ref + ref2
#
#x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
#                                         len_leg=13, len_tor=14,
#                                         dev_ang=.1)
##    pp.plot_gait(*pp.start_mid_end(*data))
#plt.figure('sim model')
#markers = pp.marker_history(marks)
#for idx, marker in enumerate(markers):
#    x, y = marker
#    plt.plot(x, y, '-', color=col[idx])


# New model
gam0 = -75
init_pose = [(front_foot_nfx(-gam0), front_foot_nfx(gam0), gam0,
              rear_foot_nfx(gam0), rear_foot_nfx(-gam0)), 0, (-5, 0)]
step = 2
ref = [[[front_foot_nfx(-gam), front_foot_fix(gam), gam,
         rear_foot_fix(gam), rear_foot_nfx(-gam)], [0, 1, 0, 0]]
       for gam in range(gam0, -gam0, step)]
ref2 = [[[front_foot_fix(-gam), front_foot_nfx(gam), gam,
          rear_foot_nfx(gam), rear_foot_fix(-gam)], [1, 0, 0, 0]]
        for gam in range(gam0, -gam0, step)[::-1]]  # revers
ref = ref + ref2

x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
                                         len_leg=13, len_tor=14,
                                         dev_ang=1)
## %% plot
# pp.plot_gait(*data)
fig, ax = plt.subplots(num='Simulate Model',
                           subplot_kw=dict(aspect='equal'))
markers = pp.marker_history(marks)
for idx, marker in enumerate(markers):
    x, y = marker
    plt.plot(x, y, ':', color=col[idx])

eps = []
for idx in range(len(marks)):
    x1, y1, x4, y4 = (marks[idx][0][1], marks[idx][1][1],
                      marks[idx][0][4], marks[idx][1][4])
    dx = x1 - x4
    dy = y1 - y4
    eps.append(np.rad2deg(np.arctan2(dy, dx)))

# %% Real Track
positions = [{}, {}]
alpha = {}
SIGXY = [{}, {}]
for axis in [0, 1, 2, 3, 4, 5]:
    x, sigx = ev.calc_mean_of_axis(db, cyc, 'x{}'.format(axis), [1])
    y, sigy = ev.calc_mean_of_axis(db, cyc, 'y{}'.format(axis), [1])
    a, siga = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [1])
    alpha[axis] = a
    positions[0][axis] = x
    positions[1][axis] = y
    SIGXY[0][axis] = sigx
    SIGXY[1][axis] = sigy
# %%

for idx in [0, len(ref)/4, len(ref)/2, 3*len(ref)/4, len(ref)-2]:
    fig, ax = plt.subplots(num='Track of feet {}'.format(idx),
                           subplot_kw=dict(aspect='equal'))
    for axis in range(6):
        x, y = positions[0][axis], positions[1][axis]
        sigx, sigy = SIGXY[0][axis], SIGXY[1][axis]
        # downsample for tikz
        prop = .1
        x, y = ev.downsample(x, prop), ev.downsample(y, prop)
        sigx, sigy = ev.downsample(sigx, prop), ev.downsample(sigy, prop)
        # plot xy
        plt.plot(x, y, color=col[axis], linewidth=20)

    # plot eps inclination
    x1, y1, x4, y4 = (marks[idx][0][1], marks[idx][1][1],
                      marks[idx][0][4], marks[idx][1][4])
    dx = x4 - x1
    dy = y4 - y1
    plt.plot([x1-dx, x4+dx], [y1-dy, y4+dy],
             '--', color='mediumpurple', linewidth=20)

    for jdx, marker in enumerate(markers):
        x, y = marker
        plt.plot(x, y, ':', color=col[jdx], linewidth=15)

    for jdx in range(6):
        plt.plot(marks[idx][0][jdx], marks[idx][1][jdx], 'o',
                 markersize=60, color=col[jdx])
    
    # draw selected Geckos poses
    
    
    alp = ref[idx+1][0]
    ell = [13, 13, 14, 13, 13]
    eps_ = eps[idx]
    gecko_tikz_str = save.tikz_draw_gecko(alp, ell, eps_,
                                          (marks[idx][0][0], marks[idx][1][0]),
                                          linewidth='2mm')
    plt.axis('off')
    save.save_as_tikz('pics/kin_model/track_{}.tex'.format(idx), gecko_tikz_str,
                      scale=.2)


# %% Epsilon of model
eps = []
for idx in range(len(marks)):
    x1, y1, x4, y4 = (marks[idx][0][1], marks[idx][1][1],
                      marks[idx][0][4], marks[idx][1][4])
    dx = x1 - x4
    dy = y1 - y4
    eps.append(np.rad2deg(np.arctan2(dy, dx)))
plt.plot(eps)

# %%

## ############################################# SINGLE SHOTS ##############
## ################# plain track
#for idx in [0, 160, 285, 434, 569]:
#    print(idx)
#    fig, ax = plt.subplots(num='Track of feet {}'.format(idx),
#                           subplot_kw=dict(aspect='equal'))
#    for axis in range(6):
#        x, y = positions[0][axis], positions[1][axis]
#        sigx, sigy = SIGXY[0][axis], SIGXY[1][axis]
#        # downsample for tikz
#        prop = .1
#        x, y = ev.downsample(x, prop), ev.downsample(y, prop)
#        sigx, sigy = ev.downsample(sigx, prop), ev.downsample(sigy, prop)
#        # plot xy
#        plt.plot(x, y, color=col[axis], linewidth=20)
#        # plot sigma xy
##            for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
##                el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
##                                 facecolor=col[axis], alpha=.3)
##                ax.add_artist(el)
#    # plot eps inclination
#    x1, y1, x4, y4 = (positions[0][1][idx], positions[1][1][idx],
#                      positions[0][4][idx], positions[1][4][idx])
#    dx = x4 - x1
#    dy = y4 - y1
#    plt.plot([x1-dx, x4+dx], [y1-dy, y4+dy],
#             '--', color='mediumpurple', linewidth=20)
#    # draw best fit gecko
#    pos = ([positions[0][axis][idx] for axis in range(6)],
#           [positions[1][axis][idx] for axis in range(6)])
#    alp = [alpha[axis][idx] for axis in range(6)]
#    alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
#    eps_ = eps[idx]
#
#    pose, marks, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
#    for jdx in range(6):
#        plt.plot(marks[0][jdx], marks[1][jdx], 'o',
#                 markersize=60, color=col[jdx])
#
#    gecko_tikz_str = save.tikz_draw_gecko(alp__, ell, eps_,
#                                          (marks[0][0], marks[1][0]),
#                                          linewidth='2mm')
#
##        plt.plot(pose[0], pose[1], '.', color='gray', markersize=1)
#    plt.axis('off')
##        save.save_as_tikz('pics/track/track_{}.tex'.format(idx), gecko_tikz_str,
##                          scale=.2)


plt.show()
