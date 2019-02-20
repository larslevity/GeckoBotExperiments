# -*- coding: utf-8 -*-
"""
Created on Thu Feb 07 15:15:37 2019

@author: AmP
"""
from eval import *

###############################################################################
# ################## SHIFT  in POSITION #######################################
###############################################################################

# exp 0 data qualilty:

#           0 1 2 3 4 5 6 7 8 9 10 11 12
# small     1 0 1 1 1 1 x 1 1 1 x  x  x
# big       1 1 0 0 1 1 1 1 1 0 0  1  1

ggg = 1
sets = ['{}'.format(idx).zfill(2) for idx in [0, 2, 3, 4, 5, 7, 8, 9]]
ds, cyc_small = load_data('small_0_', sets)

# sets = ['{}'.format(idx).zfill(2) for idx in [0,1,4,5,6,7,8,11,12]]
# db, cyc_big = load_data('big_0', sets)

sets = ['{}'.format(idx).zfill(2) for idx in [0, 1, 4, 5, 6, 7, 8, 11]]
db, cyc_big = load_data('exp190201/big_1_', sets)


color_prs = 'darkslategray'
color_ref = 'lightcoral'
color_alp = 'red'


if 0:

    plt.figure()
    # # small
    centers, t = calc_centerpoint(ds, cyc_small)
    mat = make_matrix_plain(centers)
    mu, sigma = calc_mean_stddev(mat)
    plt.plot(t, mu, '-', lw=2, label='p_{v0}', color=color_prs)
    plt.fill_between(t, mu+sigma, mu-sigma, facecolor=color_prs, alpha=0.2)

    # # big
    centers, t = calc_centerpoint(db, cyc_big)
    mat = make_matrix_plain(centers)
    mu, sigma = calc_mean_stddev(mat)
    plt.plot(t, mu, '-', lw=2, label='p_{v0}', color=color_alp)
    plt.fill_between(t, mu+sigma, mu-sigma, facecolor=color_alp, alpha=0.2)

    plt.xlabel('time (s)')
    plt.ylabel('$\bar{x}$ (cm)')
    plt.grid()

    save.save_as_tikz('pics/Shift.tex')

    if SAVE:
        plt.savefig('pics/Shift.png', dpi=500, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)


###############################################################################
# TRACK OF FEET ---- ALL CYCLE ANALYSE ########################################
###############################################################################

def flat_list(l):
    return [item for sublist in l for item in sublist]


if 1:
    plt.figure()
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    for axis in [4, 0, 1, 3, 2, 5]:
        x, sigx = calc_mean_of_axis_in_exp_and_cycle(db, cyc_big, axis='x{}'.format(axis))
        y, sigy = calc_mean_of_axis_in_exp_and_cycle(db, cyc_big, axis='y{}'.format(axis))
        xxx = x
        x = flat_list(x)
        y = (np.array(flat_list(y)) + yshift)
        sigy = np.array(flat_list(sigy))
        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)

        # Small
        x, sigx = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='x{}'.format(axis))
        y, sigy = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='y{}'.format(axis))
        x = flat_list(x)
        y = (np.array(flat_list(y)))
        sigy = np.array(flat_list(sigy))
        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)

    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axis('equal')


plt.show()





###############################################################################
# TRACK OF FEET ---- ALL CYCLE ANALYSE ---- WITHOUT SYNCHRONISATION ###########
###############################################################################


if 1:
    plt.figure()
    plt.title('Track')

    axes = range(6)

    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    X, Y, Xstd, Ystd = calc_foot_mean_of_all_exp(ds, cyc_small)
    for idx in axes:
        x = np.array(X[idx])
        y = np.array(Y[idx])
        sigx = np.array(Xstd[idx])
        sigy = np.array(Ystd[idx])
        plt.plot(x, y, color=col[idx])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[idx], alpha=0.6)

    yshift = -50
    X, Y, Xstd, Ystd = calc_foot_mean_of_all_exp(db, cyc_big)
    for idx in axes:
        x = np.array(X[idx])
        y = (np.array(Y[idx]) + yshift)
        sigx = np.array(Xstd[idx])
        sigy = np.array(Ystd[idx])
        plt.plot(x, y, color=col[idx])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[idx], alpha=0.5, label='mark_{}'.format(idx))

    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axis('equal')


#    save.save_as_tikz('pics/Track.tex')

    if SAVE:
        plt.savefig('pics/Track.png', dpi=500, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)



###############################################################################
# TRACK OF FEET ---- SINGLE CYCLE ANALYSE #####################################
###############################################################################




if 1:
    plt.figure()
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    skip_first = 1
    skip_last = 1

    positions = [{}, {}]
    alpha = {}

    for axis in [0, 1, 2, 3, 4, 5]:
        x, sigx = calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc_big, 'x{}'.format(axis), (skip_first, skip_last))
        y, sigy = calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc_big, 'y{}'.format(axis), (skip_first, skip_last))
        a, siga = calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc_big, 'aIMG{}'.format(axis), (skip_first, skip_last))

        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)

        alpha[axis] = a
        positions[0][axis] = x
        positions[1][axis] = y
    eps, sige = calc_mean_of_axis_for_all_exp_and_cycles(
            db, cyc_big, 'eps', (skip_first, skip_last))

    # ####### plot gecko in first, mid and end position:
    for jdx, idx in enumerate([0, len(eps)/2, len(eps)-1]):
        pos = ([positions[0][axis][idx] for axis in range(6)],
               [-positions[1][axis][idx] for axis in range(6)])  # mind minus
        alp = [alpha[axis][idx] for axis in range(6)]
        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
        eps_ = 3   # cheat

        pose, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
        pose = (pose[0], [-val for val in pose[1]])         # flip again
        plt.plot(pose[0], pose[1], '.', color=col[jdx])
        print 'idx: ', ell, [a_ - a__ for a_, a__ in zip(alp_, alp__)]

#        # Small
#        x, sigx = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='x{}'.format(axis))
#        y, sigy = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='y{}'.format(axis))
#        x = flat_list(x)
#        y = (np.array(flat_list(y)))
#        sigy = np.array(flat_list(sigy))
#        plt.plot(x, y, color=col[axis])
#        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)







    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axis('equal')
    plt.grid()


plt.show()


###############################################################################
# ########################## TRACK MEAN #######################################
###############################################################################

if 0:
    plt.figure()
    plt.title('Track Mean')
    smapleval = .8


    # Big
    yshift = -20
    X, _ = calc_centerpoint(db, cyc_big, axis='x')
    mat = make_matrix_plain(X)
    x, sigx = calc_mean_stddev(mat)

    Y, _ = calc_centerpoint(db, cyc_big, axis='y')
    mat = make_matrix_plain(Y)
    y, sigy = calc_mean_stddev(mat)
    y = (y + yshift)

    x = np.array(downsample(x, smapleval))
    y = np.array(downsample(y, smapleval))
    sigy = np.array(downsample(sigy, smapleval))

    plt.plot(x, y, color=col[idx])
    plt.fill_between(x, y+sigy, y-sigy, facecolor=color_alp, alpha=0.5, label='mean_big')

    # Mean Small
    yshift = 0
    X, _ = calc_centerpoint(ds, cyc_small, axis='x')
    mat = make_matrix_plain(X)
    x, sigx = calc_mean_stddev(mat)

    Y, _ = calc_centerpoint(ds, cyc_small, axis='y')
    mat = make_matrix_plain(Y)
    y, sigy = calc_mean_stddev(mat)
    y = (y + yshift)

    x = np.array(downsample(x, smapleval))
    y = np.array(downsample(y, smapleval))
    sigy = np.array(downsample(sigy, smapleval))

    plt.plot(x, y, color=col[idx])
    plt.fill_between(x, y+sigy, y-sigy, facecolor=color_prs, alpha=0.5, label='mean_small')

    plt.axis('equal')
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.grid()

    save.save_as_tikz('pics/Track_Mean.tex')

    if SAVE:
        save.save_as_tikz('pics/Track_Mean.png')
        plt.savefig('pics/Track_Mean.png', dpi=500, facecolor='w',
                    edgecolor='w', orientation='portrait', papertype=None,
                    format=None, transparent=False, bbox_inches=None,
                    pad_inches=0.1, frameon=None, metadata=None)



