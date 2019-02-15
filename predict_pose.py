#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:21:07 2019

@author: ls
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.animation as animation


f_l = 100.     # factor on length objective
f_o = 0.001  # .0003     # factor on orientation objective
f_a = 10     # factor on angle objective
len_leg = 1
len_tor = 1.1

blow = .9       # lower stretching bound
bup = 1.1       # upper stretching bound
dev_ang = 100  # allowed deviation of angles
arc_res = 40    # resolution of arcs
n_foot = 4
n_limbs = 5


def predict_pose(pattern, initial_pose, stats=False, debug=False,
                 f=[f_l, f_o, f_a], len_leg=len_leg, len_tor=len_tor,
                 dev_ang=dev_ang):
    f_len, f_ori, f_ang = f
    alpha, eps, F1 = initial_pose
    ell_nominal = (len_leg, len_leg, len_tor, len_leg, len_leg)
    xlast, rlast, markers = _set_initial_pose(alpha, ell_nominal, eps, F1)

    data, data_fp, data_nfp, data_x, costs, data_marks = [], [], [], [], [], []
    (x, y), (fpx, fpy), (nfpx, nfpy) = get_repr(xlast, rlast, (1, 0, 0, 0))
    data.append((x, y))
    data_fp.append((fpx, fpy))
    data_nfp.append((nfpx, nfpy))
    data_x.append(xlast)
    data_marks.append(markers)

    for idx, reference in enumerate(pattern):
        alpref_, f = reference
        alpref = []
        for jdx, alp in enumerate(alpref_):
            if jdx == 2:
                alpref.append(alp)
            else:
                alpref.append(alp if alp > 0 else 0.01)
        alplast, epslast = xlast[0:n_limbs], xlast[-1]
        philast = _calc_phi(alplast, epslast)

        # Initial guess
        x0 = xlast

        def objective(x):
            alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
            phi = _calc_phi(alp, eps)
            obj_ori, obj_len, obj_ang = 0, 0, 0
            for idx in range(n_foot):
                if f[idx]:
                    obj_ori = (obj_ori + (phi[idx] - philast[idx])**2)
            for idx in range(n_limbs):
                obj_ang = obj_ang + (alpref[idx]-alp[idx])**2
            for idx in range(n_limbs):
                obj_len = obj_len + (ell[idx]-ell_nominal[idx])**2
            objective = (f_ori*np.sqrt(obj_ori) + f_ang*np.sqrt(obj_ang) +
                         f_len*np.sqrt(obj_len))
            return objective

        def constraint1(x):
            """ feet should be at the right position """
            xf, yf, _, _ = _calc_coords(x, rlast, f)
            xflast, yflast = rlast
            constraint = 0
            for idx in range(n_foot):
                if f[idx]:
                    constraint = constraint + \
                        np.sqrt((xflast[idx] - xf[idx])**2 +
                                (yflast[idx] - yf[idx])**2)
            return constraint

        bleg = (blow*len_leg, bup*len_leg)
        btor = (blow*len_tor, bup*len_tor)
        bang = [(alpref[i]-dev_ang, alpref[i]+dev_ang) for i in range(n_limbs)]
        bnds = (bang[0], bang[1], bang[2], bang[3], bang[4],
                bleg, bleg, btor, bleg, bleg,
                (-360, 360))
        con1 = {'type': 'eq', 'fun': constraint1}
        cons = ([con1])
        solution = minimize(objective, x0, method='SLSQP',
                            bounds=bnds, constraints=cons)
        x = solution.x
        alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
        phi = _calc_phi(alp, eps)
        rx, ry, mx, my = _calc_coords(x, rlast, f)
        # save opt meta data
        xlast = x
        rlast = (rx, ry)
        if debug:
            print '\n\nPOSE ', idx, '\n'
            print 'constraint function: \t', round(constraint1(x), 2)
            print 'objective function: \t', round(objective(x), 2)
            print 'alpref: \t\t', [round(xx, 2) for xx in alpref]
            print 'alp: \t\t\t', [round(xx, 2) for xx in alp]
            print 'ell: \t\t\t', [round(xx, 2) for xx in ell], '\n'

            print 'rx: \t\t\t', [round(xx, 2) for xx in rx]
            print 'ry: \t\t\t', [round(xx, 2) for xx in ry]
            print 'phi: \t\t\t', [round(xx, 2) for xx in phi], '\n'

            print 'eps: \t\t\t', round(eps, 2)
        if stats:
            costs.append(objective(x))
            (xa, ya), (fpx, fpy), (nfpx, nfpy) = get_repr(x, (rx, ry), f)
            data.append((xa, ya))
            data_fp.append((fpx, fpy))
            data_nfp.append((nfpx, nfpy))
            data_x.append(x)
            data_marks.append((mx, my))

    return x, (rx, ry), (data, data_fp, data_nfp, data_x), costs, data_marks


def _set_initial_pose(alpha, ell, eps, F1):
    alp = alpha
    x = flat_list([alp, ell, [eps]])
    f = [1, 0, 0, 0]
    rinit = ([F1[0], None, None, None], [F1[1], None, None, None])
    rx, ry, mx, my = _calc_coords(x, rinit, f)
    return (x, (rx, ry), (mx, my))


def _calc_coords(x, r, f):
    alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
    c1, c2, _, _ = _calc_phi(alp, eps)
    R = [_calc_rad(ell[i], alp[i]) for i in range(5)]

    if f[0]:
        xf1, yf1 = r[0][0], r[1][0]
        # coords cp upper left leg
        xr1 = xf1 + np.cos(np.deg2rad(c1))*R[0]
        yr1 = yf1 + np.sin(np.deg2rad(c1))*R[0]
        # coords upper torso
        xom = xr1 - np.sin(np.deg2rad(90-c1-alp[0]))*R[0]
        yom = yr1 - np.cos(np.deg2rad(90-c1-alp[0]))*R[0]
        # coords cp R2
        xr2 = xom + np.cos(np.deg2rad(c1+alp[0]))*R[1]
        yr2 = yom + np.sin(np.deg2rad(c1+alp[0]))*R[1]
        # coords F2
        xf2 = xr2 + np.sin(np.deg2rad(alp[1] - (90-c1-alp[0])))*R[1]
        yf2 = yr2 - np.cos(np.deg2rad(alp[1] - (90-c1-alp[0])))*R[1]
    elif f[1]:
        xf2, yf2 = r[0][1], r[1][1]
        # coords cp upper right leg
        xr2 = xf2 - np.sin(np.deg2rad(c2-90))*R[1]
        yr2 = yf2 + np.cos(np.deg2rad(c2-90))*R[1]
        # coords upper torso
        xom = xr2 - np.sin(np.deg2rad(90-c2+alp[1]))*R[1]
        yom = yr2 - np.cos(np.deg2rad(90-c2+alp[1]))*R[1]
        # coords cp R1
        xr1 = xom + np.sin(np.deg2rad(90-c2+alp[1]))*R[0]
        yr1 = yom + np.cos(np.deg2rad(90-c2+alp[1]))*R[0]
        # coords F1
        xf1 = xr1 - np.sin(np.deg2rad(90-c2+alp[1]+alp[0]))*R[0]
        yf1 = yr1 - np.cos(np.deg2rad(90-c2+alp[1]+alp[0]))*R[0]
    # coords cp torso
    xrom = xom + np.cos(np.deg2rad(90-c1-alp[0]))*R[2]
    yrom = yom - np.sin(np.deg2rad(90-c1-alp[0]))*R[2]
    # coords lower torso
    xum = xrom - np.cos(np.deg2rad(alp[2] - (90-c1-alp[0])))*R[2]
    yum = yrom - np.sin(np.deg2rad(alp[2] - (90-c1-alp[0])))*R[2]
    # coords cp lower right foot
    xr4 = xum + np.sin(np.deg2rad(alp[2] - (90-c1-alp[0])))*R[4]
    yr4 = yum - np.cos(np.deg2rad(alp[2] - (90-c1-alp[0])))*R[4]
    # coords of F4
    xf4 = xr4 + np.sin(np.deg2rad(alp[4] - (alp[2] - (90-c1-alp[0]))))*R[4]
    yf4 = yr4 + np.cos(np.deg2rad(alp[4] - (alp[2] - (90-c1-alp[0]))))*R[4]
    # coords cp R3
    xr3 = xum + np.sin(np.deg2rad(alp[2] - (90-c1-alp[0])))*R[3]
    yr3 = yum - np.cos(np.deg2rad(alp[2] - (90-c1-alp[0])))*R[3]
    # coords of F3
    xf3 = xr3 - np.cos(np.deg2rad(-alp[3] - alp[2] + 180-c1-alp[0]))*R[3]
    yf3 = yr3 + np.sin(np.deg2rad(-alp[3] - alp[2] + 180-c1-alp[0]))*R[3]

    return ([xf1, xf2, xf3, xf4], [yf1, yf2, yf3, yf4],
            [xf1, xom, xf2, xf3, xum, xf4], [yf1, yom, yf2, yf3, yum, yf4])


def get_repr(x, r, f):
    alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
    c1, _, _, _ = _calc_phi(alp, eps)
    l1, l2, lg, l3, l4 = ell
    alp1, bet1, gam, alp2, bet2 = alp
    xf, yf = r

    x, y = [xf[0]], [yf[0]]
    # draw upper left leg
    xl1, yl1 = _calc_arc_coords((x[-1], y[-1]), c1, c1+alp1,
                                _calc_rad(l1, alp1))
    x = x + xl1
    y = y + yl1
    # draw torso
    xt, yt = _calc_arc_coords((x[-1], y[-1]), -90+c1+alp1, -90+c1+alp1+gam,
                              _calc_rad(lg, gam))
    x = x + xt
    y = y + yt
    # draw lower right leg
    xl4, yl4 = _calc_arc_coords((x[-1], y[-1]), -90+gam-(90-c1-alp1),
                                -90+gam-(90-c1-alp1)-bet2, _calc_rad(l4, bet2))
    x = x + xl4
    y = y + yl4
    # draw upper right leg
    xl2, yl2 = _calc_arc_coords((xl1[-1], yl1[-1]), c1+alp1,
                                c1+alp1+bet1, _calc_rad(l2, bet1))
    x = x + xl2
    y = y + yl2
    # draw lower left leg
    xl3, yl3 = _calc_arc_coords((xt[-1], yt[-1]), -90+gam-(90-c1-alp1),
                                -90+gam-(90-c1-alp1)+alp2, _calc_rad(l3, alp2))
    x = x + xl3
    y = y + yl3

    fp = ([], [])
    nfp = ([], [])
    for idx in range(n_foot):
        if f[idx]:
            fp[0].append(xf[idx])
            fp[1].append(yf[idx])
        else:
            nfp[0].append(xf[idx])
            nfp[1].append(yf[idx])

    return (x, y), fp, nfp


def flat_list(l):
    return [item for sublist in l for item in sublist]


def _calc_len(radius, angle):
    return angle/360.*2*np.pi*radius


def _calc_rad(length, angle):
    return 360.*length/(2*np.pi*angle)


def _calc_arc_coords(xy, alp1, alp2, rad):
    x0, y0 = xy
    x, y = [x0], [y0]
    xr = x0 + np.cos(np.deg2rad(alp1))*rad
    yr = y0 + np.sin(np.deg2rad(alp1))*rad
    steps = [angle for angle in np.linspace(0, alp2-alp1, arc_res)]
    for dangle in steps:
        x.append(xr - np.sin(np.deg2rad(90-alp1-dangle))*rad)
        y.append(yr - np.cos(np.deg2rad(90-alp1-dangle))*rad)

    return x, y


def calc_difference(phi0, phi1):
    theta = -np.radians(phi0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x2 = np.cos(np.radians(phi1))
    y2 = np.sin(np.radians(phi1))
    vec_ = R*np.c_[[x2, y2, 0]]
    diff = np.degrees(np.arctan2(float(vec_[1]), float(vec_[0])))
    return diff


def _calc_phi(alpha, eps):
    c1 = np.mod(eps - alpha[0] - alpha[2]*.5 + 360, 360)
    c2 = np.mod(c1 + alpha[0] + alpha[1] + 360, 360)
    c3 = np.mod(180 + alpha[2] - alpha[1] + alpha[3] + c2 + 360, 360)
    c4 = np.mod(180 + alpha[2] + alpha[0] - alpha[4] + c1 + 360, 360)
    phi = [c1, c2, c3, c4]
    return phi


def plot_gait(data_xy, data_fp, data_nfp, data_x):
    for idx in range(len(data_xy)):
        (x, y) = data_xy[idx]
        (fpx, fpy) = data_fp[idx]
        (nfpx, nfpy) = data_nfp[idx]
        c = (1-float(idx)/len(data_xy))*.8
        col = (c, c, c)
        plt.plot(x, y, '.', color=col)
        plt.plot(fpx, fpy, 'o', markersize=10, color=col)
        plt.plot(nfpx, nfpy, 'x', markersize=10, color=col)
    plt.axis('equal')


def marker_history(marks):
    markers = [([], []), ([], []), ([], []), ([], []), ([], []), ([], [])]
    for pose in range(len(marks)):
        x, y = marks[pose]
        for xm, ym, idx in zip(x, y, range(len(x))):
            markers[idx][0].append(xm)
            markers[idx][1].append(ym)
    return markers


def start_end(data_xy, data_fp, data_nfp, data_x):
    return ([data_xy[0], data_xy[-1]], [data_fp[0], data_fp[-1]],
            [data_nfp[0], data_nfp[-1]], [data_x[0], data_x[-1]])


def start_mid_end(data_xy, data_fp, data_nfp, data_x):
    mid = len(data_xy)/2
    return ([data_xy[0], data_xy[mid], data_xy[-1]],
            [data_fp[0], data_fp[mid], data_fp[-1]],
            [data_nfp[0], data_nfp[mid], data_nfp[-1]],
            [data_x[0], data_x[mid], data_x[-1]])


def animate_gait(fig1, data_xy, data_markers, inv=500,
                 col=['red', 'orange', 'green', 'blue', 'magenta', 'darkred']):

    def update_line(num, data_xy, line_xy, data_markers,
                    lm0, lm1, lm2, lm3, lm4, lm5, leps):
        x, y = data_xy[num]
        line_xy.set_data(np.array([[x], [y]]))

        xm0, ym0 = data_markers[num][0][0], data_markers[num][1][0]
        xm1, ym1 = data_markers[num][0][1], data_markers[num][1][1]
        xm2, ym2 = data_markers[num][0][2], data_markers[num][1][2]
        xm3, ym3 = data_markers[num][0][3], data_markers[num][1][3]
        xm4, ym4 = data_markers[num][0][4], data_markers[num][1][4]
        xm5, ym5 = data_markers[num][0][5], data_markers[num][1][5]
        lm0.set_data(np.array([[xm0], [ym0]]))
        lm1.set_data(np.array([[xm1], [ym1]]))
        lm2.set_data(np.array([[xm2], [ym2]]))
        lm3.set_data(np.array([[xm3], [ym3]]))
        lm4.set_data(np.array([[xm4], [ym4]]))
        lm5.set_data(np.array([[xm5], [ym5]]))
        leps.set_data(np.array([[xm1, xm4], [ym1, ym4]]))
        return line_xy, lm0, lm1, lm2, lm3, lm4, lm5, leps

    n = len(data_xy)
    l_xy, = plt.plot([], [], 'k.', markersize=3)
    msize = 5
    lm0, = plt.plot([], [], 'o', color=col[0], markersize=msize)
    lm1, = plt.plot([], [], 'o', color=col[1], markersize=msize)
    lm2, = plt.plot([], [], 'o', color=col[2], markersize=msize)
    lm3, = plt.plot([], [], 'o', color=col[3], markersize=msize)
    lm4, = plt.plot([], [], 'o', color=col[4], markersize=msize)
    lm5, = plt.plot([], [], 'o', color=col[5], markersize=msize)
    leps, = plt.plot([], [], '-', color='mediumpurple', linewidth=1)

    minx, maxx, miny, maxy = 0, 0, 0, 0
    for dataset in data_markers:
        x, y = dataset
        minx = min(x) if minx > min(x) else minx
        maxx = max(x) if maxx < max(x) else maxx
        miny = min(y) if miny > min(y) else miny
        maxy = max(y) if maxy < max(y) else maxy
    plt.xlim(minx-5, maxx+5)
    plt.ylim(miny-5, maxy+5)
    line_ani = animation.FuncAnimation(
        fig1, update_line, n, fargs=(data_xy, l_xy, data_markers,
                                     lm0, lm1, lm2, lm3, lm4, lm5, leps),
        interval=inv, blit=True)
    return line_ani


def save_animation(line_ani, name='gait.mp4', conv='avconv'):
    """
    To create gif:
        0. EASY: Use : https://ezgif.com/video-to-gif
        OR:
        1. Create a directory called frames in the same directory with
           your .mp4 file. Use command:
            ffmpeg -i video.mp4  -r 5 'frames/frame-%03d.jpg'

            -r 5 stands for FPS value
                for better quality choose bigger number
                adjust the value with the -delay in 2nd step
                to keep the same animation speed

            %03d gives sequential filename number in decimal form

        1a. Loop the thing (python):
            import os
            for jdx, idx in enumerate(range(1, 114)[::-1]):
                os.rename('frame-'+'{}'.format(idx).zfill(3)+'.jpg',
                          'frame-'+'{}'.format(114+jdx).zfill(3)+'.jpg')

        1b. Reduce size of singe frames (bash):
            for i in *.jpg; do convert "$i" -quality 20 "${i%%.jpg*}_new.jpg"; done

        2. Convert Images to gif (bash):
            cd frames
            convert -delay 20 -loop 0 *.jpg myimage.gif

            -delay 20 means the time between each frame is 0.2 seconds
               which match 5 fps above.
               When choosing this value
                   1 = 100 fps
                   2 = 50 fps
                   4 = 25 fps
                   5 = 20 fps
                   10 = 10 fps
                   20 = 5 fps
                   25 = 4 fps
                   50 = 2 fps
                   100 = 1 fps
                   in general 100/delay = fps

            -loop 0 means repeat forever
        2a. To further compress you can skip frames:
            gifsicle -U input.gif `seq -f "#%g" 0 2 99` -O2 -o output.gif
            
    """
    # Set up formatting for the movie files
    Writer = animation.writers[conv]
    writer = Writer(fps=15, metadata=dict(artist='Lars Schiller'),
                    bitrate=1800)
    line_ani.save(name, writer=writer)


if __name__ == "__main__":
    """
    To save the animation you need the libav-tool to be installed:
    sudo apt-get install libav-tools
    """

    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']


#   init_pose = [(alp)          , eps, pos foot1]
    init_pose = [(90, 1, -90, 90, 1), 0, (0, 0)]

    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 1, 0]]
           for gam in range(-90, 91, 45)]
    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 1]]
            for gam in range(-90, 90, 45)[::-1]]  # revers
    ref = ref + ref2

    x, r, data, cst, marks = predict_pose(ref, init_pose, True, False)

#    plt.figure()
#    plot_gait(*start_end(*data))
#
#    markers = marker_history(marks)
#    for idx, marker in enumerate(markers):
#        x, y = marker
#        plt.plot(x, y, color=col[idx])

    # ## withot stretching
    init_pose = [(90, 1, -90, 90, 1), 0, (0, 0)]
    step = 45
    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 0, 0]]
           for gam in range(-90, 91, step)]
    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 0]]
            for gam in range(-90, 90, step)[::-1]]  # revers
    ref = ref + ref2

    x, r, data, cst, marks = predict_pose(ref, init_pose, True, False,
                                          dev_ang=.1)

    plt.figure()
    plot_gait(*start_mid_end(*data))

    markers = marker_history(marks)
    for idx, marker in enumerate(markers):
        x, y = marker
        plt.plot(x, y, '-', color=col[idx])

    # Animation
    data_xy = data[0]
    fig_ani = plt.figure()
    plt.title('Test')
    _ = animate_gait(fig_ani, data_xy, marks)  # _ = --> important

    for idx, marker in enumerate(markers):
        x, y = marker
        plt.plot(x, y, '-', color=col[idx])

    plt.show()

