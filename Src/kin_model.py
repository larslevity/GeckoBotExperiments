# -*- coding: utf-8 -*-
"""
Created on Fri Feb 01 12:32:16 2019

@author: AmP
"""
import numpy as np
from scipy.optimize import minimize

len_leg = 13
len_tor = 14
ell_nom = [len_leg, len_leg, len_tor, len_leg, len_leg]


n_foot = 4
n_limbs = 5

arc_res = 40    # resolution of arcs

blow, bup = 0.8, 1.5
max_alp_dif = 20


def extract_pose(alpha, eps, fpos, len_leg=len_leg, len_tor=len_tor,
                 max_alp_dif=max_alp_dif):

    # unknown
    ell0 = [len_leg, len_leg, len_tor, len_leg, len_leg]
    alpha0 = alpha
    x0 = ell0 + alpha
    xpos, ypos = fpos[0], fpos[1]

    w1, w2, w3 = 10, .1, 1

    def objective(x):
        ell, alp = x[0:n_limbs], x[n_limbs:]
        xpos_est, ypos_est = _calc_coords2(x, alp, eps, (xpos[0], ypos[0]))
        err = [np.linalg.norm([xpos[idx]-xpos_est[idx],
                               ypos[idx]-ypos_est[idx]]) for idx in range(6)]
        len_dif = [abs(ell_nom[idx] - ell[idx]) for idx in range(n_limbs)]
        alp_dif = [abs(alpha0[idx] - alp[idx]) for idx in range(n_limbs)]
        obj = w1*sum(err) + w2*sum(alp_dif) + w3*sum(len_dif)

        return obj

    bleg = (blow*len_leg, bup*len_leg)
    btor = (blow*len_tor, bup*len_tor)
    bbet = [(val-max_alp_dif, val+max_alp_dif) for val in alpha0]
    bnds = (bleg, bleg, btor, bleg, bleg, bbet[0], bbet[1], bbet[2], bbet[3], bbet[4])
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
    x = solution.x
    ell, alp = x[0:n_limbs], x[n_limbs:]
    xpos_est, ypos_est = _calc_coords2(x, alp, eps, (xpos[0], ypos[0]))

    r = ([xpos[0]]+xpos[2:4]+[xpos[5]],
         [ypos[0]]+ypos[2:4]+[ypos[5]])

    x_ = list(alp) + list(ell) + [eps]
    (xa, ya), (fpx, fpy), (nfpx, nfpy) = get_repr(x_, r, [1, 1, 1, 1])

    return (xa, ya), (xpos_est, ypos_est), ell, alp


def _calc_coords2(x, alp, eps, (xf1, yf1)):
    ell, _ = x[0:n_limbs], x[n_limbs:]
    c1, c2, _, _ = _calc_phi(alp, eps)
    R = [_calc_rad(ell[i], alp[i]) for i in range(5)]

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

    return [xf1, xom, xf2, xf3, xum, xf4], [yf1, yom, yf2, yf3, yum, yf4]


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


def animate_gait(fig1, data, data_fp, data_nfp, data_x, inv=500):

    def update_line(num, data, line):
        x, y = data[num]
        line.set_data(np.array([[x], [y]]))
        return line

    n = len(data)
    l, = plt.plot([], [], '.')
    lfp, = plt.plot([], [], 'o', markersize=15)
    lnfp, = plt.plot([], [], 'x', markersize=10)
    minx, maxx, miny, maxy = 0, 0, 0, 0
    for dataset in data:
        x, y = dataset
        minx = min(x) if minx > min(x) else minx
        maxx = max(x) if maxx < max(x) else maxx
        miny = min(y) if miny > min(y) else miny
        maxy = max(y) if maxy < max(y) else maxy
    plt.xlim(minx-1, maxx+1)
    plt.ylim(miny-1, maxy+1)
    line_ani = animation.FuncAnimation(
        fig1, update_line, n, fargs=(data, l, data_fp, lfp, data_nfp, lnfp),
        interval=inv, blit=True)
    return line_ani

