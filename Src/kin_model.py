#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:21:07 2019

@author: ls
"""
import numpy as np
from scipy.optimize import minimize


f_l = 100.      # factor on length objective
f_o = 0.1     # .0003     # factor on orientation objective
f_a = 10        # factor on angle objective

blow = .9       # lower stretching bound
bup = 1.1       # upper stretching bound
dev_ang = 100   # allowed deviation of angles

n_foot = 4
n_limbs = 5


def get_feet_pos(markers):
    mx, my = markers
    xf = [mx[0], mx[2], mx[3], mx[5]]
    yf = [my[0], my[2], my[3], my[5]]
    return (xf, yf)


def predict_next_pose(reference, x_init, markers_init,
                      f=[f_l, f_o, f_a],
                      len_leg=1, len_tor=1.2,
                      dev_ang=dev_ang, bounds=(blow, bup)):
    blow, bup = bounds
    f_len, f_ori, f_ang = f
    ell_nominal = (len_leg, len_leg, len_tor, len_leg, len_leg)

    alpref_, f = reference
    alpref = _check_alpha(alpref_)
    alp_init, eps_init = x_init[0:n_limbs], x_init[-1]
    phi_init = _calc_phi(alp_init, eps_init)

    # Initial guess
    x0 = x_init

    def objective(x):
        alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
        phi = _calc_phi(alp, eps)
        obj_ori, obj_len, obj_ang = 0, 0, 0
        for idx in range(n_foot):
            if f[idx]:
                # dphi = calc_difference(phi[idx], phi_init[idx])
                dphi = phi[idx] - phi_init[idx]
                obj_ori = (obj_ori + dphi**2)
        for idx in range(n_limbs):
            obj_ang = obj_ang + (alpref[idx]-alp[idx])**2
        for idx in range(n_limbs):
            obj_len = obj_len + (ell[idx]-ell_nominal[idx])**2
        objective = (f_ori*np.sqrt(obj_ori) + f_ang*np.sqrt(obj_ang) +
                     f_len*np.sqrt(obj_len))
        return objective

    def constraint1(x):
        """ feet should be at the right position """
        mx, my = _calc_coords(x, markers_init, f)
        xf, yf = get_feet_pos((mx, my))
        xflast, yflast = get_feet_pos(markers_init)
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
    mx, my = _calc_coords(x, markers_init, f)

    constraint = constraint1(x)
    cost = objective(x)

    return (x, (mx, my), f, constraint, cost)


# , # (px, py), (data, data_fp, data_nfp, data_x), costs, data_marks


def T0(eps, p0):
    eps = np.deg2rad(eps)
    c = np.cos(eps)
    s = np.sin(eps)
    return np.matrix([
        [c, -s, p0[0]],
        [s, c, p0[1]],
        [0, 0, 1]])


def T(alp, L):
    alp = np.deg2rad(alp)
    if alp == 0:
        alp = .001
    c = np.cos(alp)
    s = np.sin(alp)
    return np.matrix([
        [c, -s, L/alp*s],
        [s, c, L/alp*(1-c)],
        [0, 0, 1]])


def vec2lis(vec):
    return [float(vec[i]) for i in range(2)]


def _calc_coords2(x, marks, f):
    alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
    mx, my = marks
    if f[0]:
        p0 = (mx[0], my[0])
        T_Ori_0 = T0(eps-alp[2]/2-alp[0]-90, p0)
        T_01 = T(alp[0], ell[0])
        T_12 = T(alp[1], ell[1])
        p1 = vec2lis(T_Ori_0*T_01*np.c_[[0, 0, 1]])
        p2 = vec2lis(T_Ori_0*T_01*T_12*np.c_[[0, 0, 1]])
    elif f[1]:
        p2 = (mx[2], my[2])
        T_Ori_2 = T0(eps-alp[2]/2+alp[1]+90, p2)
        T_21 = T(-alp[1], ell[1])
        T_10 = T(-alp[0], ell[0])
        p1 = vec2lis(T_Ori_2*T_21*np.c_[[0, 0, 1]])
        p0 = vec2lis(T_Ori_2*T_21*T_10*np.c_[[0, 0, 1]])

    T_Ori_1 = T0(eps-alp[2]/2, p1)
    T_14 = T(180, 0)*T(alp[2], ell[2])
    T_45 = T(90, 0)*T(-alp[4], ell[4])
    T_43 = T(-90, 0)*T(alp[3], ell[3])

    p3 = vec2lis(T_Ori_1*T_14*T_43*np.c_[[0, 0, 1]])
    p4 = vec2lis(T_Ori_1*T_14*np.c_[[0, 0, 1]])
    p5 = vec2lis(T_Ori_1*T_14*T_45*np.c_[[0, 0, 1]])

    mx, my = zip(p0, p1, p2, p3, p4, p5)
    return mx, my


def _calc_coords(x, marks, f):
    alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
    c1, c2, c3, c4 = _calc_phi(alp, eps)
    R = [_calc_rad(ell[i], alp[i]) for i in range(5)]
    r = get_feet_pos(marks)
    if f[0] or f[1]:
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
    elif f[2] or f[3]:
        if f[2]:
            xf3, yf3 = r[0][2], r[1][2]
            # coords cp R3
            xr3 = xf3 + np.cos(np.deg2rad(c3))*R[3]
            yr3 = yf3 + np.sin(np.deg2rad(c3))*R[3]
            # coords UM
            xum = xr3 - np.cos(np.deg2rad(360-c3+alp[3]))*R[3]
            yum = yr3 + np.sin(np.deg2rad(360-c3+alp[3]))*R[3]
            # coords R4
            xr4 = xum + np.sin(np.deg2rad(c3-270-alp[3]))*R[4]
            yr4 = yum - np.cos(np.deg2rad(c3-270-alp[3]))*R[4]
            # coords F4
            xf4 = xr4 + np.cos(np.deg2rad(c4-180))*R[4]
            yf4 = yr4 + np.sin(np.deg2rad(c4-180))*R[4]
        else:
            raise(NotImplementedError)
        # coords cp torso
        xrom = xum + np.cos(np.deg2rad(c3-270-alp[3]))*R[2]
        yrom = yum + np.sin(np.deg2rad(c3-270-alp[3]))*R[2]
        # coords upper torso
        xom = xrom - np.cos(np.deg2rad(alp[2] - (c3-270-alp[3])))*R[2]
        yom = yrom + np.sin(np.deg2rad(alp[2] - (c3-270-alp[3])))*R[2]
        # coords cp R2
        xr2 = xom + np.cos(np.deg2rad(c1+alp[0]))*R[1]
        yr2 = yom + np.sin(np.deg2rad(c1+alp[0]))*R[1]
        # coords F2
        xf2 = xr2 + np.sin(np.deg2rad(alp[1] - (90-c1-alp[0])))*R[1]
        yf2 = yr2 - np.cos(np.deg2rad(alp[1] - (90-c1-alp[0])))*R[1]
        # coords cp R1
        xr1 = xom + np.sin(np.deg2rad(90-c2+alp[1]))*R[0]
        yr1 = yom + np.cos(np.deg2rad(90-c2+alp[1]))*R[0]
        # coords F1
        xf1 = xr1 - np.sin(np.deg2rad(90-c2+alp[1]+alp[0]))*R[0]
        yf1 = yr1 - np.cos(np.deg2rad(90-c2+alp[1]+alp[0]))*R[0]

    return [xf1, xom, xf2, xf3, xum, xf4], [yf1, yom, yf2, yf3, yum, yf4]


def flat_list(l):
    return [item for sublist in l for item in sublist]


def _calc_len(radius, angle):
    return angle/360.*2*np.pi*radius


def _calc_rad(length, angle):
    return 360.*length/(2*np.pi*angle)


def _calc_phi_obs(alpha, eps):
    c1 = np.mod(eps - alpha[0] - alpha[2]*.5 + 360, 360)
    c2 = np.mod(c1 + alpha[0] + alpha[1] + 360, 360)
    c3 = np.mod(180 + alpha[2] - alpha[1] + alpha[3] + c2 + 360, 360)
    c4 = np.mod(180 + alpha[2] + alpha[0] - alpha[4] + c1 + 360, 360)
#    c1 = eps - alpha[0] - alpha[2]*.5
#    c2 = c1 + alpha[0] + alpha[1]
#    c3 = 180 + alpha[2] - alpha[1] + alpha[3] + c2
#    c4 = 180 + alpha[2] + alpha[0] - alpha[4] + c1
    phi = [c1, c2, c3, c4]
    return phi


def _calc_phi(alpha, eps):
    c1 = eps - alpha[2]*.5 - alpha[0]
    c2 = eps - alpha[2]*.5 + alpha[1]
    c3 = eps + alpha[2]*.5 + alpha[3] + 180
    c4 = eps + alpha[2]*.5 - alpha[4] + 180
    phi = [c1, c2, c3, c4]
    return phi



def calc_difference(phi0, phi1):
    theta = -np.radians(phi0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x2 = np.cos(np.radians(phi1))
    y2 = np.sin(np.radians(phi1))
    vec_ = R*np.c_[[x2, y2, 0]]
    diff = np.degrees(np.arctan2(float(vec_[1]), float(vec_[0])))
    return diff


def _check_alpha(alpha):
    alpref = []
    for alp in alpha:
        alpref.append(alp if abs(alp) > 0 else 0.1)
    return alpref


def set_initial_pose(alp_, eps, p1, ell=None, len_leg=1, len_tor=1.2):
    alp = _check_alpha(alp_)
    if isinstance(ell, type(None)):
        ell = (len_leg, len_leg, len_tor, len_leg, len_leg)
    x = flat_list([alp, ell, [eps]])
    f = [1, 0, 0, 0]
    T_Ori_1 = T0(eps-alp[2]/2, p1)
    T_10 = T(90, 0)*T(-alp[0], ell[0])
    p0 = vec2lis(T_Ori_1*T_10*np.c_[[0, 0, 1]])
    marks_init = ([p0[0], None, None, None, None, None],
                  [p0[1], None, None, None, None, None])
    mx, my = _calc_coords(x, marks_init, f)
    # return (x, (mx, my), f)
    return x, (mx, my), f
