# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:45:50 2019

@author: AmP
"""


import numpy as np
import matplotlib.pyplot as plt
from os import path
import os

from Src import kin_model as model
from Src import save as mysave

n_limbs = 5
n_foot = 4
arc_res = 40    # resolution of arcs


class GeckoBotPose(object):
    def __init__(self, x, marks, f, constraint=0, cost=0,
                 len_leg=1, len_tor=1.2, fpos_real=None):
        self.x = x
        self.markers = marks
        self.f = f
        self.constraint = constraint
        self.cost = cost
        self.alp = self.x[0:n_limbs]
        self.ell = self.x[n_limbs:2*n_limbs]
        self.eps = self.x[-1]
        self.fpos_real = fpos_real

    def get_eps(self):
        return self.x[-1]

    def get_m1_pos(self):
        mx, my = self.markers
        return (mx[1], my[1])

    def plot(self, col='k', ax=None):
        (x, y), (fpx, fpy), (nfpx, nfpy) = \
            get_point_repr(self.x, self.markers, self.f)
        if ax:
            ax.plot(x, y, '.', color=col)
            ax.plot(fpx, fpy, 'o', markersize=10, color=col)
            ax.plot(nfpx, nfpy, 'x', markersize=10, color=col)
        else:
            plt.plot(x, y, '.', color=col)
            plt.plot(fpx, fpy, 'o', markersize=10, color=col)
            plt.plot(nfpx, nfpy, 'x', markersize=10, color=col)

    def get_tikz_repr(self, col='black', shift=None):
        alp, ell, eps = (self.x[0:n_limbs], self.x[n_limbs:2*n_limbs],
                         self.x[-1])
        mx, my = self.markers
        if shift:
            geckostring = '\\begin{scope}[xshift=%scm]' % str(shift)
        else:
            geckostring = ''
        geckostring += tikz_draw_gecko(
                alp, ell, eps, (mx[0], my[0]), fix=self.f, col=col,
                linewidth='1mm')
        if shift:
            geckostring += '\\end{scope}\n \n \n'
        else:
            geckostring += '\n\n'
        return geckostring

    def save_as_tikz(self, filename, compileit=True):
        direc = path.dirname(path.dirname(path.dirname(
                    path.abspath(__file__)))) + '/Out/'
        gstr = self.get_tikz_repr()
        name = direc+filename+'.tex'
        mysave.save_geckostr_as_tikz(name, gstr)
        if compileit:
            out_dir = os.path.dirname(name)
            print(name)
            os.system('pdflatex -output-directory {} {}'.format(out_dir, name))
            print('Done')

    def get_phi(self):
        alp, eps = (self.x[0:n_limbs], self.x[-1])
        phi = model._calc_phi(alp, eps)
        return phi

    def get_alpha(self):
        return self.x[0:n_limbs]

    def plot_markers(self, col=None, markernum=range(6)):
        """plots the history of markers in *markernum*"""
        if type(markernum) == int:
            markernum = [markernum]
        mx, my = self.markers
        if not col:
            col = markers_color()
        elif type(col) == str:
            col = [col]*6
        for idx, (x, y) in enumerate(zip(mx, my)):
            plt.plot(x, y, 'd', color=col[idx])

    def plot_real_markers(self, col=None, markernum=range(6), ax=None):
        """plots the history of markers in *markernum*"""
        if type(markernum) == int:
            markernum = [markernum]
        try:
            mx, my = self.fpos_real
            if not col:
                col = markers_color()
            else:
                col = [col]*6
            for idx, (x, y) in enumerate(zip(mx, my)):
                if ax:
                    ax.plot(x, y, 'd', color=col[idx])
                else:
                    plt.plot(x, y, 'd', color=col[idx])
        except TypeError:
            pass


class GeckoBotGait(object):
    def __init__(self, initial_pose=None):
        self.poses = []
        if initial_pose:
            self.append_pose(initial_pose)

    def append_pose(self, pose):
        self.poses.append(pose)

    def plot_gait(self, fignum='', figname='GeckoBotGait', g=0, ax=None):
        if fignum:
            plt.figure(figname+fignum)
        for idx, pose in enumerate(self.poses):
            c = (1-float(idx)/len(self.poses))*.8
            col = (c, c, g)
            pose.plot(col, ax=ax)
            pose.plot_real_markers(col, ax=ax)

    def get_tikz_repr(self, shift=None):
        gait_str = ''
        for idx, pose in enumerate(self.poses):
            c = int(20 + (float(idx)/len(self.poses))*80.)
            col = 'black!{}'.format(c)
            shift_ = idx*shift if shift else None
            gait_str += pose.get_tikz_repr(col, shift_)
        return gait_str

    def plot_markers(self, markernum=range(6), figname='GeckoBotGait'):
        """plots the history of markers in *markernum*"""
        if type(markernum) == int:
            markernum = [markernum]
        plt.figure(figname)
        marks = [pose.markers for pose in self.poses]
        markers = marker_history(marks)
        col = markers_color()
        for idx, marker in enumerate(markers):
            if idx in markernum:
                x, y = marker
                plt.plot(x, y, color=col[idx])

    def plot_com(self, markernum=range(6)):
        """plots the history of center of markers in *markernum*"""
        if type(markernum) == int:
            markernum = [markernum]
        plt.figure('GeckoBotGait')
        marks = [pose.markers for pose in self.poses]
        markers = marker_history(marks)
        x = np.zeros(len(markers[0][0]))
        y = np.zeros(len(markers[0][0]))
        for idx, marker in enumerate(markers):
            if idx in markernum:
                xi, yi = marker
                x = x + np.r_[xi]
                y = y + np.r_[yi]
        x = x/len(markernum)
        y = y/len(markernum)
        plt.plot(x, y, color='purple')

    def plot_markers2(self, markernum=range(6)):
        """
        plots every value of markers in *markernum*
        if torso is bent positive"""
        if type(markernum) == int:
            markernum = [markernum]
        plt.figure('GeckoBotGait')
        marks = []
        for pose in self.poses:
            if pose.x[2] > 0:
                marks.append(pose.markers)
        markers = marker_history(marks)
        col = markers_color()
        for idx, marker in enumerate(markers):
            if idx in markernum:
                x, y = marker
                plt.plot(x, y, color=col[idx])

    def get_travel_distance(self):
        last = self.poses[-1].get_m1_pos()
        start = self.poses[0].get_m1_pos()
        dist = (last[0]-start[0], last[-1]-start[1])
        deps = self.poses[-1].get_eps() - self.poses[0].get_eps()
        return dist, deps

    def plot_travel_distance(self):
        plt.figure('GeckoBotGait')
        dist, deps = self.get_travel_distance()
        start = self.poses[0].get_m1_pos()
        plt.plot([start[0], start[0]+dist[0]], [start[1], start[1]+dist[1]])

    def plot_orientation(self, length=.5, poses=[0, -1]):
        plt.figure('GeckoBotGait')
        for pose in poses:
            start = self.poses[pose].get_m1_pos()
            eps = self.poses[pose].get_eps()
            plt.plot([start[0], start[0]+np.cos(np.deg2rad(eps))*length],
                     [start[1], start[1]+np.sin(np.deg2rad(eps))*length], 'r')

    def plot_epsilon(self):
        plt.figure('GeckoBotGaitEpsHistory')
        Eps = []
        for pose in self.poses:
            eps = pose.get_eps()
            Eps.append(eps)
        plt.plot(Eps, 'purple')
        return Eps

    def plot_phi(self):
        Phi = [[], [], [], []]
        for pose in self.poses:
            phi = pose.get_phi()
            for idx, phii in enumerate(phi):
                Phi[idx].append(phii)
        plt.figure('GeckoBotGaitPhiHistory')
        col = markers_color()
        for idx, phi in enumerate(Phi):
            plt.plot(phi, color=col[idx])
        plt.legend(['0', '1', '2', '3'])
        return Phi

    def plot_alpha(self):
        Alp = [[], [], [], [], []]
        for pose in self.poses:
            alp = pose.get_alpha()
            for idx, alpi in enumerate(alp):
                Alp[idx].append(alpi)
        plt.figure('GeckoBotGaitAlphaHistory')
        col = markers_color()
        for idx, alp in enumerate(Alp):
            plt.plot(alp, color=col[idx])
        plt.legend(['0', '1', '2', '3', '4'])
        return Alp

    def plot_stress(self, fignum=''):
        plt.figure('GeckoBotGaitStress'+fignum)
        stress = [pose.cost for pose in self.poses]
        plt.plot(stress)
        return sum(stress)

    def save_as_tikz(self, filename):
        direc = path.dirname(path.dirname(path.dirname(
                    path.abspath(__file__)))) + '/tikz/'
        name = direc+filename+'.tex'
        out_dir = os.path.dirname(name)
        gstr = ''
        for idx, pose in enumerate(self.poses):
            if len(self.poses) == 1:
                col = 'black'
            else:
                c = 50 + int((float(idx)/(len(self.poses)-1))*50)
                col = 'black!{}'.format(c)
            gstr += pose.get_tikz_repr(col)
        mysave.save_geckostr_as_tikz(name, gstr)
        os.system('pdflatex -output-directory {} {}'.format(out_dir, name))
        print('Done')


def predict_gait(references, initial_pose, weight=None):
    len_leg = initial_pose.len_leg
    len_tor = initial_pose.len_tor
    if not weight:
        weight = [model.f_l, model.f_o, model.f_a]

    gait = GeckoBotGait(initial_pose)
    for idx, ref in enumerate(references):
        x, (mx, my), f, constraint, cost = model.predict_next_pose(
                ref, gait.poses[idx].x, gait.poses[idx].markers,
                f=weight, len_leg=len_leg, len_tor=len_tor)
        gait.append_pose(GeckoBotPose(
                x, (mx, my), f, constraint=constraint,
                cost=cost))
    return gait


def markers_color():
    return ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']


def get_point_repr(x, marks, f):
    alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
    c1, _, _, _ = model._calc_phi(alp, eps)
    l1, l2, lg, l3, l4 = ell
    alp1, bet1, gam, alp2, bet2 = alp
    xf, yf = model.get_feet_pos(marks)

    x, y = [xf[0]], [yf[0]]
    # draw upper left leg
    xl1, yl1 = _calc_arc_coords((x[-1], y[-1]), c1, c1+alp1,
                                model._calc_rad(l1, alp1))
    x = x + xl1
    y = y + yl1
    # draw torso
    xt, yt = _calc_arc_coords((x[-1], y[-1]), -90+c1+alp1, -90+c1+alp1+gam,
                              model._calc_rad(lg, gam))
    x = x + xt
    y = y + yt
    # draw lower right leg
    xl4, yl4 = _calc_arc_coords((x[-1], y[-1]), -90+gam-(90-c1-alp1),
                                -90+gam-(90-c1-alp1)-bet2,
                                model._calc_rad(l4, bet2))
    x = x + xl4
    y = y + yl4
    # draw upper right leg
    xl2, yl2 = _calc_arc_coords((xl1[-1], yl1[-1]), c1+alp1,
                                c1+alp1+bet1, model._calc_rad(l2, bet1))
    x = x + xl2
    y = y + yl2
    # draw lower left leg
    xl3, yl3 = _calc_arc_coords((xt[-1], yt[-1]), -90+gam-(90-c1-alp1),
                                -90+gam-(90-c1-alp1)+alp2,
                                model._calc_rad(l3, alp2))
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


def marker_history(marks):
    """ formats the marks from predictpose to:
        marks[marker_idx][x/y][pose_idx]
    """
    markers = [([], []), ([], []), ([], []), ([], []), ([], []), ([], [])]
    for pose in range(len(marks)):
        x, y = marks[pose]
        for xm, ym, idx in zip(x, y, range(len(x))):
            markers[idx][0].append(xm)
            markers[idx][1].append(ym)
    return markers


def extract_eps(data):
    (data_, data_fp, data_nfp, data_x) = data
    eps = []
    for pose_idx in range(len(data_x)):
        eps.append(data_x[pose_idx][-1])
    return eps


def extract_ell(data):
    (data_, data_fp, data_nfp, data_x) = data
    ell = []
    for pose_idx in range(len(data_x)):
        ell.append(data_x[pose_idx][n_limbs:2*n_limbs])
    return ell


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


def tikz_draw_gecko(alp, ell, eps, F1, col='black',
                    linewidth='.5mm', fix=None):
    c1, c2, c3, c4 = model._calc_phi(alp, eps)
    l1, l2, lg, l3, l4 = ell
    for idx, a in enumerate(alp):
        if abs(a) < 2:
            alp[idx] = 2 * a/abs(a)
    alp1, bet1, gam, alp2, bet2 = alp
    r1, r2, rg, r3, r4 = [model._calc_rad(ell[i], alp[i]) for i in range(5)]

    elem = """
\\def\\col{%s}
\\def\\lw{%s}
\\def\\alpi{%f}
\\def\\beti{%f}
\\def\\gam{%f}
\\def\\alpii{%f}
\\def\\betii{%f}
\\def\\gamh{%f}

\\def\\eps{%f}
\\def\\ci{%f}
\\def\\cii{%f}
\\def\\ciii{%f}
\\def\\civ{%f}

\\def\\ri{%f}
\\def\\rii{%f}
\\def\\rg{%f}
\\def\\riii{%f}
\\def\\riv{%f}

\\path (%f, %f)coordinate(F1);

\\draw[\\col, line width=\\lw] (F1)arc(180+\\ci:180+\\ci+\\alpi:\\ri)coordinate(OM);
\\draw[\\col, line width=\\lw] (OM)arc(180+\\ci+\\alpi:180+\\ci+\\alpi+\\beti:\\rii)coordinate(F2);
\\draw[\\col, line width=\\lw] (OM)arc(90+\\ci+\\alpi:90+\\ci+\\alpi+\\gam:\\rg)coordinate(UM);
\\draw[\\col, line width=\\lw] (UM)arc(\\gam+\\ci+\\alpi:\\gam+\\ci+\\alpi+\\alpii:\\riii)coordinate(F3);
\\draw[\\col, line width=\\lw] (UM)arc(\\gam+\\ci+\\alpi:\\gam+\\ci+\\alpi-\\betii:\\riv)coordinate(F4);

""" % (col, linewidth, alp1, bet1, gam, alp2, bet2, gam*.5, eps, c1, c2, c3, c4,
       r1, r2, rg, r3, r4, F1[0], F1[1])
    if fix:
        for idx, fixation in enumerate(fix):
            c = [c1, c2, c3, c4]
            if fixation:
                fixs = '\\draw[\\col, line width=\\lw, fill] (F%s)++(%f :.15) circle(.15);\n' % (str(idx+1), 
                              c[idx]+90 if idx in [0, 3] else c[idx]-90)
                elem += fixs
            else:
                fixs = '\\draw[\\col, line width=\\lw] (F%s)++(%f :.15) circle(.15);\n' % (str(idx+1),
                              c[idx]+90 if idx in [0, 3] else c[idx]-90)
                elem += fixs

    return elem
