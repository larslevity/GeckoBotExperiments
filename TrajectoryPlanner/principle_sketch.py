#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:02:13 2019

@author: ls
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from Src import save
from Src import kin_model
from Src import predict_pose as pp


def start_pose(alp, feet, eps0=90, F1=(0, 0)):
    a0, a1, a23, a4, a5 = alp
    f0, f1, f2, f3 = feet
    init_pose = [(a0, a1, a23, a4, a5), eps0, F1]
    return init_pose


def calc_gait_and_draw_tikz(ref, filename='test', eps0=90, shift=True,
                            save_tikz=True, F1=(0, 0)):
    len_leg = 1
    len_tor = 1.2

    init_pose = start_pose(*ref[0], eps0=eps0, F1=F1)
    x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
                                             len_leg=len_leg, len_tor=len_tor,
                                             dev_ang=.1)
    pp.plot_gait(*data)
    plt.figure()

    marks_ = pp.marker_history(marks)  # marks[marker_idx][x/y][pose_idx]
    eps = pp.extract_eps(data)  # eps[pose_idx]
    ell = pp.extract_ell(data)

    geckostring = ''
    for pose_idx in range(len(marks_[0][0])-1):  # since initial pose is double
        pf1_x = marks_[0][0][pose_idx+1]
        pf1_y = marks_[0][1][pose_idx+1]  # position of foot 1
#        print(pf1_x, pf1_y)
#        print(ell[pose_idx+1])
        if len(ref) == 1:
            col = 'black'
        elif not save_tikz:
            col = 'black!40'
        else:
            col = 'black!{}'.format(50+50.*pose_idx/(len(marks_[0][0])-2))
        geckostring += '%%%% POSE {}\n'.format(pose_idx)
        if shift:
            geckostring += '\\begin{scope}[xshift=%s cm]\n' % (pose_idx*3*len_leg)
        geckostring = geckostring + save.tikz_draw_gecko(
                ref[pose_idx][0], ell[pose_idx+1], eps[pose_idx+1],
                (pf1_x, pf1_y), linewidth='1mm', fix=ref[pose_idx][1], col=col)
        if pose_idx == 0 or pose_idx == len(marks_[0][0])-2:
            geckostring += '\\path (OM)coordinate(%s); \n' % ('start' if pose_idx == 0 else 'end')
            geckostring += '\\draw[very thick, -latex] (OM)--++(\\eps:1); \n'
            geckostring += '\\draw[fill] (OM)circle(.1); \n'
        if shift:
            geckostring += '\\end{scope}'
        geckostring += '\n\n\n\n'
    if save_tikz:
        xshift, yshift = -3, -1
        geckostring += '\\draw[line width=1mm, latex-, blue] (end)++({},{})coordinate(end_)--($(start)+({},{})$)coordinate(start_) ;\n'.format(-pose_idx*3*len_leg+xshift if shift else xshift, yshift, xshift, yshift)
        geckostring += '\\draw[-latex, red] (start_)++({}:1)arc({}:{}:1);\n'.format(eps[0], eps[0], eps[-1])
        geckostring += '\\path (start_)++(%s:1)node[anchor=%s]{$\\Delta \\varepsilon$};' % ((eps[0]+(eps[-1]-eps[0])/2.), (eps[0]+(eps[-1]-eps[0])/2+180))
        geckostring += '\\draw[fill] (start_)circle(.1); \n'
        geckostring += '\\draw[very thick, -latex] (start_)--++({}:1); \n'.format(eps[0])
        geckostring += '\\draw[fill] (end_)circle(.1); \n'
        geckostring += '\\draw[very thick, -latex] (end_)--++({}:1); \n'.format(eps[-1])

    if save_tikz:
        save.save_as_tikz('tikz/'+filename+'.tex', geckostring, scale=1)
    else:
        return geckostring, eps, marks_


def draw_multiple_gaits_in_a_row(refs, filename='test', eps0=90):
    GECKOSTR = ''
    epslast = eps0
    F1last = (0, 0)
    for ref in refs:
        geckostring, eps, marks_ = calc_gait_and_draw_tikz(
                ref, None, eps0=epslast, shift=False, save_tikz=False,
                F1=F1last)
        geckostring += """
\\draw[line width=1mm, latex-, blue] (end)--(start);
\\draw[-latex, red] (start)++(%s:1)arc(%s:%s:1);
\\path (start)++(%s:1)node[anchor=%s]{$\\Delta \\varepsilon$};
                """ % (eps[0], eps[0], eps[-1], (eps[0]+(eps[-1]-eps[0])/2.),
                       (eps[0]+(eps[-1]-eps[0])/2+180))
        GECKOSTR += geckostring
        epslast = eps[-1]
        F1last = (marks_[0][0][-1], marks_[0][1][-1])  # position of foot 1

    save.save_as_tikz('tikz/'+filename+'.tex', GECKOSTR, scale=1)


if __name__ == "__main__":
    straight = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
#    calc_gait_and_draw_tikz(straight, 'straight')

    curve_spezial = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[45, 45, 1, 45, 45], [1, 0, 0, 1]],
        [[45, 45, 90, 45, 45], [1, 1, 0, 0]],
        [[45, 45, 1, 45, 45], [0, 0, 1, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
#    calc_gait_and_draw_tikz(curve_spezial, 'curve_spezial')

    curve_left = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
        [[1, 40, 10, 10, 60], [0, 1, 1, 0]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
#    calc_gait_and_draw_tikz(curve_left, 'curve_left')

    curve_right = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[40, 1, -10, 60, 10], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
#    calc_gait_and_draw_tikz(curve_right, 'curve_right')

    curve_right_tight = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[48, 104, 114, 27, 124], [0, 1, 1, 0]],
        [[1, 72, 70, 1, 55], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
#    calc_gait_and_draw_tikz(curve_right_tight, 'curve_right_tight')

    curve_right_super_tight = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[50, 30, 90, 30, 150], [1, 0, 0, 1]],
        [[124, 164, 152, 62, 221], [0, 1, 1, 0]],
        [[30, 90, 80, 10, 10], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
#    calc_gait_and_draw_tikz(curve_right_super_tight, 'curve_right_super_tight')

    example = [
            curve_right_super_tight,
            curve_right,
            curve_right,
            straight,
            straight,
            straight,
            curve_left,
            curve_left,
            curve_left,
            curve_left
             ]
#    draw_multiple_gaits_in_a_row(example, 'example', eps0=110)
    example2 = [
            straight,
            straight,
            curve_right,
            straight,
            curve_right,
            straight,
            straight,
            curve_right_tight,
            curve_right_super_tight,
            straight,
            curve_left,
            straight,
            straight,
            straight,
            straight,
            straight,
            curve_right,
            curve_right_tight,
            straight,
            curve_right_tight,
            straight
             ]

    draw_multiple_gaits_in_a_row(example2, 'example2', eps0=110)
