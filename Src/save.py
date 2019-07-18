# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:18:20 2019

@author: AmP
"""

from matplotlib2tikz import save as tikz_save
import fileinput
from PIL import Image, ImageChops
import os
import numpy as np


def save_as_tikz(filename, gecko_str=None, scale=1, **kwargs):
    print('Saving as TikZ-Picture...')
    aux_fn = filename + '_aux'
    if gecko_str:
        kwargs = {'extra_axis_parameters':
                  {'anchor=origin', 'disabledatascaling', 'x=1cm', 'y=1cm'}}
    tikz_save(aux_fn, **kwargs)
    insert_tex_header(aux_fn, gecko_str, scale)

    # remove blank lines:
    with open(aux_fn, 'r') as file:
        try:
            with open(filename, 'x') as ofile:
                for line in file:
                    if not line == '\n':
                        ofile.write(line)
        except FileExistsError:
            with open(filename, 'w') as ofile:
                for line in file:
                    if not line == '\n':
                        ofile.write(line)
    os.remove(aux_fn)
    print('Done!')


def save_geckostr_as_tikz(filename, geckostr):
    header = """
\\documentclass[crop,tikz]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage{tikz}
\\begin{document}
\\begin{tikzpicture}[scale=1]
"""
    ending = """
%% End geckostr %%
\\end{tikzpicture}
\\end{document}
"""
    with open(filename, 'w') as fout:
            fout.writelines(header + geckostr + ending)


def insert_tex_header(filename, gecko_str=None, scale=1):
    header = """
\\documentclass[crop,tikz]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=newest}
\\usepgfplotslibrary{groupplots}
\\begin{document}
"""
    if gecko_str:
        # remove \begin{tikzpicture}
        with open(filename, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(filename, 'w') as fout:
            fout.writelines([data[0]] + data[2:])
        # add geckostr between header and matplotlib2tikz data
        header = header + '\n\\begin{tikzpicture}[scale=%s]' % (scale) + gecko_str
    line_pre_adder(filename, header)
    # Append Ending
    ending = "\n%% End matplotlib2tikz content %% \n\\end{document}"
    with open(filename, "a") as myfile:
        myfile.write(ending)


def line_pre_adder(filename, line_to_prepend):
    f = fileinput.input(filename, inplace=1)
    for xline in f:
        if f.isfirstline():
            print(line_to_prepend.rstrip('\r\n') + '\n' + xline,)
        else:
            print(xline,)


def trim(filename, border=1):
    im = Image.open(filename)
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        im2 = im.crop(bbox)
        im2.save(filename)

def crop_img(filename):
    im = Image.open(filename)
    print(im.getbbox())
    im2 = im.crop(im.getbbox())
    im2.save(filename)



def _calc_phi(alpha, eps):
    c1 = np.mod(eps - alpha[0] - alpha[2]*.5 + 360, 360)
    c2 = np.mod(c1 + alpha[0] + alpha[1] + 360, 360)
    c3 = np.mod(180 + alpha[2] - alpha[1] + alpha[3] + c2 + 360, 360)
    c4 = np.mod(180 + alpha[2] + alpha[0] - alpha[4] + c1 + 360, 360)
    phi = [c1, c2, c3, c4]
    return phi


def _calc_rad(length, angle):
    return 360.*length/(2*np.pi*angle)


def tikz_draw_gecko(alp, ell, eps, F1, col='black', linewidth='.5mm', fix=None):
    c1, c2, c3, c4 = _calc_phi(alp, eps)
    l1, l2, lg, l3, l4 = ell
    for idx, a in enumerate(alp):
        if abs(a) < 2:
            alp[idx] = 2 * a/abs(a)
    alp1, bet1, gam, alp2, bet2 = alp
    r1, r2, rg, r3, r4 = [_calc_rad(ell[i], alp[i]) for i in range(5)]

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
