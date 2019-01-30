# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:18:20 2019

@author: AmP
"""

from matplotlib2tikz import save as tikz_save
import fileinput


def save_as_tikz(filename):
    print('Saving as TikZ-Picture...')

    tikz_save(filename)
    insert_tex_header(filename)
    print('Done!')


def insert_tex_header(filename):
    header = \
        "\\documentclass[crop,tikz]{standalone} \
         \\usepackage[utf8]{inputenc} \
         \\usepackage{tikz} \
         \\usepackage{pgfplots} \
         \\pgfplotsset{compat=newest} \
         \\usepgfplotslibrary{groupplots} \
         \\begin{document} \
         "
    line_pre_adder(filename, header)
    # Append Ending
    ending = "%% End matplotlib2tikz content %% \n \\end{document}"
    with open(filename, "a") as myfile:
        myfile.write(ending)


def line_pre_adder(filename, line_to_prepend):
    f = fileinput.input(filename, inplace=1)
    for xline in f:
        if f.isfirstline():
            print line_to_prepend.rstrip('\r\n') + '\n' + xline,
        else:
            print xline,
