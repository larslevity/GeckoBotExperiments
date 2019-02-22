# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:17:06 2019

@author: AmP
"""

import os

try:
    from os import scandir
except ImportError:
    from scandir import scandir


def get_tex_names(dirpath):
    for entry in scandir(dirpath):
        if entry.is_file() and entry.name.endswith('.tex'):
            yield entry.path
        elif entry.is_dir():
            for subfile in get_tex_names(entry.path):
                yield subfile


dir_path = os.path.dirname(os.path.realpath(__file__))

names = get_tex_names(dir_path)

for name in names:
    out_dir = os.path.dirname(name)
    print name
    os.system('pdflatex -output-directory {} {}'.format(out_dir, name))
    print 'Done'
