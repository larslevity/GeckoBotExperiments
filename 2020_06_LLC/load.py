# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:30:09 2020

@author: AmP
"""

import csv
import numpy as np


try:
    from os import scandir
except ImportError:
    from scandir import scandir


def read_csv(filename):
    dic = {}
    mapping = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for idx, row in enumerate(reader):
            # print ', '.join(row)
            if idx == 0:
                for jdx, key in enumerate(row):
                    mapping[jdx] = key
                    dic[key] = []
            else:
                for jdx, val in enumerate(row):
                    try:
                        val_ = float(val)
                    except ValueError:
                        val_ = np.nan
                    dic[mapping[jdx]].append(val_)
    return dic


def _get_csv_names(dirpath):
    for entry in scandir(dirpath):
        if entry.is_file() and entry.name.endswith('.csv'):
            yield entry.name


def get_csv_set(dirpath):
    names = _get_csv_names(dirpath)
    return [name.split('.csv')[0] for name in sorted(names)]
