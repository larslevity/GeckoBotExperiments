# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:46:31 2019

@author: AmP
"""
import csv
import numpy as np


def read_csv(filename):
    dic = {}
    mapping = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for idx, row in enumerate(reader):
            # print ', '.join(row)
            if idx == 0:
                for jdx, key in enumerate(row):
                    mapping[jdx] = key
                    dic[key] = []
            else:
                for jdx, val in enumerate(row):
                    dic[mapping[jdx]].append(float(val) if val else np.nan)
    return dic
