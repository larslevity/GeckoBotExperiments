#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:37:31 2019

@author: ls
"""

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')



axes[0].plot(range(50))
axes[1].plot(range(100))

plt.axis('equal')


plt.show()