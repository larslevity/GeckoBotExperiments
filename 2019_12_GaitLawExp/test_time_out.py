# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:02:50 2019

@author: AmP
"""

from func_timeout import func_timeout, FunctionTimedOut

import time


def doit(a, b):
    time.sleep(.5)
    return 1


try:
    doitReturnValue = func_timeout(1, doit, args=('arg1', 'arg2'))
    print(doitReturnValue)
except FunctionTimedOut:
    print('time out ...')


