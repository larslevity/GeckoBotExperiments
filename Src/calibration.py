#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:55:25 2019

@author: ls
"""
import numpy as np

clb = {
    'vS11': {  # special 3
        0: [4.965121562109887e-13, -6.44097132882056e-10, 1.5158359627494762e-07, -3.205605883004579e-05, 0.010052638820931926, 0.134082375974453],
        1: [5.5226508224484436e-11, -2.4185976436640417e-08, 3.8076295002966234e-06, -0.000274055038179612, 0.015949085821324958, -0.02961559405387044],
        2: [9.160037425064399e-10, -3.08455572578132e-07, 3.9453676119984866e-05, -0.002331085055379165, 0.06829721246843559, -0.37656593017926027],
        3: [-1.0393633988836174e-10, 1.765936421017443e-08, 1.6329753165788568e-07, -0.00011664751581913587, 0.011938499721436678, 0.17433420994695759],
        4: [2.89230113993481e-10, -9.113299625275847e-08, 1.0259866988732249e-05, -0.0005220166542632878, 0.02092563431788812, -0.033566176975896325],
        5: [1.6425202957498428e-10, -5.3635789525259444e-08, 6.24097106975067e-06, -0.0003298483624510243, 0.016289891890431146, 0.004707638851766482]
        },
    'v40': {
        0: [-2.1251499400543305e-10, 6.511742295994745e-08, -6.930341079275748e-06, 0.00025539238007335713, 0.008163307461283085, 0.09946327907294761], 
        1: [5.305232665510037e-11, -1.2984794839993249e-08, 9.95585604716407e-07, -7.100932077073888e-05, 0.013487181695231856, 0.03852251493296447], 
        2: [7.64444165215644e-10, -2.061530199942083e-07, 2.1108308998962592e-05, -0.001001839465373247, 0.027356321293677172, 0.13277666040348357],
        3: [3.501266798189199e-10, -1.219168564951355e-07, 1.5799395015860383e-05, -0.0009300875239806064, 0.030503845838564712, 0.051094988401221605],
        4: [1.1276541202883278e-10, -4.375209700714454e-08, 6.657834475485559e-06, -0.0005344578344845812, 0.029076204519095847, -0.11010599833885393],
        5: [3.937626786344588e-11, -1.5979916338336208e-08, 2.7645524193951676e-06, -0.0002858009953042876, 0.020945769202969186, 0.04972652892347207]
        },
    }


clb_len = {
    'vS11': [9.1, 10.3],
    'v40': [11.9, 14.6]
    }


def get_kin_model_params():
    f_l = 100.      # factor on length objective
    f_o = 0.1     # .0003     # factor on orientation objective
    f_a = 10   # factor on angle objective
    return (f_l, f_o, f_a)



def get_len(version):
    return clb_len[version]


def get_pressure(alpha, version, max_pressure=1):
    pressure = []
    alpha_ = alpha[0:3] + [None] + alpha[3:]

    def cut_off(p):
        if p > max_pressure:
            # Warning('clb pressure > max_presse: I cutted it off')
            p_ = max_pressure
        elif p < 0:
            p_ = 0.00
        else:
            p_ = round(p, 2)
        return p_
    try:
        for idx, alp in enumerate(alpha_):
            if alp is not None:
                if idx == 2:
                    if alp > 0:  # left belly actuated
                        p = eval_poly(clb[version][idx], alp)
                        pressure.append(cut_off(p))
                        pressure.append(.00)  # actuator 3 -> right belly
                    elif alp == 0:
                        pressure.append(.00)
                        pressure.append(.00)
                    else:  # right belly actuated
                        pressure.append(.00)  # actuator 2 -> left belly
                        p = eval_poly(clb[version][idx+1], abs(alp))
                        pressure.append(cut_off(p))
                else:
                    p = eval_poly(clb[version][idx], alp)
                    pressure.append(cut_off(p))
        return pressure + [0, 0]  # 8 channels
    except KeyError:
        raise NotImplementedError


def get_alpha(pressure, version):
    pressure = pressure[:6]  # 6 clbs
    sign_alp = -1 if pressure[2] == 0 else 1

    alp = []
    for idx, p in enumerate(pressure):
        coeff = clb[version][idx]
        poly = np.poly1d(coeff)
        roots = (poly - p).roots
        roots = roots[~np.iscomplex(roots)].real
        roots = roots[roots > 0]
        roots = roots[roots < 120]
        if len(roots) == 1:
            alp.append(roots[0])
        elif len(roots) == 0:
            alp.append(0)
        else:
            alp.append(np.mean(roots))
    if sign_alp < 0:
        alp = alp[:2] + alp[-3:]
    else:
        alp = alp[:3] + alp[-2:]
    alp[2] *= sign_alp

    return alp


def eval_poly(coef, x):
    if x < 0:
        return 0
    else:
        poly = np.poly1d(coef)
        return poly(x)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    version = 'vS11'

    alp = [30, 45, -100, 45, 45]
    print('alp:\t', alp)
    p = get_pressure(alp, version=version)
    print('p:\t', p)
    alp_ = get_alpha(p, version)
    print('alp_:\t', [round(a) for a in alp_])

    alp = [30, 45, 100, 45, 45]
    print('\nalp:\t', alp)
    p = get_pressure(alp, version=version)
    print('p:\t', p)
    alp_ = get_alpha(p, version)
    print('alp_:\t', [round(a) for a in alp_])


    for alpha in np.linspace(-120, 120, 50):
        alp = [alpha]*5
        p = get_pressure(alp, version=version)
        alp_ = get_alpha(p, version)

        for idx, alp__ in enumerate(alp_):
            plt.figure(idx)
            if idx > 2:
                plt.plot(alp[idx], p[idx+1]*100, 'r.')
            else:
                plt.plot(alp[idx], p[idx]*100, 'r.')
            if idx == 2:
                plt.plot(alp[idx], p[idx+1]*100, 'r.')

            plt.plot(alp[idx], alp__, 'k.')
    for idx, _ in enumerate(alp_):
        plt.figure(idx)
        plt.xlabel('alpha [deg]')
        plt.ylabel('pressure(alpha) [bar*100] / alpha_ [deg]')
        plt.plot(0, 0, 'r.', label='p(alp) * 100')
        plt.plot(0, 0, 'k.', label='alp_(p(alp))')
        plt.legend()
