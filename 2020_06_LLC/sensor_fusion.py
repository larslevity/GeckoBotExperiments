#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:50:56 2020

@author: ls
"""

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

import load as my_load
import compute_utils as utils



col = ['red', 'blue', 'orange']  # x y z colors


# %% LOAD
exp = 'pattern0_0g/PPID.csv'
#exp = 'pattern0_0g/PBooster_1_0.csv'
data = my_load.read_csv(exp)

# %% find pattern start and end

smargin = 40  # number of samples before pattern
emargin = 100  # number of samples after pattern


alpha_marica = np.array(data['aIMU0'])

time = np.array(data['time'])
Ts = np.mean(np.diff(time))  # calc mean sampling time
gamma = .07

gyrscale = 1/2500
accscale = 9.81

accA = np.array([np.array(data['accx0']),
                 np.array(data['accy0']),
                 np.array(data['accz0'])])*accscale
accA_filt = np.array(
        [utils.lowpass_array(data['accx0'], Ts, gamma),
         utils.lowpass_array(data['accy0'], Ts, gamma),
         utils.lowpass_array(data['accz0'], Ts, gamma)])*accscale

accB = np.array([np.array(data['accx1']),
                 np.array(data['accy1']),
                 np.array(data['accz1'])])*accscale
accB_filt = np.array(
        [utils.lowpass_array(data['accx1'], Ts, gamma),
         utils.lowpass_array(data['accy1'], Ts, gamma),
         utils.lowpass_array(data['accz1'], Ts, gamma)])*accscale

gyrA = np.array([np.array(data['gyrx0']),
                 np.array(data['gyry0']),
                 np.array(data['gyrz0'])])*gyrscale
gyrA_filt = np.array(
        [utils.lowpass_array(data['gyrx0'], Ts, gamma),
         utils.lowpass_array(data['gyry0'], Ts, gamma),
         utils.lowpass_array(data['gyrz0'], Ts, gamma)])*gyrscale

gyrB = np.array([np.array(data['gyrx1']),
                 np.array(data['gyry1']),
                 np.array(data['gyrz1'])])*gyrscale
gyrB_filt = np.array(
        [utils.lowpass_array(data['gyrx1'], Ts, gamma),
         utils.lowpass_array(data['gyry1'], Ts, gamma),
         utils.lowpass_array(data['gyrz1'], Ts, gamma)])*gyrscale


aref = np.array(data['aref0'])
pressure_ref = np.array(data['u0'])/100.

N = len(aref)
start = [i-smargin for i in range(N-1) if (aref[i]==0 and aref[i+1]==60)]
end = [i+emargin for i in range(N-1) if (aref[i]==40 and aref[i+1]==0)]


which_cyc = 4
s = start[which_cyc]
e = end[which_cyc]
s0 = s


# %% Check lowpass filt
fig = plt.figure(2)
for i in range(3):
    plt.plot(time[s:e]-time[s], accB[i][s:e], ':', color=col[i])
for i in range(3):
    plt.plot(time[s:e]-time[s], accB_filt[i][s:e], color=col[i])
plt.ylabel('acc$_B$')
plt.grid()
fig.set_size_inches(18.5, 10.5)
plt.savefig('Out/sensorfusion/check_filt.png', dpi=300)


# %% compute alphas

alpha_a = []
for (aA, aB) in zip(accA.T, accB.T):
    alpha_a.append(utils.calc_angle(aA, aB))
alpha_acc = np.array(alpha_a)

alpha_a = []
for (aA, aB) in zip(accA_filt.T, accB_filt.T):
    alpha_a.append(utils.calc_angle(aA, aB))
alpha_acc_filt = np.array(alpha_a)


# %% a dyn

ell = 11.2
#domegaB_z = np.diff(np.insert(gyrB_filt[2], 0, 0))

domegaB_z = utils.diff_array((gyrB_filt[2]-gyrA_filt[2]), Ts) 
#domegaB_z = utils.diff_array(gyrB_filt[2], Ts)

accB_static_x = [0]
accB_static_y = [0]
alpha_dyn_filt = [0]


for (aA, aB, dwB) in zip(accA_filt.T[1:], accB_filt.T[1:], domegaB_z):
    adynx, adyny = utils.a_dyn(dwB, alpha_dyn_filt[-1], ell)
    aB_static = [aB[0]+adynx, aB[1]+adyny, aB[2]]
    accB_static_x.append(aB_static[0])
    accB_static_y.append(aB_static[1])
    alpha_dyn_filt.append(utils.calc_angle(aA, aB_static))
alpha_dyn_filt = np.array(alpha_dyn_filt)
accB_static_x = np.array(accB_static_x)
accB_static_y = np.array(accB_static_y)


    
# %% plot only alpha

fig = plt.figure(3)
plt.plot(time[s:e]-time[s], alpha_acc[s:e], '-', color='green',  label='$\\alpha(a_{raw})$')
#plt.plot(time[s:e]-time[s], alpha_marica[s:e], '--', color='blue',  label='$\\alpha(marica)$')
plt.plot(time[s:e]-time[s], alpha_acc_filt[s:e], '-', color='red',  label='$\\alpha(a_{filt})$')
plt.plot(time[s:e]-time[s], alpha_dyn_filt[s:e], '-', color='blue',  label='$\\alpha(a_{static})$')


plt.ylabel('bending angle $\\alpha$ ($^\\circ$)')
plt.xlabel('time (s)')
plt.grid()
plt.legend()

#fig.set_size_inches(18.5, 10.5)
#plt.savefig('Out/sensorfusion/alpha_course.png', dpi=300)

tikzplotlib.save('Out/sensorfusion/alpha_course.tex', standalone=True)


# %% plot tripple axis


limacc = [np.min(accB)-.3, np.max(accB)+.3]
limgyr = [np.min(gyrB)-.05, np.max(gyrB)+.05]


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
     gridspec_kw={'height_ratios': [1, 1]})


# IMU A
for i in range(3):
    ax1.plot(time[s:e]-time[s], gyrA[i][s:e], '--', color=col[i])
for i in range(3):
    ax1.plot(time[s:e]-time[s], gyrA_filt[i][s:e], color=col[i])
ax1.set_ylabel('gyr$_A$')
ax1.grid()
ax1.set_ylim(*limgyr)

for i in range(3):
    ax2.plot(time[s:e]-time[s], accA[i][s:e], '--', color=col[i])
for i in range(3):
    ax2.plot(time[s:e]-time[s], accA_filt[i][s:e], color=col[i])
ax2.set_ylabel('acc$_A$')
ax2.grid()
ax2.set_ylim(*limacc)


#fig.set_size_inches(18.5, 10.5)
#plt.savefig('Out/sensorfusion/ImuA.png', dpi=300)

tikzplotlib.save('Out/sensorfusion/ImuA.tex', standalone=True)

# %% IMU B


fig, (ax3, axdW, ax4, ax5) = plt.subplots(nrows=4, sharex=True,
     gridspec_kw={'height_ratios': [2, 1, 2, 3]})

axdW.plot(time[s:e]-time[s], domegaB_z[s:e], '-', color=col[2])
axdW.set_ylabel(r'$\dot{\omega}$ (rad/s$^{-2}$)')
axdW.grid()


# IMU B
for i in range(3):
    ax3.plot(time[s:e]-time[s], gyrB[i][s:e], '-', color=col[i], alpha=.5, linewidth=.7)
for i in range(3):
    ax3.plot(time[s:e]-time[s], gyrB_filt[i][s:e], color=col[i], alpha=.5, linewidth=1)
ax3.set_ylabel(r'gyr$_B$ (rad/s$^{-1}$)')
ax3.grid()
ax3.set_ylim(*limgyr)

for i in range(3):
    ax4.plot(time[s:e]-time[s], accB[i][s:e], '-', color=col[i], alpha=.5, linewidth=.5)
for i in range(3):
    ax4.plot(time[s:e]-time[s], accB_filt[i][s:e], '-', color=col[i], alpha=.7, linewidth=.7)
ax4.plot(time[s:e]-time[s], accB_static_x[s:e], '-', color=col[0], linewidth=1)
ax4.plot(time[s:e]-time[s], accB_static_y[s:e], '-', color=col[1], linewidth=1)

ax4.set_ylabel(r'acc$_B$ (m/s$^{-2}$)')
ax4.grid()
ax4.set_ylim(*limacc)

# alpha
ax5.plot(time[s:e]-time[s], alpha_acc[s:e], '-', color='green',  label='$\\alpha(a_{raw})$')
ax5.plot(time[s:e]-time[s], alpha_acc_filt[s:e], '-', color='red',  label='$\\alpha(a_{filt})$')
ax5.plot(time[s:e]-time[s], alpha_dyn_filt[s:e], '-', color='blue',  label='$\\alpha(a_{static})$')
ax5.set_yticks([0, 20, 60, 90])
ax5.set_xticks([0, 1.85, 3.1, 6.9, 11.75, 13.7, 17, 21.6])
ax5.set_ylim(-10, 110)
ax5.set_xlim(0, 25)
ax5.set_ylabel(r'$\alpha$ ($^\circ$)')
ax5.set_xlabel('time (s)')
ax5.grid()
ax5.legend()

kwargs = {
    'strict': 1,
    'extra_tikzpicture_parameters': {},
    'extra_axis_parameters': {'height={4cm}', 'width={12cm}',
                              'y label style={at={(axis description cs:-0.12,.5)},rotate=0,anchor=south}'},
    'extra_groupstyle_parameters': {'vertical sep={0pt}',
                                    'x descriptions at=edge bottom'}
        }


tikzplotlib.save('Out/sensorfusion/ImuB.tex', standalone=True, **kwargs)

#fig.set_size_inches(18.5, 10.5)
#plt.savefig('Out/sensorfusion/ImuB.png', dpi=300)


# %% close up

s1=3910
e1=3970

s2=4225
e2=4285

s3=4540
e3=4600


fig, ((axgyrB1, axgyrB2, axgyrB3), 
      (axdW1, axdW2, axdW3), 
      (axaccB1, axaccB2, axaccB3),
      (axalp1, axalp2, axalp3)) = plt.subplots(
        nrows=4, ncols=3, 
        sharex='col', sharey='row',
        gridspec_kw={'height_ratios': [2, 1, 2, 3]}
        )

# labels
axdW1.set_ylabel(r'$\dot{\omega}$ (rad/s$^{-2}$)')
axaccB1.set_ylabel(r'acc$_B$ (m/s$^{-2}$)')
axgyrB1.set_ylabel(r'gyr$_B$ (rad/s$^{-1}$)')
axalp1.set_ylabel(r'bending angle $\alpha$ ($^\circ$)')
axalp1.set_xlabel(r'time (s)')
axalp2.set_xlabel(r'time (s)')
axalp3.set_xlabel(r'time (s)')

## PLOTS
# axgyrB1
for i in range(3):
    axgyrB1.plot(time[s1:e1]-time[s0], gyrB[i][s1:e1], '-', color=col[i], alpha=.5, linewidth=.7)
for i in range(3):
    axgyrB1.plot(time[s1:e1]-time[s0], gyrB_filt[i][s1:e1], color=col[i], alpha=.5, linewidth=1)
# axgyrB2
for i in range(3):
    axgyrB2.plot(time[s2:e2]-time[s0], gyrB[i][s2:e2], '-', color=col[i], alpha=.5, linewidth=.7)
for i in range(3):
    axgyrB2.plot(time[s2:e2]-time[s0], gyrB_filt[i][s2:e2], color=col[i], alpha=.5, linewidth=1)
# axgyrB3
for i in range(3):
    axgyrB3.plot(time[s3:e3]-time[s0], gyrB[i][s3:e3], '-', color=col[i], alpha=.5, linewidth=.7)
for i in range(3):
    axgyrB3.plot(time[s3:e3]-time[s0], gyrB_filt[i][s3:e3], color=col[i], alpha=.5, linewidth=1)


# axdW1
axdW1.plot(time[s1:e1]-time[s0], domegaB_z[s1:e1], '-', color=col[2])
# axdW2
axdW2.plot(time[s2:e2]-time[s0], domegaB_z[s2:e2], '-', color=col[2])
# axdW3
axdW3.plot(time[s3:e3]-time[s0], domegaB_z[s3:e3], '-', color=col[2])

# axaccB1
for i in range(3):
    axaccB1.plot(time[s1:e1]-time[s0], accB[i][s1:e1], '-', color=col[i], alpha=.5, linewidth=.5)
for i in range(3):
    axaccB1.plot(time[s1:e1]-time[s0], accB_filt[i][s1:e1], '-', color=col[i], alpha=.7, linewidth=.7)
axaccB1.plot(time[s1:e1]-time[s0], accB_static_x[s1:e1], '-', color=col[0], linewidth=1)
axaccB1.plot(time[s1:e1]-time[s0], accB_static_y[s1:e1], '-', color=col[1], linewidth=1)
# axaccB2
for i in range(3):
    axaccB2.plot(time[s2:e2]-time[s0], accB[i][s2:e2], '-', color=col[i], alpha=.5, linewidth=.5)
for i in range(3):
    axaccB2.plot(time[s2:e2]-time[s0], accB_filt[i][s2:e2], '-', color=col[i], alpha=.7, linewidth=.7)
axaccB2.plot(time[s2:e2]-time[s0], accB_static_x[s2:e2], '-', color=col[0], linewidth=1)
axaccB2.plot(time[s2:e2]-time[s0], accB_static_y[s2:e2], '-', color=col[1], linewidth=1)
# axaccB3
for i in range(3):
    axaccB3.plot(time[s3:e3]-time[s0], accB[i][s3:e3], '-', color=col[i], alpha=.5, linewidth=.5)
for i in range(3):
    axaccB3.plot(time[s3:e3]-time[s0], accB_filt[i][s3:e3], '-', color=col[i], alpha=.7, linewidth=.7)
axaccB3.plot(time[s3:e3]-time[s0], accB_static_x[s3:e3], '-', color=col[0], linewidth=1)
axaccB3.plot(time[s3:e3]-time[s0], accB_static_y[s3:e3], '-', color=col[1], linewidth=1)

# axalp1
axalp1.plot(time[s1:e1]-time[s0], alpha_acc[s1:e1], '-', color='green',  label='$\\alpha(a_{raw})$')
axalp1.plot(time[s1:e1]-time[s0], alpha_acc_filt[s1:e1], '-', color='red',  label='$\\alpha(a_{filt})$')
axalp1.plot(time[s1:e1]-time[s0], alpha_dyn_filt[s1:e1], '-', color='blue',  label='$\\alpha(a_{static})$')
# axalp2
axalp2.plot(time[s2:e2]-time[s0], alpha_acc[s2:e2], '-', color='green',  label='$\\alpha(a_{raw})$')
axalp2.plot(time[s2:e2]-time[s0], alpha_acc_filt[s2:e2], '-', color='red',  label='$\\alpha(a_{filt})$')
axalp2.plot(time[s2:e2]-time[s0], alpha_dyn_filt[s2:e2], '-', color='blue',  label='$\\alpha(a_{static})$')
# axalp3
axalp3.plot(time[s3:e3]-time[s0], alpha_acc[s3:e3], '-', color='green',  label='$\\alpha(a_{raw})$')
axalp3.plot(time[s3:e3]-time[s0], alpha_acc_filt[s3:e3], '-', color='red',  label='$\\alpha(a_{filt})$')
axalp3.plot(time[s3:e3]-time[s0], alpha_dyn_filt[s3:e3], '-', color='blue',  label='$\\alpha(a_{static})$')





## FORMATING

axalp3.set_ylim(-10, 105)
axalp3.set_yticks([0, 20, 60, 90])
axaccB1.set_yticks([-10, 0, 10])
axdW1.set_yticks([-.2, 0, .2])
axgyrB1.set_yticks([-.04, 0, .04])


axalp1.grid()
axalp2.grid()
axalp3.grid()

axaccB1.grid()
axaccB2.grid()
axaccB3.grid()

axdW1.grid()
axdW2.grid()
axdW3.grid()

axgyrB1.grid()
axgyrB2.grid()
axgyrB3.grid()

## SAVE
axis_params = {'height={4cm}', 'width={6cm}', 'y label style={at={(axis description cs:-0.25,.5)},rotate=0,anchor=south}'}

kwargs = {
    'strict': 1,
    'extra_tikzpicture_parameters': {},
    'extra_axis_parameters': axis_params,
    'extra_groupstyle_parameters': {'vertical sep={5pt}', 'horizontal sep={5pt}',
                                    'x descriptions at=edge bottom'}
        }


tikzplotlib.save('Out/sensorfusion/ImuB_closeUp3.tex', standalone=True, **kwargs)
