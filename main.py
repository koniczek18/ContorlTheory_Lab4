"""
@author of example code that was modified for the purpose of this task:
Radoslaw Patelski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# initialize constant values
b = 0.5
k = 1
m = 1

# set time step and calculate vector of timestamps
Tp = 0.1
Tf = 10
samples = int(Tf / Tp + 1)
T = np.linspace(0, Tf, samples)

# prepare all input signals - control U and noised W and V
U = np.full(samples, 1)
W = np.random.normal(0, 1, samples)
V = np.array([np.random.normal(0, 0.0, samples),
              np.random.normal(0, 0.0, samples)])
# for now assume no process noise V (model is perfect)

# define arrays A, B and C and D of a linear system
A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([[1, 0]])
D = np.array([[0]])

# modify signals to include noise (for simulation of continuous model only)
B0 = np.array([[1, 0], [0, 1]])
U0 = np.array([V[0], V[1] + 1 / m * U])
D0 = np.array([[0, 0]])

# simulate the dynamic system, pass matrices of linear system model, control signal and timestamps
res = signal.lsim([A, B0, C, D0], U0.T, T)
X = res[2]
Y = res[1]

# plot states of dynamic system
if False:
    plt.figure()
    plt.plot(T, X[:, 0], label='x1')
    plt.plot(T, X[:, 1], label='x2')
    plt.title("Real plant")
    plt.grid()
    plt.legend()

# add noise to the measurement
Y_noised = Y + W
if False:
    plt.figure()
    plt.plot(T, Y_noised, label='noised output')
    plt.title("Noised measurement")
    plt.grid()
    plt.legend()

# calculate matrices A, B, C and D of discrete-time system based on continous system
Ad, Bd, Cd, Dd, dt = signal.cont2discrete((A, B, C, D), Tp)

# initialize all matrices of Kalman Filter - feel free to change these values
Xc = np.array([[0], [0]])  # Xc (a posteriori estimate X) - assumed initial conditions
Pc = np.array([[0, 0], [0, 0]])  # Pc (a posteriori P) - how unsure we are of initial conditions
# a priori estiamtes Xp and Pp are not initialized, as they will be calculated based on a posteriori values and measured signals

Q = np.cov(V)  # Q - assumed covariance of process noise
R = np.cov(W)  # R - assumed covariance of measurement noise

# some vectors to keep samples of Xc aposteriori and Xp apriori estimate for later plotting
XC = Xc.T
XP = Xc.T

# main loop of discrete-time simulation
for i in range(0, samples - 1):
    # Kalman filter
    temp = C.T
    Xp = Ad @ np.array([XC[-1]]).T + Bd * U[i]
    Pn = Ad @ Pc[-2:] @ A.T + Q

    S = Cd[0] @ Pn @ C[0].T + R
    K = Pn @ C.T * (1 / S)
    e = Y_noised[i] - Cd @ Xp
    Xc = Xp + K @ e
    P = Pn - K * S @ K.T

    # Add calculated values to XC and XP 
    XC = np.vstack([XC, Xc.T])
    XP = np.vstack([XP, Xp.T])
    Pc = np.vstack([Pc, P])

# plot the results
if True:
    plt.figure()
    plt.plot(T, XP[:,0],label='x1')
    plt.plot(T, XP[:, 1],label='x2')
    plt.title("Apriori estimate")
    plt.grid()
    plt.legend()

if True:
    plt.figure()
    plt.plot(T, XC[:,0],label='x1')
    plt.plot(T, XC[:, 1], label='x2')
    plt.title("Aposteriori estimate")
    plt.grid()
    plt.legend()

if True:
    plt.figure()
    plt.plot(T, X[:,0]- XC[:,0],label='x1')
    plt.plot(T, X[:, 1] - XC[:, 1], label='x2')
    plt.title("Estimation errors")
    plt.grid()
    plt.legend()

plt.show()