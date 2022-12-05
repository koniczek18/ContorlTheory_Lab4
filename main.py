import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal

b = 0.5
k = 1
m = 1

Tp = 0.1
Tf = 10
samples = int(Tf / Tp + 1)
T = np.linspace(0, Tf, samples)

A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([[1, 0]])
D = np.array([[0]])

U = np.full(samples, 1)
W = np.random.normal(0, 1, samples)
V = np.array([np.random.normal(0, 0.0, samples),
              np.random.normal(0, 0.0, samples)])

B0 = np.array([[1, 0], [0, 1]])
U0 = np.array([V[0], V[1] + 1 / m * U])
D0 = np.array([[0, 0]])

res = signal.lsim([A, B0, C, D0], U0.T, T)
Y = res[1]
X = res[2]
x1 = X[:, 0]
x2 = X[:, 1]
noisedY = Y + W

plt.figure('System ')
plt.title("Real plant")
plt.plot(T, x1, label='cont x1')
plt.plot(T, x2, label='cont x2')
#plt.plot(T, noisedY, label='noised cont')

Ad, Bd, Cd, Dd, dt = signal.cont2discrete((A, B, C, D), Tp)
resDiscrete=signal.dlsim((Ad,Bd,Cd,Dd,dt),U,T)
Yd = resDiscrete[1].T
Xd = resDiscrete[2].T
x1d = np.array(Xd[0,:])
x2d = np.array(Xd[1,:])
noisedYd = Yd + W

plt.plot(T, x1d, label='disc x1',linestyle='dotted')
plt.plot(T, x2d, label='disc x2',linestyle='dotted')
#plt.plot(T, noisedYd, label='disc cont')

plt.grid()
plt.legend()
plt.show()

