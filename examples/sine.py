""" Reproduces the sine function using a DDE """

from pylab import linspace, sin, subplots, pi
from ddesolver import solve_dde


def model(Y, t):
    return Y(t - 3 * pi / 2)  # Model


tt = linspace(0, 50, 10000)  # Time start, time end, nb of pts/steps
g = sin  # Expression of Y(t) before the integration interval
yy = solve_dde(model, g, tt)  # Solving

fig, ax = subplots(1, figsize=(4, 4))
ax.plot(tt, yy)
ax.figure.savefig("sine.jpeg")
