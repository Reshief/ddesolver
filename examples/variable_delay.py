""" DDE where the delay depends on Y(t). """

from pylab import cos, linspace, subplots
from ddesolver import solve_dde


def model(Y, t):
    return -Y(t - 3 * cos(Y(t)) ** 2)


def values_before_zero(t):
    return 1


tt = linspace(0, 30, 2000)
yy = solve_dde(model, values_before_zero, tt)

fig, ax = subplots(1, figsize=(4, 4))
ax.plot(tt, yy)
ax.figure.savefig("variable_delay.jpeg")
