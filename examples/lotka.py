""" Lotka volterra DDE """


"""
    We will solve the delayed Lotka-Volterra system defined as
    
        For t < 0:
        x(t) = 1+t
        y(t) = 2-t
    
        For t >= 0:
        dx/dt =  0.5* ( 1- y(t-d) )
        dy/dt = -0.5* ( 1- x(t-d) )
    
    The delay ``d`` is a tunable parameter of the model.
"""
from pylab import array, linspace, subplots
from ddesolver import solve_dde


def model(Y, t, d):
    x, y = Y(t)
    xd, yd = Y(t - d)
    return array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])


g = lambda t: array([1, 2])
tt = linspace(2, 30, 20000)

fig, ax = subplots(1, figsize=(4, 4))

# The parameter d is configurable
for d in [0, 0.2]:
    print("Computing for d=%.02f" % d)
    yy = solve_dde(model, g, tt, fargs=(d,))
    # WE PLOT X AGAINST Y
    ax.plot(yy[:, 0], yy[:, 1], lw=2, label="delay = %.01f" % d)

ax.figure.savefig("lotka.jpeg")
