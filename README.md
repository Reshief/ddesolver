# DDEsolver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Scipy-based delay differential equation (DDE) solver. See the docstrings and examples for more infos.

## Examples


```python
from pylab import cos, linspace, subplots
from ddesolver import solve_dde

# We solve the following system:
# Y(t) = 1 for t < 0
# dY/dt = -Y(t - 3cos(t)**2) for t > 0

def values_before_zero(t):
    return 1

def model(Y, t):
    return -Y(t - 3 * cos(Y(t)) ** 2)

tt = linspace(0, 30, 2000)
yy = solve_dde(model, values_before_zero, tt)

fig, ax = subplots(1, figsize=(4, 4))
ax.plot(tt, yy)
ax.figure.savefig("variable_delay.jpeg")
```

![screenshot](./examples/variable_delay.jpeg)

```python
from pylab import array, linspace, subplots
from ddesolver import solve_dde

# We solve the following system:
# X(t) = 1 (t < 0)
# Y(t) = 2 (t < 0)
# dX/dt = X * (1 - Y(t-d)) / 2
# dY/dt = -Y * (1 - X(t-d)) / 2


def model(Y, t, d):
    x, y = Y(t)
    xd, yd = Y(t - d)
    return array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])


g = lambda t: array([1, 2])
tt = linspace(2, 30, 20000)

fig, ax = subplots(1, figsize=(4, 4))

for d in [0, 0.2]:
    print("Computing for d=%.02f" % d)
    yy = solve_dde(model, g, tt, fargs=(d,))
    # WE PLOT X AGAINST Y
    ax.plot(yy[:, 0], yy[:, 1], lw=2, label="delay = %.01f" % d)

ax.figure.savefig("lotka.jpeg")
```

![screenshot](./examples/lotka.jpeg)

## Licence

Licensed under the MIT license, see `LICENSE` file for the detailed license text.

## Installation

ddesolver can be installed by unzipping the source code in one directory, installing setuptools using pip and then running setup.py: ::
```
    (sudo) python3 -m pip install --upgrade pip setuptools
    (sudo) python3 setup.py install
```


### Installation of the original repository provided by Zulko

You can also install the original ddeint package directly from the Python Package Index with this command: ::
```
    (sudo) pip install ddeint 
```
