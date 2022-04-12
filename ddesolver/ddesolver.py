"""
This module implements ddesolver, a simple Differential Delay Equation
solver built on top of Scipy's odeint.

It has been expanded to support different ode backends provided by scipy.
"""

# REQUIRES Numpy and Scipy.
import numpy as np
import scipy.integrate
import scipy.interpolate


class ddeVar:
    """
    The instances of this class are special function-like
    variables which store their past values in an interpolator and
    can be called for any past time: Y(t), Y(t-d).
    Very convenient for the integration of DDEs.

    Initial values for the variable prior to an initial cutoff time 
    are provided by the generator function.

    By default, the cutoff between the generator and the simulated 
    instances of the variable is set to t=0.
    """

    def __init__(self, generator, generator_cutoff_time=0):
        """ generator(t) = expression of Y(t) for t<generator_cutoff_time """

        self.generator = generator
        self.generator_cutoff_time = generator_cutoff_time
        # We must fill the interpolator with 2 points minimum

        self.interpolator = scipy.interpolate.interp1d(
            np.array([generator_cutoff_time - 1, generator_cutoff_time]),  # X
            np.array([self.generator(generator_cutoff_time),
                      self.generator(generator_cutoff_time)]).T,  # Y
            kind="linear",
            bounds_error=False,
            fill_value=self.generator(generator_cutoff_time)
        )

    def update(self, t, Y):
        """ Add one new (ti,yi) to the interpolator """
        Y2 = Y if (Y.size == 1) else np.array([Y]).T
        self.interpolator = scipy.interpolate.interp1d(
            np.hstack([self.interpolator.x, [t]]),  # X
            np.hstack([self.interpolator.y, Y2]),  # Y
            kind="linear",
            bounds_error=False,
            fill_value=Y
        )

    def __call__(self, t=0):
        """ Y(t) will return the instance's value at time t """

        return self.generator(t) if (t <= self.generator_cutoff_time) else self.interpolator(t)


class dde(scipy.integrate.ode):
    """
    This class overwrites a few functions of ``scipy.integrate.ode``
    to allow for updates of the pseudo-variable Y between each
    integration step.
    """

    def __init__(self, f, jac=None):
        def f2(t, y, args):
            return f(self.Y, t, *args)

        scipy.integrate.ode.__init__(self, f2, jac)
        self.set_f_params(None)

    def integrate(self, t, step=0, relax=0):

        scipy.integrate.ode.integrate(self, t, step, relax)
        self.Y.update(self.t, self.y)
        return self.y

    def set_initial_value(self, Y):

        self.Y = Y  # !!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(
            self, Y(Y.generator_cutoff_time), Y.generator_cutoff_time)


def solve_dde(func, generator, tt, fargs=None, integrator="dopri5", **integrator_params):
    """ Solves Delayed Differential Equations

    Similar to scipy.integrate.odeint. Solves a Delayed differential
    Equation system (DDE) defined by

        Y(t) = generator(t) for t<0
        Y'(t) = func(Y,t) for t>= 0

    Where func can involve past values of Y, like Y(t-d).


    Parameters
    -----------

    func
      a function Y,t,args -> Y'(t), where args is optional.
      The variable Y is an instance of class ddeVar, which means that
      it is called like a function: Y(t), Y(t-d), etc. Y(t) returns
      either a number or a numpy array (for multivariate systems).

    generator
      The 'history function'. A function generator(t)=Y(t) for t<0, generator(t)
      returns either a number or a numpy array (for multivariate
      systems).

    tt
      The vector of times [t0, t1, ...] at which the system must
      be solved.

    fargs
      Additional arguments to be passed to parameter ``func``, if any.

    integrator
      Name of the integrator to be used. The total list of available integrators with 
      explanations can be found at: 
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
      The available options are:
        - dopri5: A Runge-Kutta 5(4) integrator
        - dop853: An explicit runge-kutta method of order 8(5,3) 
        - lsoda: Real-valued Variable-coefficient Ordinary Differential Equation solver
        - zvode: Complex-valued Variable-coefficient Ordinary Differential Equation solver, with fixed-leading-coefficient implementation.
        - vode: Real-valued Variable-coefficient Ordinary Differential Equation solver, with fixed-leading-coefficient implementation.
      By default, dopri5 will be used for integration to increase accuracy

    integrator_params
      Can be used to pass integrator parameters to the chosen integrator

    """

    # Configure the custom DDE integrator
    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(generator, tt[0]))
    dde_.set_f_params(fargs if fargs else [])

    dde_.set_integrator(integrator, **integrator_params)

    # Setup result array
    results = []
    results.append(generator(tt[0]))

    # Calculate necessary simulation time steps
    t_diffs = np.diff(tt)

    # Execute DDE integration
    for step_index in range(len(t_diffs)):
        dt = t_diffs[step_index]
        results.append(dde_.integrate(dde_.t + dt))

    return np.array(results)
