"""
Implementation of integration methods.

"""

import jax.numpy as jnp
from jax import random


class DeterministicIntegrator:
    """
    Integration using domain.deterministic_integration_points().

    This integrator uses the deterministic_integration_points
    method of the domain given as an argument to perform
    integration via taking an average. Once instantiated, an
    Integrator is intended to be used as a Callable which
    then performs integration.

    Parameters
    ----------
    domain
        A domain class. Needs to provide the two methods
        domain.measure() and
        domain.deterministic_integration_points(<int>).
    N: int = 50
        The number or density of integration points drawn.
        How many points exactly are being used depends on the
        implementation of the concrete domain used.
    K: int or None
        If K is not None then this splits the integration points
        in K chunks that are processed sequentially instead of
        parallel. Can be used when GPU memory is limited.

    """

    def __init__(self, domain, N=50, K=None):
        self._domain = domain
        self._x = domain.deterministic_integration_points(N)
        self._K = K

        if K is not None:
            splits = [i * len(self._x) // K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)

    def __call__(self, f):
        mean = 0
        if self._K is not None:
            means = []
            for x in self.x_split:
                means.append(jnp.mean(f(x), axis=0))

            mean = jnp.mean(jnp.array(means), axis=0)

        else:
            y = f(self._x)
            mean = jnp.mean(y, axis=0)

        return self._domain.measure() * mean
