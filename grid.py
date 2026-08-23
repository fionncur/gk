"""Fixed simulation-domain description, shared across interpolation and advection.

Bundles the coordinate arrays and per-axis conventions (which axis is x or y, and
whether the domain is periodic) that stay fixed for an entire run. The magnetic
moment ``MU`` is an optional third coordinate and purely spatial uses may omit it.
``MU`` need not be uniformly spaced -- quadrature nodes (e.g. Gauss-Laguerre)
are expected, with ``WMU`` holding the matching weights so that mu integrals
are ``sum(WMU * integrand)``. ``periodic`` refers to x/y only, mu is never
advected or wrapped.

Registered as a JAX pytree: ``X``/``Y``/``MU``/``WMU`` are traced leaves,
``axis_x``/``axis_y``/``axis_mu``/``periodic`` are static aux data baked into
the jit cache key.
"""

from __future__ import annotations

import jax

Array = jax.Array

__all__ = ["Grid"]


class Grid:
    def __init__(
        self,
        X: Array,
        Y: Array,
        MU: Array | None = None,
        WMU: Array | None = None,
        *,
        axis_x: int = 0,
        axis_y: int = 1,
        axis_mu: int = 2,
        periodic: bool = True,
    ):
        self.X = X
        self.Y = Y
        self.MU = MU
        self.WMU = WMU
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_mu = axis_mu
        self.periodic = periodic

    @property
    def Nx(self) -> int:
        return self.X.shape[0]

    @property
    def Ny(self) -> int:
        return self.Y.shape[0]

    @property
    def dx(self) -> Array:
        return self.X[1] - self.X[0]

    @property
    def dy(self) -> Array:
        return self.Y[1] - self.Y[0]

    @property
    def x0(self) -> Array:
        return self.X[0]

    @property
    def y0(self) -> Array:
        return self.Y[0]

    @property
    def Lx(self) -> Array:
        return self.dx * self.Nx

    @property
    def Ly(self) -> Array:
        return self.dy * self.Ny

    @property
    def Nmu(self) -> int:
        return self.MU.shape[0]


def _grid_flatten(g: Grid):
    children = (g.X, g.Y, g.MU, g.WMU)
    aux = (g.axis_x, g.axis_y, g.axis_mu, g.periodic)
    return children, aux


def _grid_unflatten(aux, children) -> Grid:
    axis_x, axis_y, axis_mu, periodic = aux
    X, Y, MU, WMU = children
    return Grid(
        X, Y, MU, WMU, axis_x=axis_x, axis_y=axis_y, axis_mu=axis_mu, periodic=periodic
    )


jax.tree_util.register_pytree_node(Grid, _grid_flatten, _grid_unflatten)
