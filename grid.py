"""Fixed simulation-domain description, shared across interpolation and advection.

Bundles the coordinate arrays and per-axis conventions (which axis is x,
which is y, and whether the domain wraps) that stay fixed for an entire
run. 

Registered as a JAX pytree: ``X``/``Y`` are traced leaves, ``axis_x``/
``axis_y``/``periodic`` are static aux data baked into the jit cache key.
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
        *,
        axis_x: int = 0,
        axis_y: int = 1,
        periodic: bool = True,
    ):
        self.X = X
        self.Y = Y
        self.axis_x = axis_x
        self.axis_y = axis_y
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


def _grid_flatten(g: Grid):
    children = (g.X, g.Y)
    aux = (g.axis_x, g.axis_y, g.periodic)
    return children, aux


def _grid_unflatten(aux, children) -> Grid:
    axis_x, axis_y, periodic = aux
    X, Y = children
    return Grid(X, Y, axis_x=axis_x, axis_y=axis_y, periodic=periodic)


jax.tree_util.register_pytree_node(Grid, _grid_flatten, _grid_unflatten)
