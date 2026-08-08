"""Backward characteristic tracing for the semi-Lagrangian advection step.

Given a velocity field on the grid, traces each arrival (grid) point
backward over one time step to its departure point, then advects a field
by interpolating it there. Tracing and interpolation are independent
choices: :data:`TRACE_METHODS` selects how the departure point itself is
found; the interpolant used to sample the field there is a separate
choice, from :data:`interpolation.INTERPOLATORS_2D`.
"""

from __future__ import annotations

from functools import partial

import jax
from jax.typing import ArrayLike

from grid import Grid
from interpolation import INTERPOLATORS_2D

Array = jax.Array

__all__ = ["trace_characteristics", "advect", "TRACE_METHODS"]


def _grid_shape(n: int, axis: int, ndim: int) -> tuple[int, ...]:
    shape = [1] * ndim
    shape[axis] = n
    return tuple(shape)


def _trace_euler(ux, uy, x_grid, y_grid, dt):
    return x_grid - ux * dt, y_grid - uy * dt


# Each entry maps (ux, uy, x_grid, y_grid, dt) to a departure point
# (x_star, y_star).
TRACE_METHODS = {
    "euler": _trace_euler,
}


@partial(jax.jit, static_argnames=("method",))
def trace_characteristics(
    ux: Array,
    uy: Array,
    grid: Grid,
    dt: float | ArrayLike,
    *,
    method: str = "euler",
) -> tuple[Array, Array]:
    """Trace each grid point backward one time step to its departure point."""
    ndim = ux.ndim
    x_grid = grid.X.reshape(_grid_shape(grid.Nx, grid.axis_x, ndim))
    y_grid = grid.Y.reshape(_grid_shape(grid.Ny, grid.axis_y, ndim))
    return TRACE_METHODS[method](ux, uy, x_grid, y_grid, dt)


@partial(jax.jit, static_argnames=("trace_method", "interp_method"))
def advect(
    f: Array,
    ux: Array,
    uy: Array,
    grid: Grid,
    dt: float | ArrayLike,
    *,
    trace_method: str = "euler",
    interp_method: str = "bilinear",
    fill_value: float | ArrayLike = 0.0,
) -> Array:
    """Advect ``f`` by tracing characteristics, then interpolating there."""
    x_star, y_star = trace_characteristics(ux, uy, grid, dt, method=trace_method)
    interp = INTERPOLATORS_2D[interp_method]
    return interp(f, x_star, y_star, grid, fill_value=fill_value)
