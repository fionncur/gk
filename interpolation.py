"""Periodic linear interpolation along one axis of an N-d array.

There are as many trajectories as arrival points (the N points of the
uniform grid). Each is traced backward under the flow map for one time
step, landing at a departure point ``x_star`` in some grid cell; the map
need not be injective or surjective, so cells may receive zero, one, or
several departure points. For each departure point this locates the
bounding grid indices ``i0`` (left) and ``i1`` (right), gathers
``f[i0]`` and ``f[i1]``, and blends them with the local coordinate ``a``:

    out[..., i, ...] = (1-a) * f[i0] + a * f[i1]

with ``i0 = floor((x_star - X[0]) / dx)`` and ``a`` the fractional cell
offset. This is the interpolation half of a backward semi-Lagrangian
step: given the foot of a characteristic, return the field value there.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from grid import Grid

Array = jax.Array

__all__ = ["interp1_linear_x", "interp2_bilinear_xy", "INTERPOLATORS_2D"]


@partial(jax.jit, static_argnames=("axis", "periodic"))
def interp1_linear_x(
    f: Array,
    x_star: ArrayLike,  # Departure points. .shape must be broadcastable to f.shape
    X: Array,  # Coordinate array corresponding to f[axis]
    *,
    axis: int = 0,
    periodic: bool = True,
    fill_value: float | ArrayLike = 0.0,
) -> Array:

    N = f.shape[axis]
    dx = X[1] - X[0]
    x0 = X[0]
    L = dx * N

    if periodic:
        xw = (x_star - x0) % L + x0
        valid = None
    else:
        xw = x_star
        valid = (xw >= x0) & (xw < x0 + L)

    s = (xw - x0) / dx
    i0 = jnp.floor(s).astype(jnp.int32)
    a = s - i0

    if periodic:
        i0 = i0 % N
        i1 = (i0 + 1) % N
    else:
        i0 = jnp.clip(i0, 0, N - 1)
        i1 = jnp.clip(i0 + 1, 0, N - 1)

    i0 = jnp.broadcast_to(i0, f.shape)
    i1 = jnp.broadcast_to(i1, f.shape)
    a = jnp.broadcast_to(a, f.shape)

    f0 = jnp.take_along_axis(f, i0, axis=axis)
    f1 = jnp.take_along_axis(f, i1, axis=axis)
    out = (1.0 - a) * f0 + a * f1

    if not periodic:
        valid = jnp.broadcast_to(valid, f.shape)
        out = jnp.where(valid, out, fill_value)

    return out


@jax.jit
def interp2_bilinear_xy(
    f: Array,
    x_star: ArrayLike,  # Departure x-coords. .shape must be broadcastable to f.shape
    y_star: ArrayLike,  # Departure y-coords. .shape must be broadcastable to f.shape
    grid: Grid,
    *,
    fill_value: float | ArrayLike = 0.0,
) -> Array:
    """Periodic bilinear interpolation over two axes of an N-d array.

    Generalizes :func:`interp1_linear_x` to a genuine 2D departure point
    ``(x_star, y_star)`` in the plane spanned by ``grid.axis_x`` and
    ``grid.axis_y``, bounded by grid indices ``i0x``/``i1x`` and
    ``i0y``/``i1y`` with local coordinates ``ax``, ``ay``. The four
    bounding corners are gathered jointly from ``f``, in a single pass,
    and blended:

        out = (1-ax)(1-ay) * f[i0x,i0y] + ax(1-ay) * f[i1x,i0y]
            + (1-ax)ay * f[i0x,i1y] + ax*ay * f[i1x,i1y]

    with ``i0x = floor((x_star - grid.X[0]) / grid.dx)`` and likewise for
    ``i0y``. Any axes other than ``grid.axis_x``/``grid.axis_y`` (e.g. a
    velocity-space axis) pass through unchanged, each of their slices
    interpolated independently.
    """
    axis_x, axis_y, periodic = grid.axis_x, grid.axis_y, grid.periodic
    Nx = f.shape[axis_x]
    Ny = f.shape[axis_y]
    dx = grid.dx
    dy = grid.dy
    x0 = grid.x0
    y0 = grid.y0
    Lx = dx * Nx
    Ly = dy * Ny

    if periodic:
        xw = (x_star - x0) % Lx + x0
        yw = (y_star - y0) % Ly + y0
        valid = None
    else:
        xw = x_star
        yw = y_star
        valid = (xw >= x0) & (xw < x0 + Lx) & (yw >= y0) & (yw < y0 + Ly)

    sx = (xw - x0) / dx
    sy = (yw - y0) / dy
    i0x = jnp.floor(sx).astype(jnp.int32)
    i0y = jnp.floor(sy).astype(jnp.int32)
    ax = sx - i0x
    ay = sy - i0y

    if periodic:
        i0x, i1x = i0x % Nx, (i0x + 1) % Nx
        i0y, i1y = i0y % Ny, (i0y + 1) % Ny
    else:
        i0x = jnp.clip(i0x, 0, Nx - 1)
        i1x = jnp.clip(i0x + 1, 0, Nx - 1)
        i0y = jnp.clip(i0y, 0, Ny - 1)
        i1y = jnp.clip(i0y + 1, 0, Ny - 1)

    i0x, i1x, i0y, i1y, ax, ay = (
        jnp.broadcast_to(v, f.shape) for v in (i0x, i1x, i0y, i1y, ax, ay)
    )

    # The operators i0x[p,q], i0y[p,q], etc. in general depend on both coordinates
    # of the grid points (p,q), so they cannot be factored as a sequence of 1D
    # gathers (take_along_axis): each one holds every non-gathered axis fixed at
    # the output position (p,q), so a second gather along axis_y would evaluate
    # axis_x's index at the already-shifted y-position instead of the true q.
    # Merging axis_x and axis_y into one flat axis lets a single gather apply the
    # joint index i0x*Ny + i0y directly, computed from the true (p,q) throughout.

    def merge(v):
        m = jnp.moveaxis(v, (axis_x, axis_y), (-2, -1))
        return m.reshape(m.shape[:-2] + (Nx * Ny,))

    f_flat = merge(f)
    f00 = jnp.take_along_axis(f_flat, merge(i0x * Ny + i0y), axis=-1)
    f10 = jnp.take_along_axis(f_flat, merge(i1x * Ny + i0y), axis=-1)
    f01 = jnp.take_along_axis(f_flat, merge(i0x * Ny + i1y), axis=-1)
    f11 = jnp.take_along_axis(f_flat, merge(i1x * Ny + i1y), axis=-1)
    ax_m, ay_m = merge(ax), merge(ay)

    out_flat = (
        (1.0 - ax_m) * (1.0 - ay_m) * f00
        + ax_m * (1.0 - ay_m) * f10
        + (1.0 - ax_m) * ay_m * f01
        + ax_m * ay_m * f11
    )
    out = jnp.moveaxis(
        out_flat.reshape(out_flat.shape[:-1] + (Nx, Ny)), (-2, -1), (axis_x, axis_y)
    )

    if not periodic:
        valid = jnp.broadcast_to(valid, f.shape)
        out = jnp.where(valid, out, fill_value)

    return out


INTERPOLATORS_2D = {
    "bilinear": interp2_bilinear_xy,
}
