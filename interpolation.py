"""Periodic linear interpolation along one axis of an N-d array.

For each index tuple, gathers along ``axis`` at a departure point:

    out[..., i, ...] = (1-a) * f[i0] + a * f[i1]

with ``i0 = floor((x_star - X[0]) / dx)`` and ``a`` the fractional cell offset.
This is the interpolation half of a backward semi-Lagrangian step: given the
foot of a characteristic, return the field value there.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

Array = jax.Array

__all__ = ["interp1_linear_x"]


@partial(jax.jit, static_argnames=("axis", "periodic"))
def interp1_linear_x(
    f: Array,
    x_star: ArrayLike,  # Departure points. .shape must be broadcastable to f.shape
    X: Array,  # Coordinate grid corresponding to f[axis]
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
