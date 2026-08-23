"""Run configuration: physical parameters and domain construction from YAML.

All run-defining numbers, i.e. the plasma parameters of the adiabatic-electron,
single-ion model and the grid sizes, live in a YAML file
with ``plasma:`` and ``grid:`` sections; :func:`load_config` parses it and
builds the typed objects the solver consumes. ``PlasmaParams`` is a NamedTuple
and hence automatically a JAX pytree, so it can be passed through jitted
functions with every field traced.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.special import roots_laguerre

from grid import Grid

Array = jax.Array

__all__ = ["PlasmaParams", "gauss_laguerre_mu", "load_config"]


class PlasmaParams(NamedTuple):
    n0: float = 1.0  # Background density (both species, quasineutral)
    Te: float = 1.0  # Electron temperature
    Ti: float = 1.0  # Ion temperature
    qi: float = 1.0  # Ion charge
    mi: float = 1.0  # Ion mass
    B: float = 1.0  # Background magnetic field B0


def gauss_laguerre_mu(Nmu: int, params: PlasmaParams) -> tuple[Array, Array]:
    """Gauss-Laguerre nodes and weights for mu integrals of Maxwellian-type
    integrands.

    Returns the mu grid ``(MU, WMU)`` such that, for an integrand ``H(mu)``
    that decays like the Maxwellian ``exp(-mu B/Ti)``, ``sum(WMU * H(MU))``
    approximates ``int_0^inf H(mu) dmu`` exponentially accurately. The nodes
    are the standard Laguerre roots ``s_j`` rescaled to physical units,
    ``MU = s_j Ti/B``; the weights carry the same ``Ti/B`` Jacobian and an
    ``e^{s_j}`` factor, since ``H`` supplies its own Maxwellian decay.
    """
    s, omega = roots_laguerre(Nmu)
    scale = params.Ti / params.B
    return jnp.asarray(s * scale), jnp.asarray(scale * omega * np.exp(s))


def load_config(path: str) -> tuple[PlasmaParams, Grid]:
    """Parse a YAML run configuration into ``(PlasmaParams, Grid)``.

    Expects ``plasma:`` and ``grid:`` (Nx, Ny, Lx, Ly, Nmu) sections.
    X and Y are uniform periodic axes (endpoint excluded, so
    ``dx * Nx == Lx``); the mu axis is Gauss-Laguerre.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = PlasmaParams(**cfg["plasma"])
    g = cfg["grid"]
    X = jnp.linspace(0.0, g["Lx"], g["Nx"], endpoint=False)
    Y = jnp.linspace(0.0, g["Ly"], g["Ny"], endpoint=False)
    MU, WMU = gauss_laguerre_mu(g["Nmu"], params)
    return params, Grid(X, Y, MU, WMU)
