"""Shared fixtures for the FisherFlex test suite.

Everything here is deliberately tiny: a 60-point redshift grid, 3 multipoles,
2 source bins and 3 lens bins.  The CCL-touching tests use
``transfer_function='eisenstein_hu'`` so no Boltzmann code is ever called.
"""

import numpy as np
import pytest

from fisherA2Z import flex_layout as fl
from fisherA2Z import nz_decomposition as nzd


@pytest.fixture(scope="session")
def z_grid():
    return np.linspace(0.02, 3.0, 60)


@pytest.fixture(scope="session")
def toy_nz(z_grid):
    """Two source bins, each a Gaussian core plus a distinct outlier island."""
    z = z_grid
    nz = np.array([
        0.85 * nzd.gaussian(z, 0.6, 0.12) + 0.15 * nzd.gaussian(z, 1.8, 0.25),
        0.90 * nzd.gaussian(z, 1.1, 0.15) + 0.10 * nzd.gaussian(z, 2.4, 0.20),
    ])
    return nz / np.trapz(nz, z, axis=1)[:, None]


@pytest.fixture(scope="session")
def toy_realizations(z_grid, toy_nz):
    """50 realizations with jitter injected into mu, sigma and f_out."""
    rng = np.random.default_rng(1234)
    z = z_grid
    out = np.empty((50, 2, z.size))
    for r in range(50):
        for b, (mu, sig, f, zo) in enumerate(
            [(0.6, 0.12, 0.15, 1.8), (1.1, 0.15, 0.10, 2.4)]
        ):
            m = mu + rng.normal(0, 0.01)
            s = sig * (1 + rng.normal(0, 0.03))
            ff = np.clip(f + rng.normal(0, 0.01), 0.01, 0.5)
            n = (1 - ff) * nzd.gaussian(z, m, s) + ff * nzd.gaussian(z, zo, 0.25)
            out[r, b] = n / np.trapz(n, z)
    return out


@pytest.fixture(scope="session")
def toy_layout():
    """3 lens bins, 2 source bins, all lens-source pairs allowed."""
    pairs = [(l, s) for l in range(3) for s in range(2)]
    return fl.build_layout(3, 2, pairs, mode="3x2pt")


@pytest.fixture(scope="session")
def toy_ell():
    return np.array([100.0, 500.0, 2000.0])


@pytest.fixture(scope="session")
def toy_cl_all(toy_layout, toy_ell):
    """Smooth, positive-definite fake spectra for every leg pair."""
    rng = np.random.default_rng(7)
    n = toy_layout.n_legs
    amp = rng.uniform(0.5, 1.5, size=n)
    shape = (toy_ell / 100.0) ** -1.2
    # Outer product in leg space keeps the ell-by-ell matrix PSD.
    return amp[:, None, None] * amp[None, :, None] * shape[None, None, :] * 1e-8


@pytest.fixture(scope="session")
def toy_cov(toy_cl_all, toy_layout, toy_ell):
    noise = fl.noise_power(3, 2, np.full(3, 4.0), np.full(2, 10.0), 0.26)
    nmodes = np.array([1e4, 5e4, 2e5])
    return fl.gaussian_covariance(toy_cl_all, noise, toy_layout, nmodes)


@pytest.fixture(scope="session")
def tiny_cosmo():
    import pyccl as ccl

    return ccl.Cosmology(
        Omega_c=0.2666, Omega_b=0.049, h=0.6727, sigma8=0.831, n_s=0.9645,
        transfer_function="eisenstein_hu",
    )
