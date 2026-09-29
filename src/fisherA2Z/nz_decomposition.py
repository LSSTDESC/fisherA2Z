"""Decompose an arbitrary tomographic n(z) into the A2Z photo-z parameterization.

The FisherA2Z photo-z error model describes each tomographic bin as a Gaussian
core plus a fixed-shape outlier population::

    n_i(z; dz_i, sigma_i, f_i) = (1 - f_i) * N(z; mu_i + dz_i, sigma_i)
                                 + f_i * n_out,i(z)

This module solves the *inverse* problem: given a measured n(z) it recovers
``(mu_i, sigma_i, f_i)`` and the outlier shape ``n_out,i(z)``, and given an
ensemble of n(z) realizations it turns the scatter of those numbers into a
prior for the Fisher forecast.

Nothing here imports CCL or :mod:`fisherA2Z.fisher` -- it is pure numpy/scipy.

When to use it
--------------
This decomposition earns its keep when a bin really does have a separable
outlier population: a distinct island of catastrophic redshifts, well away from
the core.  When it does not -- a smooth, mildly skewed n(z) whose "outliers"
are the shoulders of the core itself -- the core/residual split is ill-posed,
``f_out`` comes out large and meaningless, and its realization scatter is set
by pixel noise rather than by photo-z calibration.  For that (common) case use
:mod:`fisherA2Z.nz_shift_stretch`, which keeps the measured n(z) as its own
template and parameterizes it with a shift and a stretch.  It is the default
model in :class:`fisherA2Z.fisher_flex.FisherFlex`.

Algorithm
---------
For a fixed ``(mu, sigma)`` the non-negativity constraint on the residual,
``A * G(z) <= n(z)``, is a set of linear upper bounds on the scalar amplitude
``A``, so the largest feasible amplitude is exactly ``min_z n(z) / G(z)``.
That profiles ``A`` out analytically and leaves a two-dimensional search over
``(mu, log sigma)``, which is done with Nelder-Mead (the ``min_z`` makes the
objective piecewise smooth, so gradient-based line searches misbehave).

Because real n(z) arrays contain exact zeros, the naive ``min_z n/G`` collapses
to ``A = 0``.  A *relative* floor ``eta_rel * max(n)`` in the numerator, paired
with the support mask ``G > tau * max(G)``, removes the collapse without
biasing the amplitude (see :func:`profile_amplitude`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "NzDecomposition",
    "TomoDecomposition",
    "gaussian",
    "profile_amplitude",
    "decompose_nz",
    "decompose_tomography",
    "decompose_realizations",
    "photoz_prior_from_realizations",
    "build_model_nz",
    "orient_realizations",
    "smooth_nz",
]

#: Below this outlier fraction a bin is treated as pure Gaussian.
F_OUT_MIN = 1e-3

#: Default relative floor added to n(z) when profiling the amplitude.
#:
#: The floor only has to be large enough to stop ``min_z n/G`` collapsing to
#: zero on grids that contain exact zeros; beyond that it is pure bias.  A scan
#: over ``eta_rel`` from 1e-3 down to 0 on the HSC Y1 bins and on a synthetic
#: mixture moved the recovered ``f_out`` by less than 1% while the mass of the
#: (unphysical) negative residual fell by four orders of magnitude, so the
#: default sits well below the value at which the floor starts to bind.
ETA_REL = 1e-5

#: Default support threshold on the Gaussian, as a fraction of its peak.
#:
#: Kept equal to :data:`ETA_REL` on purpose: the floor's protection against the
#: hard-zeros trap relies on ``eta_rel / tau`` sitting at the natural amplitude
#: scale (see :func:`profile_amplitude`).  Lowering one without the other would
#: either reintroduce the bias or let a single zero-valued pixel inside the core
#: collapse the amplitude.
TAU = 1e-5

PARAM_NAMES = ("mu", "sigma", "f_out")


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------


@dataclass
class NzDecomposition:
    """Result of decomposing a single tomographic bin."""

    mu: float
    sigma: float
    f_out: float
    amplitude: float
    z: np.ndarray
    nz: np.ndarray
    nz_out: np.ndarray
    degenerate: bool = False
    n_restarts: int = 0
    objective: float = np.nan

    @property
    def params(self) -> np.ndarray:
        """``(mu, sigma, f_out)`` as a length-3 array."""
        return np.array([self.mu, self.sigma, self.f_out])

    def core(self) -> np.ndarray:
        """The Gaussian core evaluated on ``z``, normalized on the grid."""
        g = gaussian(self.z, self.mu, self.sigma)
        norm = np.trapz(g, self.z)
        return g / norm if norm > 0 else g

    def model(self, dz: float = 0.0, sigma: float | None = None,
              f_out: float | None = None) -> np.ndarray:
        """Reassemble the model n(z), optionally with shifted parameters."""
        return build_model_nz(
            self.z,
            self.mu + dz,
            self.sigma if sigma is None else sigma,
            self.f_out if f_out is None else f_out,
            self.nz_out,
        )


@dataclass
class TomoDecomposition:
    """Decomposition of a full set of tomographic bins."""

    z: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    f_out: np.ndarray
    nz_out: np.ndarray
    nz_fid: np.ndarray
    degenerate: np.ndarray
    bins: list = field(default_factory=list)

    @property
    def n_tomo(self) -> int:
        return len(self.mu)

    @property
    def params(self) -> np.ndarray:
        """``(n_tomo, 3)`` array of ``(mu, sigma, f_out)``."""
        return np.column_stack([self.mu, self.sigma, self.f_out])


# --------------------------------------------------------------------------
# core algorithm
# --------------------------------------------------------------------------


def gaussian(z, mu, sigma):
    """Unit-area Gaussian pdf evaluated on ``z``."""
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def profile_amplitude(mu, sigma, z, nz, eta_rel=ETA_REL, tau=TAU, quantile=0.0):
    """Largest ``A`` with ``A * G(z) <= n(z) + floor`` on the Gaussian's support.

    Parameters
    ----------
    mu, sigma : float
        Gaussian core parameters.
    z, nz : ndarray
        Redshift grid and n(z) values (need not be normalized).
    eta_rel : float
        Relative floor added to the numerator, in units of ``max(nz)``.  This
        is what keeps exact zeros in ``nz`` from forcing ``A = 0``.  At a
        redshift where ``nz = 0`` and ``G/max(G) = tau`` the ratio becomes
        ``eta_rel * max(nz) / (tau * max(G))``, so with ``eta_rel == tau`` the
        floor sits at the natural amplitude scale and stops binding.
    tau : float
        Support threshold: pixels with ``G <= tau * max(G)`` are ignored
        (``|z - mu| > 4.8 sigma`` at the default ``tau = 1e-5``).  Pixels
        further out cannot bind the minimum anyway -- their ``G`` is tiny, so
        the ratio is huge -- so ``tau`` really just sets *where* the floor's
        protection kicks in, and it must track ``eta_rel``.
    quantile : float
        If > 0, use this quantile of the ratio instead of its strict minimum.
        A small value (~0.01) makes the fit far less sensitive to pixel noise
        in n(z) realizations, at the cost of a slightly negative residual.

    Returns
    -------
    A : float
    G : ndarray
        The Gaussian evaluated on ``z`` (returned to avoid recomputing it).
    """
    G = gaussian(z, mu, sigma)
    gmax = G.max()
    if not np.isfinite(gmax) or gmax <= 0:
        return 0.0, G
    support = G > tau * gmax
    if not np.any(support):
        return 0.0, G
    ratio = (nz[support] + eta_rel * nz.max()) / G[support]
    if quantile > 0:
        A = float(np.quantile(ratio, quantile))
    else:
        A = float(ratio.min())
    return max(A, 0.0), G


def _objective(x, z, nz, eta_rel, tau, quantile):
    mu, log_sigma = x
    sigma = np.exp(log_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.inf
    A, G = profile_amplitude(mu, sigma, z, nz, eta_rel, tau, quantile)
    resid = nz - A * G
    return float(np.trapz(resid * resid, z))


def decompose_nz(z, nz, *, eta_rel=ETA_REL, tau=TAU, quantile=0.0,
                 f_out_min=F_OUT_MIN, x0=None, sigma_starts=(1.0, 0.6, 0.35),
                 xatol=1e-6, fatol=1e-14, maxiter=400):
    """Decompose one n(z) into a Gaussian core plus a non-negative residual.

    Parameters
    ----------
    z, nz : ndarray
        Redshift grid and n(z).  ``nz`` is renormalized internally; negative
        values are clipped to zero with a warning.
    x0 : (mu, sigma) tuple, optional
        Warm start.  When given, a single Nelder-Mead run is done from it
        instead of the multi-start sweep -- roughly 4x faster, which matters
        when fitting thousands of realizations.
    sigma_starts : sequence of float
        Multipliers on the moment-based sigma used as restarts.  The moment
        sigma is biased high whenever an outlier island is present, so a lone
        start there can settle into a too-wide local optimum.

    Returns
    -------
    NzDecomposition
    """
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    if z.ndim != 1 or nz.shape != z.shape:
        raise ValueError(
            f"z and nz must be 1-D arrays of the same shape, got {z.shape} and {nz.shape}"
        )
    if np.any(nz < 0):
        warnings.warn("n(z) has negative values; clipping to zero before decomposing.")
        nz = np.clip(nz, 0.0, None)
    norm = np.trapz(nz, z)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("n(z) has zero or non-finite integral; cannot decompose.")
    nz = nz / norm

    mu0 = float(np.trapz(z * nz, z))
    var0 = float(np.trapz((z - mu0) ** 2 * nz, z))
    sigma0 = np.sqrt(max(var0, 1e-12))

    if x0 is not None:
        starts = [(float(x0[0]), float(x0[1]))]
    else:
        starts = [(mu0, sigma0 * s) for s in sigma_starts]

    best, best_x = np.inf, None
    for mu_s, sig_s in starts:
        res = minimize(
            _objective,
            x0=np.array([mu_s, np.log(max(sig_s, 1e-6))]),
            args=(z, nz, eta_rel, tau, quantile),
            method="Nelder-Mead",
            options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter),
        )
        if res.fun < best:
            best, best_x = float(res.fun), res.x

    mu = float(best_x[0])
    sigma = float(np.exp(best_x[1]))
    A, G = profile_amplitude(mu, sigma, z, nz, eta_rel, tau, quantile)

    resid = np.clip(nz - A * G, 0.0, None)
    resid_mass = float(np.trapz(resid, z))
    f_out = float(np.clip(resid_mass, 0.0, 1.0))

    degenerate = f_out < f_out_min
    if degenerate:
        # A near-Gaussian bin: the residual carries no usable shape.  Keep the
        # (renormalized) input n(z) as the outlier template so that the f_out
        # derivative stays finite and well defined, and pin f_out to zero.
        f_out = 0.0
        nz_out = nz.copy()
    else:
        nz_out = resid / resid_mass

    return NzDecomposition(
        mu=mu,
        sigma=sigma,
        f_out=f_out,
        amplitude=A,
        z=z,
        nz=nz,
        nz_out=nz_out,
        degenerate=bool(degenerate),
        n_restarts=len(starts),
        objective=best,
    )


def build_model_nz(z, mu, sigma, f_out, nz_out, renormalize=True):
    """Assemble ``(1-f) N(mu, sigma) + f n_out`` on the grid ``z``.

    The Gaussian truncated to a finite grid integrates to slightly less than
    one, and that deficit *changes* with ``mu`` and ``sigma``.  Renormalizing
    keeps the ``(1-f)/f`` split meaning what it says and stops the ``f``
    derivative from picking up a spurious normalization component.
    """
    z = np.asarray(z, dtype=float)
    g = gaussian(z, mu, sigma)
    gnorm = np.trapz(g, z)
    if gnorm > 0:
        g = g / gnorm
    nz = (1.0 - f_out) * g + f_out * np.asarray(nz_out, dtype=float)
    nz = np.clip(nz, 0.0, None)
    if renormalize:
        total = np.trapz(nz, z)

        if total > 0:
            nz = nz / total
    return nz


# --------------------------------------------------------------------------
# tomography / realizations
# --------------------------------------------------------------------------


def decompose_tomography(z, nz_array, **kwargs):
    """Decompose every row of ``nz_array`` (shape ``(n_tomo, n_z)``)."""
    z = np.asarray(z, dtype=float)
    nz_array = np.atleast_2d(np.asarray(nz_array, dtype=float))
    if nz_array.shape[1] != z.size:
        raise ValueError(
            f"nz_array has {nz_array.shape[1]} redshift columns but z has {z.size} points."
        )
    bins = [decompose_nz(z, row, **kwargs) for row in nz_array]
    return TomoDecomposition(
        z=z,
        mu=np.array([b.mu for b in bins]),
        sigma=np.array([b.sigma for b in bins]),
        f_out=np.array([b.f_out for b in bins]),
        nz_out=np.array([b.nz_out for b in bins]),
        nz_fid=np.array([b.nz for b in bins]),
        degenerate=np.array([b.degenerate for b in bins]),
        bins=bins,
    )


def orient_realizations(realizations, n_tomo, realizations_axis=None):
    """Return realizations as ``(n_real, n_tomo, n_z)``.

    Accepts either ``(n_real, n_tomo, n_z)`` or ``(n_tomo, n_real, n_z)``.  If
    the leading two axes have the same length the orientation is genuinely
    ambiguous, so an explicit ``realizations_axis`` is required rather than a
    silent guess.
    """
    arr = np.asarray(realizations, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"realizations must be 3-D (n_real, n_tomo, n_z) or (n_tomo, n_real, n_z), "
            f"got shape {arr.shape}."
        )
    if realizations_axis is not None:
        if realizations_axis not in (0, 1):
            raise ValueError("realizations_axis must be 0 or 1.")
        return arr if realizations_axis == 0 else arr.transpose(1, 0, 2)

    a0, a1 = arr.shape[0], arr.shape[1]
    if a0 == a1:
        raise ValueError(
            f"realizations has shape {arr.shape}; axes 0 and 1 are both {a0}, so the "
            "realization axis is ambiguous. Pass realizations_axis=0 (n_real first) "
            "or realizations_axis=1 (n_tomo first)."
        )
    if a1 == n_tomo:
        return arr
    if a0 == n_tomo:
        return arr.transpose(1, 0, 2)
    raise ValueError(
        f"neither axis of realizations {arr.shape} has length n_tomo={n_tomo}."
    )


def smooth_nz(z, nz_array, sigma_z):
    """Gaussian-smooth n(z) along the redshift axis, with ``sigma_z`` in z units."""
    from scipy.ndimage import gaussian_filter1d

    z = np.asarray(z, dtype=float)
    dz = np.diff(z)
    if not np.allclose(dz, dz[0], rtol=1e-6):
        warnings.warn(
            "smooth_realizations assumes a uniform z grid; using the mean spacing."
        )
    sigma_pix = float(sigma_z) / float(np.mean(dz))
    if sigma_pix <= 0:
        return np.asarray(nz_array, dtype=float)
    out = gaussian_filter1d(np.asarray(nz_array, dtype=float), sigma_pix,
                            axis=-1, mode="nearest")
    return np.clip(out, 0.0, None)


def noise_ratio(z, nz_fid, realizations):
    """Median curvature of the realizations relative to the central n(z).

    A value much larger than one means the realizations are far noisier than
    the central n(z), which biases the ``min_z`` amplitude low and inflates the
    recovered ``f_out`` prior.
    """
    def curvature(arr):
        d2 = np.diff(np.asarray(arr, dtype=float), n=2, axis=-1)
        scale = np.max(np.abs(arr), axis=-1, keepdims=True)
        scale = np.where(scale > 0, scale, 1.0)
        return np.median(np.abs(d2) / scale, axis=-1)

    fid = np.median(curvature(nz_fid))
    real = np.median(curvature(realizations.reshape(-1, realizations.shape[-1])))
    return float(real / fid) if fid > 0 else np.inf


def decompose_realizations(z, realizations, warm_start=None, progress=False,
                           **kwargs):
    """Fit every realization of every bin.

    Parameters
    ----------
    realizations : ndarray, shape (n_real, n_tomo, n_z)
        Already oriented (see :func:`orient_realizations`).
    warm_start : ndarray, shape (n_tomo, 2), optional
        Per-bin ``(mu, sigma)`` from the central fit.  Passing it switches each
        fit to a single Nelder-Mead run from that point (~4x faster, no loss of
        quality on realistic n(z)).

    Returns
    -------
    params : ndarray, shape (n_real, n_tomo, 3)
        ``(mu, sigma, f_out)`` for every realization and bin.
    """
    z = np.asarray(z, dtype=float)
    realizations = np.asarray(realizations, dtype=float)
    n_real, n_tomo, _ = realizations.shape
    params = np.empty((n_real, n_tomo, 3))
    for r in range(n_real):
        if progress and r % max(1, n_real // 10) == 0:
            print(f"  decomposing realization {r}/{n_real}", flush=True)
        for b in range(n_tomo):
            x0 = None if warm_start is None else tuple(warm_start[b])
            try:
                fit = decompose_nz(z, realizations[r, b], x0=x0, **kwargs)
                params[r, b] = fit.params
            except ValueError:
                params[r, b] = np.nan
    return params


def photoz_prior_from_realizations(params, mode="full", min_sigma=1e-6):
    """Per-bin prior covariance of the photo-z nuisances from realization fits.

    ``params`` is a ``(n_real, n_tomo, n_par)`` array: ``(mu, sigma, f_out)``
    for the Gaussian-plus-residual decomposition, ``(dz, stretch)`` for the
    shift-and-stretch model of :mod:`fisherA2Z.nz_shift_stretch`.  Because the
    forecast parameter is the *shift* ``dz`` rather than ``mu`` itself, and the
    shift has the same scatter as ``mu``, the returned covariance can be used
    directly in either case.

    Returns
    -------
    cov : ndarray, shape (n_tomo, n_par, n_par)
        ``mode='diag'`` zeroes the off-diagonal terms.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 3:
        raise ValueError(
            f"params must have shape (n_real, n_tomo, n_par), got {params.shape}"
        )
    n_tomo, n_par = params.shape[1], params.shape[2]
    cov = np.empty((n_tomo, n_par, n_par))
    for b in range(n_tomo):
        block = params[:, b, :]
        good = np.all(np.isfinite(block), axis=1)
        if good.sum() <= n_par:
            raise ValueError(
                f"bin {b}: only {good.sum()} usable realization fits; cannot form a prior."
            )
        if good.sum() < len(good):
            warnings.warn(
                f"bin {b}: dropped {len(good) - good.sum()} realizations whose fit failed."
            )
        c = np.atleast_2d(np.cov(block[good], rowvar=False))
        # Guard against a degenerate direction (e.g. f_out identically zero).
        d = np.diag(c).copy()
        d = np.where(d > min_sigma ** 2, d, min_sigma ** 2)
        np.fill_diagonal(c, d)
        cov[b] = np.diag(d) if mode == "diag" else c
    return cov
