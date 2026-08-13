r"""Shift-and-stretch photo-z model: the tabulated n(z) is its own template.

:mod:`fisherA2Z.nz_decomposition` splits each bin into a Gaussian core plus a
non-negative residual, which buys a physically interpretable outlier fraction
-- but only when the n(z) really does have a separable outlier population.  For
a smooth, mildly skewed n(z) with no distant island (HSC Y1 is the motivating
example) the split is ill-posed: the "outlier" component is just the
non-Gaussian shoulders of the core, ``f_out`` lands at 0.3-0.4, and its
scatter across realizations is set by pixel noise in the realizations rather
than by the survey's photo-z calibration.

This module implements the simpler and far more standard alternative used by
DES, KiDS and HSC: keep the *measured* n(z) as the template and give each bin
two nuisance parameters, a shift and a stretch,

.. math::

    n_i(z;\ \delta z_i,\ s_i)\ \propto\
        n_i^{\rm obs}\!\left(z_{p,i} + \frac{z - z_{p,i} - \delta z_i}{s_i}\right),

with :math:`z_{p,i}` the fiducial mean of the bin.  The mean of the modelled
n(z) is then :math:`z_{p,i} + \delta z_i` and its width is :math:`s_i` times the
fiducial width.  At the fiducial values :math:`(\delta z_i, s_i) = (0, 1)` the
model returns the tabulated n(z) itself (up to normalization, which is not
observable), so the fiducial data vector, the covariance and the derivatives
are all built from the data rather than from a fit to it.  There is no
``f_out``, which is equivalent to fixing it at zero with a delta-function
prior.

Why moments rather than a least-squares fit
-------------------------------------------
The model is a location-scale family, so the shift and stretch of a
realization relative to the reference are *exactly* the difference of the means
and the ratio of the standard deviations -- no optimizer, no local minima.
Both are low-order moments of n(z): independent pixel noise averages down over
the grid instead of being absorbed into the parameters, which is precisely the
failure mode of the ``min_z n/G`` order statistic behind ``f_out``.  The prior
widths that come out are consequently stable against smoothing of the
realizations; see
:meth:`fisherA2Z.fisher_flex.FisherFlex.prior_stability_report`.

Accuracy note
-------------
Because the template is interpolated linearly, :math:`\partial n/\partial
\delta z` evaluated by central differences is *exactly* the centred
finite-difference estimate of :math:`-n'(z)` for any step smaller than the grid
spacing -- the step size is irrelevant, and the accuracy of the photo-z
derivatives is set by the resolution of ``z_grid`` instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MIN_STRETCH",
    "nz_moments",
    "shift_stretch_nz",
    "ShiftStretchModel",
    "build_shift_stretch_model",
    "fit_shift_stretch",
]

#: Smallest stretch the model will evaluate.  A stretch at or below zero would
#: reverse or collapse the template; the forecast never goes near it, but a
#: derivative step must not be able to produce one either.
MIN_STRETCH = 1e-3


def nz_moments(z, nz):
    """Mean and rms width of ``nz`` on the grid ``z``.

    Works on any trailing-axis stack, so ``(n_tomo, n_z)`` and
    ``(n_real, n_tomo, n_z)`` both return arrays of the leading shape.
    """
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    if nz.shape[-1] != z.size:
        raise ValueError(
            f"n(z) has {nz.shape[-1]} redshift columns but z has {z.size} points."
        )
    try:
        norm = np.trapz(nz, z, axis=-1)
    except AttributeError:
        norm = np.trapezoid(nz, z, axis=-1)
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("n(z) with a zero or non-finite integral has no moments.")
    try:
        mean = np.trapz(nz * z, z, axis=-1) / norm
        var = np.trapz(nz * (z - mean[..., None]) ** 2, z, axis=-1) / norm
    except AttributeError:
        mean = np.trapezoid(nz * z, z, axis=-1) / norm
        var = np.trapezoid(nz * (z - mean[..., None]) ** 2, z, axis=-1) / norm

    return mean, np.sqrt(np.clip(var, 0.0, None))


def _padded_template(z, nz_fid):
    """Extend the template by one zero-valued cell at each end.

    Without it the model has a *jump* at the fiducial point: ``np.interp``'s
    out-of-range fill sends the first pixel from ``n_0`` to zero the instant
    ``dz`` turns positive (and the last pixel likewise for negative ``dz``).
    A central difference across a jump of size ``eps`` returns ``eps / 2h``,
    i.e. a derivative error that *grows* as the step shrinks -- measured at 17%
    for ``h = 2.5e-4`` before this padding was added.  Ramping linearly to zero
    over one grid cell instead keeps the model continuous, and is also the
    least surprising extrapolation for an n(z) that has not quite decayed to
    zero at the edge of its grid.
    """
    if z.size < 2:
        raise ValueError("the redshift grid needs at least two points.")
    z_pad = np.concatenate(([2 * z[0] - z[1]], z, [2 * z[-1] - z[-2]]))
    n_pad = np.concatenate(([0.0], nz_fid, [0.0]))
    return z_pad, n_pad


def shift_stretch_nz(z, nz_fid, dz=0.0, stretch=1.0, z_pivot=None,
                     renormalize=True):
    """Evaluate the shift-and-stretch model of one tomographic bin.

    Parameters
    ----------
    z : ndarray
        Redshift grid; also the grid the template is tabulated on.
    nz_fid : ndarray
        The fiducial (measured) n(z) of this bin.
    dz : float
        Shift of the mean, in absolute redshift units.
    stretch : float
        Multiplicative scaling of the width about ``z_pivot``.
    z_pivot : float, optional
        Redshift the stretch is applied about.  Defaults to the mean of
        ``nz_fid``, which is what makes ``dz`` and ``stretch`` close to
        independent.
    renormalize : bool
        Rescale to unit integral on the grid.  Mass shifted off the end of the
        grid otherwise makes the normalization a function of ``dz``.
    """
    z = np.asarray(z, dtype=float)
    nz_fid = np.asarray(nz_fid, dtype=float)
    s = max(float(stretch), MIN_STRETCH)
    dz = float(dz)
    if z_pivot is None:
        z_pivot = float(nz_moments(z, nz_fid)[0])

    if dz == 0.0 and s == 1.0:
        # Short-circuit so the fiducial point is the tabulated n(z) itself
        # rather than an interpolation of it.  The interpolation nodes coincide
        # with the grid here, so this only saves ~1e-16 of round-off -- but it
        # makes "the fiducial data vector is the measured n(z)" true by
        # construction instead of true by numerical accident.  (The subsequent
        # renormalization can still move the last bit if ``nz_fid`` does not
        # integrate to exactly 1; normalization is not observable, since CCL
        # normalizes n(z) internally.)
        out = np.array(nz_fid, dtype=float, copy=True)
    else:
        zp, np_ = _padded_template(z, nz_fid)
        out = np.interp(z_pivot + (z - z_pivot - dz) / s, zp, np_) / s

    out = np.clip(out, 0.0, None)
    if renormalize:
        try:
            total = np.trapz(out, z)
        except:
            total = np.trapezoid(out, z)
        if total > 0:
            out = out / total
    return out


@dataclass
class ShiftStretchModel:
    """Fiducial shift-and-stretch template for a set of tomographic bins."""

    z: np.ndarray
    nz_fid: np.ndarray      #: (n_tomo, n_z), normalized to unit integral
    z_pivot: np.ndarray     #: (n_tomo,) fiducial mean of each bin
    sigma_fid: np.ndarray   #: (n_tomo,) fiducial rms width, diagnostic only

    @property
    def n_tomo(self) -> int:
        return self.nz_fid.shape[0]

    def model(self, i, dz=0.0, stretch=1.0, renormalize=True):
        """The n(z) of bin ``i`` at the given shift and stretch."""
        return shift_stretch_nz(self.z, self.nz_fid[i], dz, stretch,
                                self.z_pivot[i], renormalize)


def build_shift_stretch_model(z, nz_array):
    """Normalize ``nz_array`` and record its per-bin mean and width."""
    z = np.asarray(z, dtype=float)
    nz = np.atleast_2d(np.asarray(nz_array, dtype=float))
    if nz.shape[1] != z.size:
        raise ValueError(
            f"nz_array has {nz.shape[1]} redshift columns but z has {z.size} points."
        )
    if np.any(nz < 0):
        warnings.warn("n(z) has negative values; clipping to zero.")
        nz = np.clip(nz, 0.0, None)
    try:
        norm = np.trapz(nz, z, axis=1)
    except:
        norm = np.trapezoid(nz, z, axis=1)
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("every n(z) row must have a positive, finite integral.")
    nz = nz / norm[:, None]
    mean, sigma = nz_moments(z, nz)
    if np.any(sigma <= 0):
        raise ValueError("a tomographic bin has zero width; cannot stretch it.")
    return ShiftStretchModel(z=z, nz_fid=nz, z_pivot=mean, sigma_fid=sigma)


def fit_shift_stretch(z, nz_ref, realizations):
    """``(dz, stretch)`` of every realization relative to ``nz_ref``.

    Closed form -- see the module docstring: for a location-scale family the
    moment estimator *is* the fit.

    Parameters
    ----------
    nz_ref : ndarray, shape (n_tomo, n_z)
    realizations : ndarray, shape (n_real, n_tomo, n_z)
        Already oriented; see :func:`fisherA2Z.nz_decomposition.orient_realizations`.

    Returns
    -------
    ndarray, shape (n_real, n_tomo, 2)
    """
    z = np.asarray(z, dtype=float)
    nz_ref = np.atleast_2d(np.asarray(nz_ref, dtype=float))
    real = np.asarray(realizations, dtype=float)
    if real.ndim != 3 or real.shape[1] != nz_ref.shape[0]:
        raise ValueError(
            f"realizations must have shape (n_real, {nz_ref.shape[0]}, {z.size}), "
            f"got {real.shape}."
        )
    mean_ref, sigma_ref = nz_moments(z, nz_ref)
    mean_r, sigma_r = nz_moments(z, real)
    return np.stack([mean_r - mean_ref, sigma_r / sigma_ref], axis=-1)
