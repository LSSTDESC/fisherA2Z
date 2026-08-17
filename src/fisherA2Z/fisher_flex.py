"""FisherFlex -- a flexible 3x2pt Fisher forecast.

``fisherA2Z.fisher.Fisher`` implements the LSST 3x2pt forecast published in
arXiv:2507.01374, but it is hard-wired to one analysis: exactly five source
bins built from a fixed SRD n(z), a fixed on-disk inverse covariance with the
scale cuts already baked in, and photo-z priors that the user has to derive
externally.

:class:`FisherFlex` keeps the physics of that code -- CCL tracers, NLA
intrinsic alignments weighted by a Schechter luminosity function, linear galaxy
bias, the same multipole grid, the same Fisher algebra and figure of merit --
and replaces the three rigid pieces:

1. the photo-z parameterization and its prior are *derived* from an arbitrary
   n(z) plus a set of realizations;
2. the covariance is *built* analytically from ``neff``, ``fsky`` and
   ``sigma_e`` (see :mod:`fisherA2Z.flex_layout`);
3. the number of tomographic bins is arbitrary.

Three photo-z models are available, selected with ``nz_model``:

``'shift_stretch'`` (default)
    The measured n(z) is its own template, with a shift ``zbias_i`` and a
    stretch ``zstretch_i`` per bin (:mod:`fisherA2Z.nz_shift_stretch`).  At the
    fiducial point the model *is* the tabulated n(z), so the data vector, the
    covariance and the derivatives are all built from the data rather than from
    a fit to it, and the prior follows from the mean/width scatter of the
    realizations in closed form.  There is no outlier fraction, which is
    equivalent to fixing it at zero with a delta-function prior.
``'gaussian_outlier'``
    The A2Z parameterization: a Gaussian core plus a fitted outlier template,
    with ``zbias_i``, ``zvariance_i`` and ``zoutlier_i`` per bin
    (:mod:`fisherA2Z.nz_decomposition`).  Use it when the bins have a genuinely
    separable population of catastrophic redshifts; on a smooth n(z) the
    core/residual split is ill-posed and its ``f_out`` prior is set by pixel
    noise in the realizations rather than by the survey.
``'no_uncertainty'``
    n(z) is taken as exactly known: **no photo-z parameters at all**, so
    ``nz_realizations`` is not needed and can be omitted.  This is the
    perfect-photo-z limit, and therefore the optimistic bound the other two
    models are measured against -- the difference in a constraint between this
    and ``'shift_stretch'`` is the cost of not knowing n(z).  Note it does *not*
    make the forecast insensitive to n(z): the n(z) still sets the kernels, and
    :meth:`FisherFlex.forecast_bias` still works and is at its most severe here,
    since nothing is free to absorb the systematic.

Usage is three steps::

    flex = FisherFlex(nz_source, nz_realizations, z_grid,
                      neff_source=..., fsky=..., sigma_e=...)
    flex.compute(save='deriv_cov.npz')          # slow: derivatives + covariance
    res = flex.forecast()                       # fast: scale cuts + linear algebra
    res.fom('omega_m', 'sigma_8')

Deviations from the published A2Z code
--------------------------------------
Both are user-approved fixes to inconsistencies in ``fisher.py``; set the
corresponding option to reproduce the original behaviour.

1. The intrinsic-alignment luminosity function is evaluated with the *varied*
   cosmology.  ``Fisher.getAi`` passes ``self.cosmo`` instead of its own
   ``cosmo`` argument, freezing the luminosity function at the fiducial
   cosmology inside every cosmological derivative.  Use
   ``ia_lf_cosmo='fiducial'`` for the A2Z behaviour.
2. ``d/dOmega_b`` is taken at fixed ``Omega_m`` (``Omega_c = Omega_m -
   Omega_b``).  ``Fisher.getC_ellOfOmegab_ss`` holds ``Omega_c`` fixed, so
   ``Omega_m`` moves with ``Omega_b`` -- inconsistent with its own
   ``d/dOmega_m``.  Use ``omega_b_convention='a2z'`` for the A2Z behaviour.

Limitation
----------
The covariance is **Gaussian only**.  The SRD covariance used by ``Fisher``
also contains non-Gaussian and super-sample terms, so FisherFlex figures of
merit come out optimistically high relative to the paper.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pyccl as ccl
import scipy.integrate
from scipy.interpolate import CubicSpline
from scipy.stats import norm, uniform

from . import flex_layout as fl
from . import nz_decomposition as nzd
from . import nz_shift_stretch as nzs
from .nz_decomposition import build_model_nz
from .nz_shift_stretch import shift_stretch_nz

# A2Z sets this at import of fisher.py; it is process-global and materially
# changes the lensing kernel, so set it here too -- fisher.py may never be
# imported in a FisherFlex session.
ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False

__all__ = ["FisherFlex", "FisherFlexResult", "FisherFlexBias", "Bias2D",
           "FoM", "marginalize"]

_PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

#: Bumped to 2 when the shift-and-stretch photo-z model was added: the file now
#: carries ``nz_model`` and ``pz_pivot``, and ``pz_prior_cov`` is
#: ``(n_tomo, 2, 2)`` rather than ``(n_tomo, 3, 3)`` for that model.
SCHEMA_VERSION = 2

#: Photo-z models understood by ``nz_model``.
NZ_MODELS = ("shift_stretch", "gaussian_outlier", "no_uncertainty")

#: Photo-z parameters carried per source bin, by model.
N_PZ_PER_BIN = {"shift_stretch": 2, "gaussian_outlier": 3, "no_uncertainty": 0}

#: Fiducial cosmology (Planck 2018-like), matching the A2Z paper.
DEFAULT_COSMO_KWARGS = dict(
    Omega_c=0.2666,
    Omega_b=0.049,
    h=0.6727,
    sigma8=0.831,
    n_s=0.9645,
    w0=-1.0,
    wa=0.0,
    transfer_function="eisenstein_hu",
)

#: Gaussian prior widths from Table 1 of arXiv:2507.01374.
COSMO_PRIOR_SIGMA = dict(
    omega_m=0.15, sigma_8=0.2, n_s=0.2, w_0=0.8, w_a=1.3, omega_b=0.003, h=0.125
)
IA_PRIOR_SIGMA = dict(A0=2.5, beta=1.0, etal=1.5, etah=0.5)
GBIAS_PRIOR_SIGMA = 0.9

DEFAULT_IA_PARAMS = dict(A0=5.92, beta=1.1, etal=-0.47, etah=0.0)

#: SRD galaxy bias for the default LSST lens samples.
GBIAS_Y10 = [1.376695, 1.451179, 1.528404, 1.607983, 1.689579,
             1.772899, 1.857700, 1.943754, 2.030887, 2.118943]
GBIAS_Y1 = [1.562362, 1.732963, 1.913252, 2.100644, 2.29321]

#: LSST SRD lens number density, arcmin^-2, summed over all lens bins.
NEFF_LENS_TOTAL_Y10 = 48.0
NEFF_LENS_TOTAL_Y1 = 18.0

COSMO_PARAMS = ("omega_m", "sigma_8", "n_s", "w_0", "w_a", "omega_b", "h")
IA_PARAMS = ("A0", "beta", "etal", "etah")


# --------------------------------------------------------------------------
# small helpers borrowed from fisher.py (kept local to avoid the 1.2 s import)
# --------------------------------------------------------------------------


def FoM(matrix):
    """``1 / sqrt(det(Cov))`` for a 2x2 parameter covariance.

    Identical to ``fisherA2Z.fisher.FoM``: applied to
    ``marginalize(F, i, j)`` -- which returns the *inverse* of the marginal
    covariance -- this is the paper's figure of merit.
    """
    return np.sqrt(np.linalg.det(matrix))


def marginalize(fisher_matrix, i, j):
    """Marginal 2-parameter Fisher block for parameters ``i`` and ``j``."""
    return np.linalg.inv(np.linalg.inv(fisher_matrix)[np.ix_([i, j], [i, j])])


def _a_l(z, etal):
    """Faint-end IA luminosity scaling, vectorized."""
    return ((1.0 + np.asarray(z, dtype=float)) / 1.62) ** etal


def _a_h(z, etah):
    """Bright-end IA luminosity scaling, vectorized (A2Z's is scalar-only)."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0.75, ((1.0 + z) / 1.75) ** etah, 1.0)


def _get_phi(z, cosmo, Mr_s=-20.70, Q=1.23, alpha_lum=-1.23, phi_0=0.0094,
             P=-0.3, mlim=25.3, Mp=-22.0):
    """Normalized Schechter luminosity function (Loveday 2012 / Krause 2015).

    A direct transcription of ``Fisher.get_phi``, which never touches ``self``.
    Returns ``(L, phi_normed)``: per-redshift luminosity grids (the lower limit
    moves with z) and the luminosity function on them.
    """
    z = np.asarray(z, dtype=float)
    phi_s = phi_0 * 10.0 ** (0.4 * P * z)
    Ms = Mr_s - Q * (z - 0.1)
    Ls = 10 ** (-0.4 * (Ms - Mp))

    z_k, kcorr, *_ = np.loadtxt(_PACKAGE_PATH + "/data/kcorr.dat", unpack=True)
    z_e, ecorr, *_ = np.loadtxt(_PACKAGE_PATH + "/data/ecorr.dat", unpack=True)
    kcorr = CubicSpline(z_k, kcorr)(z)
    ecorr = CubicSpline(z_e, ecorr)(z)

    dl = ccl.luminosity_distance(cosmo, 1.0 / (1.0 + z))
    Mlim = mlim - (5.0 * np.log10(dl) + 25.0 + kcorr + ecorr)
    Llim = 10.0 ** (-0.4 * (Mlim - Mp))

    L = [np.logspace(np.log10(Llim[i]), 2.0, 1000) for i in range(z.size)]
    phi_normed = []
    for i in range(z.size):
        phi = phi_s[i] * (L[i] / Ls[i]) ** alpha_lum * np.exp(-L[i] / Ls[i])
        try:
            phi_normed.append(phi / scipy.integrate.simps(phi, L[i]))
        except AttributeError:
            phi_normed.append(phi / scipy.integrate.simpson(phi, L[i]))
    return L, phi_normed


class _AiCache:
    """Cache for ``Ai(z)``, the luminosity-weighted IA amplitude.

    ``Ai_ofzl`` depends only on the redshift grid, ``beta`` and the cosmology
    (through the luminosity distance) -- not on n(z).  ``Fisher.getAi``
    recomputes it for every source bin of every C_ell evaluation, re-reading
    ``kcorr.dat``/``ecorr.dat`` from disk each time, at ~0.3 s a call.  Caching
    it is the single largest speedup in this module.
    """

    def __init__(self):
        self._phi = {}
        self._ai = {}
        self.hits = 0
        self.misses = 0

    def ai_of_z(self, z, cosmo, cosmo_key, beta):
        key = (cosmo_key, float(beta), z.shape[0], float(z[0]), float(z[-1]))
        cached = self._ai.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        phi_key = (cosmo_key, z.shape[0], float(z[0]), float(z[-1]))
        if phi_key not in self._phi:
            self._phi[phi_key] = _get_phi(z, cosmo)
        L, phi_normed = self._phi[phi_key]
        try:
            ai = np.array([
                scipy.integrate.simps(phi_normed[i] * L[i] ** beta, L[i])
                for i in range(z.size)
            ])
        except AttributeError:
            ai = np.array([
                scipy.integrate.simpson(phi_normed[i] * L[i] ** beta, L[i])
                for i in range(z.size)
            ])
        self._ai[key] = ai
        return ai


def _ai_for_nz(z, nz, ai_of_z):
    """n(z)-weighted average of ``Ai(z)``."""
    try:
        return (scipy.integrate.simps(ai_of_z * nz, z)
                / scipy.integrate.simps(nz, z))
    except AttributeError:
        return (scipy.integrate.simpson(ai_of_z * nz, z)
                / scipy.integrate.simpson(nz, z))


# --------------------------------------------------------------------------
# the model evaluated at a point in parameter space
# --------------------------------------------------------------------------


class _FlexModel:
    """Everything needed to turn a parameter vector into a data vector.

    Deliberately picklable: it holds plain arrays and a cosmology *kwargs
    dict*, never a ``ccl.Cosmology`` or a tracer (CCL tracers contain SWIG
    objects and cannot be pickled).
    """

    def __init__(self, state):
        self.__dict__.update(state)
        self.z = np.asarray(self.z)
        self._cache = _AiCache()

    # -- state ---------------------------------------------------------
    STATE_KEYS = (
        "z", "nz_model", "nz_source_fid", "pz_pivot",
        "nz_out", "pz_mu", "pz_sigma", "pz_fout",
        "z_lens", "nz_lens", "ell", "cosmo_kwargs", "fid",
        "n_source", "n_lens",
        "ia_lf_cosmo", "omega_b_convention", "param_order",
    )

    def to_state(self):
        return {k: getattr(self, k) for k in self.STATE_KEYS}

    # -- cosmology -----------------------------------------------------
    def cosmology(self, vals):
        kw = dict(self.cosmo_kwargs)
        omega_b = vals["omega_b"]
        if self.omega_b_convention == "a2z":
            # A2Z holds Omega_c fixed at its fiducial value, so Omega_m moves
            # with Omega_b.
            omega_c = self.fid["omega_m"] - self.fid["omega_b"]
        else:
            omega_c = vals["omega_m"] - omega_b
        kw.update(
            Omega_c=omega_c, Omega_b=omega_b, h=vals["h"],
            sigma8=vals["sigma_8"], n_s=vals["n_s"],
            w0=vals["w_0"], wa=vals["w_a"],
        )
        key = tuple(sorted((k, float(v)) for k, v in kw.items()
                           if isinstance(v, (int, float))))
        return ccl.Cosmology(**kw), key

    def fiducial_cosmology(self):
        return self.cosmology(self.fid)

    # -- n(z) ----------------------------------------------------------
    def source_nz(self, vals):
        out = np.empty((self.n_source, self.z.size))
        if self.nz_model in ("shift_stretch", "no_uncertainty"):
            # Both share the template.  'no_uncertainty' pins the shift at 0 and
            # the stretch at 1 rather than reading them from ``vals``, where
            # they do not exist -- and goes through the same call so the two
            # models produce a bitwise identical fiducial n(z).
            frozen = self.nz_model == "no_uncertainty"
            for i in range(self.n_source):
                out[i] = shift_stretch_nz(
                    self.z, self.nz_source_fid[i],
                    0.0 if frozen else vals[f"zbias{i + 1}"],
                    1.0 if frozen else vals[f"zstretch{i + 1}"],
                    self.pz_pivot[i],
                )
            return out
        for i in range(self.n_source):
            out[i] = build_model_nz(
                self.z,
                self.pz_mu[i] + vals[f"zbias{i + 1}"],
                max(vals[f"zvariance{i + 1}"], 1e-4),
                np.clip(vals[f"zoutlier{i + 1}"], 0.0, 1.0),
                self.nz_out[i],
            )
        return out

    # -- tracers -------------------------------------------------------
    def legs(self, vals, nz_src=None):
        """All CCL tracers, lens legs first then source legs.

        ``nz_src`` overrides the source n(z) that :meth:`source_nz` would build
        from ``vals``.  It has to enter here rather than upstream because the
        intrinsic-alignment amplitude below is an n(z)-weighted average, so an
        overridden n(z) must carry its own IA term.
        """
        cosmo, cosmo_key = self.cosmology(vals)
        lf_cosmo, lf_key = (
            self.fiducial_cosmology() if self.ia_lf_cosmo == "fiducial"
            else (cosmo, cosmo_key)
        )
        ai_of_z = self._cache.ai_of_z(self.z, lf_cosmo, lf_key, vals["beta"])
        ia0 = vals["A0"] * _a_l(self.z, vals["etal"]) * _a_h(self.z, vals["etah"])

        tracers = []
        for i in range(self.n_lens):
            b = vals[f"gbias{i + 1}"]
            tracers.append(ccl.NumberCountsTracer(
                cosmo,
                dndz=(self.z_lens, self.nz_lens[i]),
                has_rsd=False,
                bias=(self.z_lens, b * np.ones_like(self.z_lens)),
            ))
        if nz_src is None:
            nz_src = self.source_nz(vals)
        for i in range(self.n_source):
            ia = _ai_for_nz(self.z, nz_src[i], ai_of_z) * ia0
            tracers.append(ccl.WeakLensingTracer(
                cosmo, dndz=(self.z, nz_src[i]), ia_bias=(self.z, ia)
            ))
        return cosmo, tracers

    # -- spectra -------------------------------------------------------
    def data_vector(self, vals, layout, nz_src=None):
        cosmo, tracers = self.legs(vals, nz_src=nz_src)
        out = np.empty((layout.n_blocks, len(self.ell)))
        for b in range(layout.n_blocks):
            out[b] = ccl.angular_cl(
                cosmo, tracers[layout.leg_a[b]], tracers[layout.leg_b[b]], self.ell
            )
        return out

    def all_cl(self, vals):
        """Every leg pair, needed for the Wick contractions in the covariance."""
        cosmo, tracers = self.legs(vals)
        n = len(tracers)
        out = np.zeros((n, n, len(self.ell)))
        for a in range(n):
            for b in range(a, n):
                cl = ccl.angular_cl(cosmo, tracers[a], tracers[b], self.ell)
                out[a, b] = cl
                out[b, a] = cl
        return out


# --------------------------------------------------------------------------
# multiprocessing worker (module level so it survives fork/pickling)
# --------------------------------------------------------------------------

_WORKER = {}


def _init_worker(state, layout_state, fid, steps, deriv_method, overrides=None):
    """Populate the module-level worker state.

    Called in the *parent* before the pool is created as well as in every
    child, so serial and parallel execution run literally the same code with
    the same globals -- which makes bitwise agreement a property of the design
    rather than a coincidence.
    """
    _WORKER["model"] = _FlexModel(state)
    _WORKER["layout"] = fl.BlockLayout(**layout_state)
    _WORKER["fid"] = fid
    _WORKER["steps"] = steps
    _WORKER["deriv_method"] = deriv_method
    _WORKER["overrides"] = overrides or {}


def _eval_at(param, value):
    vals = dict(_WORKER["fid"])
    vals[param] = value
    return _WORKER["model"].data_vector(vals, _WORKER["layout"])


def _deriv_one_param(param):
    """Numerical derivative of the whole data vector w.r.t. one parameter."""
    step = _WORKER["steps"][param]
    x0 = _WORKER["fid"][param]
    method = _WORKER["deriv_method"]

    if method == "forward":
        return (_eval_at(param, x0 + step) - _eval_at(param, x0)) / step
    if method == "central":
        return (_eval_at(param, x0 + step) - _eval_at(param, x0 - step)) / (2.0 * step)
    if method == "numdifftools":
        import numdifftools as nd

        d = nd.Derivative(lambda v: _eval_at(param, float(v)).reshape(-1), step=step)
        out = np.asarray(d(x0), dtype=float).reshape(-1)
        return out.reshape(_WORKER["layout"].n_blocks, -1)
    raise ValueError(f"unknown deriv_method {method!r}")


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


#: Two-dimensional ``Delta chi^2`` for 68.3% / 95.4% / 99.7% containment.
#:
#: Shared by :meth:`FisherFlexResult.contour` and :meth:`Bias2D.inside` so that
#: "the shift lands outside the 2-sigma contour" means the same thing whether it
#: is asserted numerically or read off a plot.
CHI2_2D = {1: 2.30, 2: 6.17, 3: 11.8}


@dataclass
class FisherFlexResult:
    """Output of :meth:`FisherFlex.forecast`."""

    fisher: np.ndarray
    param_order: list
    param_labels: list
    param_fid: dict
    mask: np.ndarray
    n_data_used: int
    prior_sigma: np.ndarray

    def __post_init__(self):
        self.cov = np.linalg.inv(self.fisher)

    def index(self, param):
        if isinstance(param, str):
            if param not in self.param_order:
                raise KeyError(
                    f"unknown parameter {param!r}; available: {self.param_order}"
                )
            return self.param_order.index(param)
        return int(param)

    def sigma(self, param):
        """Marginalized 1-sigma uncertainty."""
        return float(np.sqrt(self.cov[self.index(param), self.index(param)]))

    def sigmas(self):
        return {p: self.sigma(p) for p in self.param_order}

    def marginalize(self, p1, p2):
        """2x2 marginal Fisher block (the inverse of the 2x2 covariance)."""
        return marginalize(self.fisher, self.index(p1), self.index(p2))

    def fom(self, p1, p2):
        """Figure of merit ``1 / sqrt(det(Cov_2x2))`` for two parameters."""
        return float(FoM(self.marginalize(p1, p2)))

    @property
    def prior_dominated(self):
        """Parameters whose posterior is within 1% of their prior width."""
        s = np.array([self.sigma(p) for p in self.param_order])
        return s > 0.99 * self.prior_sigma

    def s8(self):
        """``S_8 = sigma_8 sqrt(Omega_m / 0.3)`` and its propagated error."""
        om = self.param_fid["omega_m"]
        s8v = self.param_fid["sigma_8"]
        i, j = self.index("omega_m"), self.index("sigma_8")
        grad = np.array([0.5 * s8v / np.sqrt(0.3 * om), np.sqrt(om / 0.3)])
        c = self.cov[np.ix_([i, j], [i, j])]
        return s8v * np.sqrt(om / 0.3), float(np.sqrt(grad @ c @ grad))

    # -- plotting ------------------------------------------------------
    def contour(self, p1, p2, ax=None, sigmas=(1, 2), centre=None, autolim=True,
                **kwargs):
        """Draw the 2-parameter confidence ellipses.

        Note: ``fisherA2Z.fisher.plot_contours`` takes ``eig[1][argmax]``,
        which is a *row* of the eigenvector matrix; ``np.linalg.eig`` returns
        eigenvectors as columns, so its ellipse orientation is wrong except by
        accident.  This uses the column.  ``fisher.py`` is left untouched so
        published figures are unaffected.

        Parameters
        ----------
        centre : (float, float), optional
            Where to put the ellipse.  Defaults to the fiducial values; a
            :class:`FisherFlexBias` passes the *shifted* centre here to draw the
            biased posterior with the same (correct) orientation code.
        autolim : bool
            Reset the axis limits to ``centre +/- 3.5 sigma``.  Switch off when
            overlaying on axes whose limits are already set -- otherwise a
            second call clobbers them and can clip a bias arrow.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        if ax is None:
            _, ax = plt.subplots()
        i, j = self.index(p1), self.index(p2)
        cov = self.cov[np.ix_([i, j], [i, j])]
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        if centre is None:
            centre = (self.param_fid[p1], self.param_fid[p2])
        centre = (float(centre[0]), float(centre[1]))
        for s in sigmas:
            w, h = 2.0 * np.sqrt(CHI2_2D[s] * evals)
            ax.add_patch(Ellipse(centre, w, h, angle=angle,
                                 fill=False, **kwargs))
        if autolim:
            pad = 3.5 * np.sqrt(cov[0, 0]), 3.5 * np.sqrt(cov[1, 1])
            ax.set_xlim(centre[0] - pad[0], centre[0] + pad[0])
            ax.set_ylim(centre[1] - pad[1], centre[1] + pad[1])
        ax.set_xlabel(self.param_labels[i])
        ax.set_ylabel(self.param_labels[j])
        return ax

    def corner(self, params, figsize=None, color="C0", label=None, fig=None):
        """Triangle plot of the marginal ellipses for ``params``."""
        import matplotlib.pyplot as plt

        n = len(params)
        if fig is None:
            fig, axes = plt.subplots(n, n, figsize=figsize or (2.2 * n, 2.2 * n))
        else:
            axes = np.array(fig.axes).reshape(n, n)
        for a in range(n):
            for b in range(n):
                ax = axes[a, b]
                if b > a:
                    ax.set_visible(False)
                    continue
                if a == b:
                    p = params[a]
                    s = self.sigma(p)
                    x = np.linspace(self.param_fid[p] - 4 * s,
                                    self.param_fid[p] + 4 * s, 200)
                    # Label only the first panel, or fig.legend() repeats the
                    # entry once per diagonal.
                    ax.plot(x, norm.pdf(x, self.param_fid[p], s),
                            color=color, label=label if a == 0 else None)
                    ax.set_yticks([])
                else:
                    self.contour(params[b], params[a], ax=ax, color=color)
                if a != n - 1:
                    ax.set_xlabel("")
                if b != 0 or a == 0:
                    ax.set_ylabel("")
        fig.tight_layout()
        return fig

    def summary(self, params=None):
        """One line per parameter: fiducial, sigma, and prior width."""
        params = params or self.param_order
        lines = [f"{'parameter':<14}{'fiducial':>12}{'sigma':>12}{'prior':>12}"]
        for p in params:
            i = self.index(p)
            lines.append(
                f"{p:<14}{self.param_fid[p]:>12.5g}{self.sigma(p):>12.5g}"
                f"{self.prior_sigma[i]:>12.5g}"
            )
        return "\n".join(lines)


@dataclass
class Bias2D:
    """Parameter bias in the plane of two parameters.

    Output of :meth:`FisherFlexBias.bias_2d`.  The per-parameter numbers on
    :class:`FisherFlexBias` answer "how far did this parameter move, in units of
    its own error bar"; this answers the joint question, "where does the biased
    point sit relative to the 2-D contour that would be plotted".

    Those are not the same, and the joint answer is usually the larger one: a
    shift can be a small fraction of each marginal error and still land well
    outside the ellipse, if it runs across the degeneracy rather than along it.
    """

    params: tuple
    labels: tuple
    fid: np.ndarray
    delta: np.ndarray
    shifted: np.ndarray
    cov: np.ndarray
    chi2: float

    @property
    def n_sigma(self):
        """Mahalanobis distance ``sqrt(chi2)`` -- the joint significance.

        Careful: this is *not* on the same footing as the 1-D ``bias/sigma``.
        In two dimensions the 68.3% contour sits at ``chi2 = 2.30``, i.e.
        ``n_sigma = 1.52``, not 1.  Use :attr:`confidence` or :meth:`inside` to
        compare against a contour.
        """
        return float(np.sqrt(self.chi2))

    @property
    def confidence(self):
        """Probability enclosed by the contour through the biased point.

        ``1 - exp(-chi2 / 2)``, the 2-dof chi-square CDF.  Reads directly: 0.68
        means the shift lands exactly on the 1-sigma ellipse.
        """
        return float(1.0 - np.exp(-0.5 * self.chi2))

    @property
    def angle(self):
        """Direction of the shift in the plane, degrees from the +``p1`` axis."""
        return float(np.degrees(np.arctan2(self.delta[1], self.delta[0])))

    def inside(self, sigmas=2):
        """Is the biased point inside the ``sigmas``-sigma contour?"""
        if sigmas not in CHI2_2D:
            raise KeyError(f"sigmas must be one of {sorted(CHI2_2D)}")
        return bool(self.chi2 <= CHI2_2D[sigmas])

    def marginal_n_sigma(self):
        """The two 1-D ``bias/sigma`` values, for contrast with :attr:`n_sigma`."""
        return tuple(self.delta / np.sqrt(np.diag(self.cov)))

    def summary(self):
        p1, p2 = self.params
        m1, m2 = self.marginal_n_sigma()
        return "\n".join([
            f"bias in the ({p1}, {p2}) plane",
            f"  {p1:<12} {self.fid[0]:>11.5g} -> {self.shifted[0]:>11.5g}"
            f"   ({m1:+.2f} sigma_1D)",
            f"  {p2:<12} {self.fid[1]:>11.5g} -> {self.shifted[1]:>11.5g}"
            f"   ({m2:+.2f} sigma_1D)",
            f"  joint: chi2 = {self.chi2:.3f}, sqrt(chi2) = {self.n_sigma:.2f}, "
            f"enclosing {self.confidence:.1%} of the posterior",
            f"  direction: {self.angle:+.1f} deg;  "
            f"inside 1 sigma: {self.inside(1)};  inside 2 sigma: {self.inside(2)}",
        ])

    def __str__(self):
        return self.summary()


@dataclass
class FisherFlexBias:
    """Output of :meth:`FisherFlex.forecast_bias`.

    The parameter bias induced by analysing data drawn from ``nz_truth`` with a
    model built around ``nz_fid``, to first order in the residual -- Eq. (13) of
    Zhang et al. (2025):

    ``delta_p = F^-1 . (dC_l/dp . Cov^-1 . (C_l^truth - C_l^fid))``

    ``F`` is the Fisher matrix of :attr:`result`, priors included, evaluated
    under the same scale cuts as the residual.

    Caveat, stated in the paper itself: this *overestimates* the bias a real
    analysis would suffer.  The Fisher posterior is centred on the centre of the
    prior, whereas an MCMC handed a biased data vector shifts the photo-z
    nuisances toward the true values and self-calibrates part of the systematic
    away.  Check :meth:`mafe` too -- a large residual invalidates the first-order
    expansion the whole formula rests on.
    """

    delta_p: dict
    bias_vec: np.ndarray
    result: FisherFlexResult
    delta_cl: np.ndarray
    data_vector: np.ndarray
    nz_truth: np.ndarray
    nz_fid: np.ndarray
    z: np.ndarray
    n_data_used: int

    @property
    def param_order(self):
        return self.result.param_order

    @property
    def param_labels(self):
        return self.result.param_labels

    def index(self, param):
        return self.result.index(param)

    def shift(self, param):
        """Bias on one parameter, in its own units."""
        if not isinstance(param, str):
            param = self.param_order[int(param)]
        return float(self.delta_p[param])

    def shifts(self):
        return dict(self.delta_p)

    def n_sigma(self, param):
        """Bias in units of the marginalized 1-sigma uncertainty."""
        return self.shift(param) / self.result.sigma(param)

    def n_sigmas(self):
        return {p: self.n_sigma(p) for p in self.param_order}

    def shifted_fid(self, param):
        """Where the inferred value lands: fiducial + bias."""
        name = param if isinstance(param, str) else self.param_order[int(param)]
        return self.result.param_fid[name] + self.shift(name)

    def s8(self):
        """``(S8_fid, S8_shifted, delta_S8, delta_S8 / sigma_S8)``.

        The shift is evaluated from the shifted ``(Omega_m, sigma_8)`` rather
        than by propagating the gradient, matching ``util.get_s8_shift``.
        """
        s8_fid, s8_err = self.result.s8()
        om = self.shifted_fid("omega_m")
        sig8 = self.shifted_fid("sigma_8")
        s8_shift = sig8 * np.sqrt(om / 0.3)
        d = s8_shift - s8_fid
        return s8_fid, s8_shift, d, d / s8_err

    def bias_2d(self, p1, p2):
        """Bias in the plane of two parameters -> :class:`Bias2D`.

        The 2-D companion to :meth:`FisherFlexResult.fom`.  Marginalizes over
        every other parameter, then measures the shift against the resulting
        2x2 covariance:

        ``chi2 = delta^T Cov_2x2^-1 delta``

        Reuses :meth:`FisherFlexResult.marginalize`, which returns exactly
        ``Cov_2x2^-1``.

        Examples
        --------
        >>> b2 = bias.bias_2d("omega_m", "sigma_8")
        >>> b2.inside(2), round(b2.confidence, 3)
        (True, 0.246)
        """
        i, j = self.result.index(p1), self.result.index(p2)
        delta = np.array([self.shift(p1), self.shift(p2)])
        fid = np.array([self.result.param_fid[p1], self.result.param_fid[p2]])
        inv_cov = self.result.marginalize(p1, p2)
        return Bias2D(
            params=(p1, p2),
            labels=(self.param_labels[i], self.param_labels[j]),
            fid=fid, delta=delta, shifted=fid + delta,
            cov=self.result.cov[np.ix_([i, j], [i, j])],
            chi2=float(delta @ inv_cov @ delta),
        )

    def mafe(self):
        """Mean absolute fractional error of the residual, Eq. (18).

        Averaged over the elements the forecast actually used.  This is the
        sanity check on the first-order expansion: a few per cent is fine, tens
        of per cent means :attr:`delta_p` should not be trusted quantitatively.
        """
        m = self.result.mask
        if not m.any():
            return float("nan")
        return float(np.mean(np.abs(self.delta_cl[m] / self.data_vector[m])))

    def summary(self, params=None):
        """One line per parameter: fiducial, bias, bias/sigma, shifted value."""
        params = params or self.param_order
        lines = [f"{'parameter':<14}{'fiducial':>12}{'bias':>12}"
                 f"{'bias/sigma':>12}{'shifted':>12}"]
        for p in params:
            lines.append(
                f"{p:<14}{self.result.param_fid[p]:>12.5g}{self.shift(p):>12.5g}"
                f"{self.n_sigma(p):>12.3f}{self.shifted_fid(p):>12.5g}"
            )
        return "\n".join(lines)

    # -- plotting ------------------------------------------------------
    def arrow(self, p1, p2, ax=None, shifted_contour=False, expand_lims=True,
              sigmas=(2,), **kwargs):
        """Draw the fiducial -> biased arrow for two parameters.

        Uses a :class:`~matplotlib.patches.FancyArrowPatch` so the head is sized
        in display units; an ``ax.arrow`` head width has to be retuned for every
        parameter pair, since e.g. ``(Omega_m, sigma_8)`` and ``(w_0, w_a)``
        differ by orders of magnitude in axis scale.

        Parameters
        ----------
        shifted_contour : bool
            Also draw the posterior re-centred on the biased values, dashed --
            the blue/orange pair of the paper's Fig. 6.
        expand_lims : bool
            Grow (never shrink) the axis limits so the arrow head stays visible.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        if ax is None:
            _, ax = plt.subplots()
        start = (self.result.param_fid[p1], self.result.param_fid[p2])
        end = (self.shifted_fid(p1), self.shifted_fid(p2))

        if shifted_contour:
            self.result.contour(p1, p2, ax=ax, sigmas=sigmas, centre=end,
                                autolim=False, linestyle="--",
                                color=kwargs.get("color", "C1"))

        style = dict(arrowstyle="-|>", mutation_scale=14, color="C1", lw=1.4,
                     shrinkA=0, shrinkB=0, zorder=5)
        style.update(kwargs)
        ax.add_patch(FancyArrowPatch(start, end, **style))

        if expand_lims:
            for lo_hi, setter, vals in (
                (ax.get_xlim(), ax.set_xlim, (start[0], end[0])),
                (ax.get_ylim(), ax.set_ylim, (start[1], end[1])),
            ):
                lo, hi = lo_hi
                pad = 0.05 * (hi - lo)
                setter(min(lo, min(vals) - pad), max(hi, max(vals) + pad))
        return ax

    def corner_arrows(self, params, fig, shifted_contour=False, **kwargs):
        """Overlay bias arrows on a figure made by :meth:`FisherFlexResult.corner`.

        Off-diagonal panels get an arrow; diagonal panels get a vertical line at
        the shifted value.
        """
        n = len(params)
        axes = np.array(fig.axes).reshape(n, n)
        color = kwargs.get("color", "C1")
        # Corner panels are small, so the default head swamps a sub-sigma arrow.
        kwargs.setdefault("mutation_scale", 9)
        for a in range(n):
            for b in range(n):
                if b > a:
                    continue
                ax = axes[a, b]
                if a == b:
                    ax.axvline(self.shifted_fid(params[a]), color=color,
                               ls="--", lw=1.2)
                    continue
                # corner() blanks the labels of interior panels; drawing a
                # shifted contour puts them back, so restore what was there.
                labels = ax.get_xlabel(), ax.get_ylabel()
                self.arrow(params[b], params[a], ax=ax,
                           shifted_contour=shifted_contour, **kwargs)
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
        return fig


# --------------------------------------------------------------------------
# the class
# --------------------------------------------------------------------------


class FisherFlex:
    """Fisher forecast for an arbitrary tomographic sample.

    Parameters
    ----------
    nz_source : ndarray, shape (n_tomo, n_z)
        Central source n(z) per tomographic bin.  Normalization is irrelevant.
    nz_realizations : ndarray, shape (n_real, n_tomo, n_z) or (n_tomo, n_real, n_z), optional
        Realizations of the same n(z); their scatter sets the photo-z prior.
        Omit it and supply ``photoz_prior_cov`` directly, or omit both with
        ``nz_model='no_uncertainty'``, which has no photo-z prior at all.
    z_grid : ndarray, shape (n_z,)
    neff_source : float or ndarray, shape (n_tomo,)
        Effective source number density per bin, arcmin^-2.
    fsky : float
    sigma_e : float
        Per-component ellipticity dispersion; the shape-noise power is
        ``sigma_e**2 / n``.
    cosmo : ccl.Cosmology, optional
        Fiducial cosmology.  Defaults to the A2Z Planck-like cosmology.
    mode : {'3x2pt', '2x2pt', 'cosmic_shear'}
    nz_model : {'shift_stretch', 'gaussian_outlier', 'no_uncertainty'}
        Photo-z parameterization; see the class docstring.  The default keeps
        the measured n(z) as its own template and gives each bin a shift and a
        stretch.  ``'no_uncertainty'`` treats n(z) as exactly known and adds no
        photo-z parameters at all -- the perfect-photo-z limit, and the
        optimistic bound on any of the others.
        ``'gaussian_outlier'`` is the A2Z Gaussian-core-plus-outlier
        model, which adds a third parameter per bin.
    step : float
        Default absolute step for the numerical derivatives.  Per-parameter
        overrides are applied for ``zvariance`` and ``zoutlier`` (see
        :attr:`param_step`).
    y1 : bool
        Use the LSST Y1 lens preset (5 bins, sigma_z = 0.06) instead of Y10.
    lens_nz, z_lens, lens_neff, gbias, n_lens_bins
        Override the default LSST lens sample.  ``lens_neff`` is **per lens
        bin**, in arcmin^-2 -- unlike the built-in default, which takes the SRD
        *total* (48 arcmin^-2 for Y10, 18 for Y1) and splits it across the bins
        by the dN/dz weights, giving 2.3-6.0 per bin for the 10-bin Y10 sample.
        A scalar is broadcast to every bin and warns, since that multiplies the
        sample by ``n_lens``.
    ggl_pairs : {'z_offset', 'a2z', 'all'} or sequence of (lens, source)
    ggl_offset : float
        For ``'z_offset'``: require ``mean(z)_source > mean(z)_lens + offset``.
    prior_mode : {'full', 'diag'}
        Whether the photo-z prior keeps its per-bin correlations.
    smooth_realizations : float, optional
        Gaussian smoothing sigma in *z units* applied to the realizations
        before they are fitted.  Only relevant for ``'gaussian_outlier'``,
        where pixel noise biases the recovered outlier fraction and can change
        the prior width by factors of a few; see
        :meth:`prior_stability_report`.  The shift-and-stretch parameters are
        moments of n(z), so they are already insensitive to it.
    deriv_method : {'numdifftools', 'central', 'forward'}
    realizations_axis : {0, 1}, optional
        Which axis of ``nz_realizations`` indexes realizations.  Only needed
        when ``n_real == n_tomo``.
    """

    def __init__(self, nz_source, nz_realizations=None, z_grid=None,
                 neff_source=None, fsky=None,
                 sigma_e=None, cosmo=None, mode="3x2pt", nz_model="shift_stretch",
                 step=0.001, y1=False,
                 lens_nz=None, z_lens=None, lens_neff=None, gbias=None,
                 n_lens_bins=None, ia_params=None, ggl_pairs="z_offset",
                 ggl_offset=0.3, prior_mode="full", smooth_realizations=None,
                 deriv_method="numdifftools", realizations_axis=None,
                 ia_lf_cosmo="varied", omega_b_convention="fixed_omega_m",
                 photoz_prior_cov=None, ell_edges=None, decompose_kwargs=None,
                 verbose=True):
        self.verbose = verbose
        self.mode = mode
        # Only ``nz_realizations`` is genuinely optional (nz_model
        # 'no_uncertainty' has no prior to derive, and the other models accept
        # photoz_prior_cov instead).  The rest default to None purely so that
        # ``nz_realizations`` can, without disturbing the positional order.
        missing = [n for n, v in (("z_grid", z_grid), ("neff_source", neff_source),
                                  ("fsky", fsky), ("sigma_e", sigma_e))
                   if v is None]
        if missing:
            raise TypeError(
                f"FisherFlex() missing required argument(s): {', '.join(missing)}")
        if mode not in fl.MODE_PROBES:
            raise ValueError(f"unknown mode {mode!r}; use one of {sorted(fl.MODE_PROBES)}.")
        if nz_model not in NZ_MODELS:
            raise ValueError(f"unknown nz_model {nz_model!r}; use one of {list(NZ_MODELS)}.")
        self.nz_model = nz_model
        self.step = float(step)
        self.y1 = bool(y1)
        self.prior_mode = prior_mode
        self.deriv_method = deriv_method
        self.ia_lf_cosmo = ia_lf_cosmo
        self.omega_b_convention = omega_b_convention

        self.z = np.asarray(z_grid, dtype=float)
        self.nz_source = np.atleast_2d(np.asarray(nz_source, dtype=float))
        if self.nz_source.shape[1] != self.z.size:
            raise ValueError(
                f"nz_source has {self.nz_source.shape[1]} redshift columns but "
                f"z_grid has {self.z.size} points."
            )
        self.n_source = self.nz_source.shape[0]

        self.fsky = float(fsky)
        self.sigma_e = float(sigma_e)
        self.neff_source = np.broadcast_to(
            np.atleast_1d(np.asarray(neff_source, dtype=float)), (self.n_source,)
        ).astype(float)

        # -- cosmology -------------------------------------------------
        self.cosmo_kwargs = dict(DEFAULT_COSMO_KWARGS)
        if cosmo is not None:
            self.cosmo_kwargs.update(_cosmo_to_kwargs(cosmo))
        self.cosmo = ccl.Cosmology(**self.cosmo_kwargs)

        self.ia_params = dict(DEFAULT_IA_PARAMS)
        if ia_params:
            self.ia_params.update(ia_params)

        # -- lens sample -----------------------------------------------
        self._setup_lens(lens_nz, z_lens, lens_neff, gbias, n_lens_bins)

        # -- photo-z model ---------------------------------------------
        self._setup_photoz(nz_realizations, realizations_axis, smooth_realizations,
                           photoz_prior_cov, decompose_kwargs or {})

        # -- layout ----------------------------------------------------
        self.ell_edges = (fl.default_ell_edges() if ell_edges is None
                          else np.asarray(ell_edges, dtype=float))
        self.ell = fl.ell_from_edges(self.ell_edges)
        self.ggl_pairs = self._resolve_ggl_pairs(ggl_pairs, ggl_offset)
        self.layout = fl.build_layout(self.n_lens, self.n_source,
                                      self.ggl_pairs, mode=mode)

        # -- parameters ------------------------------------------------
        self._setup_params()

        # computed later
        self.data_vector = None
        self.deriv = None
        self.cov_blocks = None
        self.chi_eff_lens_Mpc = None
        self._computed = False

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _setup_lens(self, lens_nz, z_lens, lens_neff, gbias, n_lens_bins):
        if lens_nz is not None:
            self.nz_lens = np.atleast_2d(np.asarray(lens_nz, dtype=float))
            self.z_lens = (self.z if z_lens is None
                           else np.asarray(z_lens, dtype=float))
            if self.nz_lens.shape[1] != self.z_lens.size:
                raise ValueError("lens_nz and z_lens have inconsistent shapes.")
            self.n_lens = self.nz_lens.shape[0]
            self.z_eff_lens = np.array([
                np.trapz(self.z_lens * n, self.z_lens) / np.trapz(n, self.z_lens)
                for n in self.nz_lens
            ])
            weights = np.ones(self.n_lens) / self.n_lens
        else:
            self.n_lens = int(n_lens_bins or (5 if self.y1 else 10))
            self.z_lens, self.nz_lens, self.z_eff_lens, weights = \
                _default_lens_sample(self.n_lens, self.y1)

        if gbias is not None:
            self.gbias = np.asarray(gbias, dtype=float)
        else:
            default = GBIAS_Y1 if self.y1 else GBIAS_Y10
            if self.n_lens == len(default):
                self.gbias = np.array(default)
            else:
                # SRD parameterization: b(z) = 1.05 / D(z), close enough to the
                # tabulated values for a non-standard binning.
                self.gbias = 1.05 / ccl.growth_factor(
                    self.cosmo, 1.0 / (1.0 + self.z_eff_lens))
        if self.gbias.size != self.n_lens:
            raise ValueError(
                f"gbias has {self.gbias.size} entries but there are {self.n_lens} lens bins."
            )

        if lens_neff is not None:
            lens_neff = np.atleast_1d(np.asarray(lens_neff, dtype=float))
            if lens_neff.size == 1 and self.n_lens > 1:
                # The default path splits a *total* across bins by dN/dz, so a
                # bare scalar here is ambiguous -- and reading it as per-bin
                # (which is what broadcasting does) silently multiplies the
                # sample by n_lens.  Say what it came out as.
                warnings.warn(
                    f"lens_neff={float(lens_neff[0]):g} is per *bin*, giving "
                    f"{float(lens_neff[0]) * self.n_lens:g} arcmin^-2 over all "
                    f"{self.n_lens} lens bins. Pass an array of length "
                    f"{self.n_lens} to set the per-bin densities directly. "
                    "(For reference the LSST SRD gold sample totals 48 "
                    "arcmin^-2 for Y10 and 18 for Y1.)"
                )
            self.neff_lens = np.broadcast_to(
                lens_neff, (self.n_lens,)).astype(float)
        else:
            total = NEFF_LENS_TOTAL_Y1 if self.y1 else NEFF_LENS_TOTAL_Y10
            self.neff_lens = total * weights

    def _setup_photoz(self, realizations, realizations_axis, smooth, prior_cov,
                      decompose_kwargs):
        """Build the per-bin n(z) template, fiducial parameters and prior."""
        self.smooth_realizations = smooth
        self.realization_params = None
        self.n_pz_per_bin = N_PZ_PER_BIN[self.nz_model]

        if self.nz_model == "no_uncertainty":
            self._setup_no_uncertainty(decompose_kwargs)
            if realizations is not None or prior_cov is not None:
                warnings.warn(
                    "nz_model='no_uncertainty' has no photo-z parameters, so "
                    "nz_realizations and photoz_prior_cov are ignored."
                )
            self.pz_prior_cov = np.zeros((self.n_source, 0, 0))
            self.pz_realization_offset = None
            return

        if self.nz_model == "shift_stretch":
            self._setup_shift_stretch(decompose_kwargs)
        else:
            self._setup_gaussian_outlier(decompose_kwargs)

        if prior_cov is not None:
            self.pz_prior_cov = np.asarray(prior_cov, dtype=float)
            k = self.n_pz_per_bin
            if self.pz_prior_cov.shape != (self.n_source, k, k):
                raise ValueError(
                    f"photoz_prior_cov must have shape ({self.n_source}, {k}, {k}) "
                    f"for nz_model={self.nz_model!r}."
                )
            return

        if realizations is None:
            raise ValueError(
                "either nz_realizations or photoz_prior_cov must be provided; "
                "the photo-z prior cannot be derived otherwise."
            )

        real = nzd.orient_realizations(realizations, self.n_source, realizations_axis)
        if self.nz_model == "gaussian_outlier":
            ratio = nzd.noise_ratio(self.z, self.nz_source, real)
            if ratio > 5.0 and smooth is None:
                warnings.warn(
                    f"the realizations are ~{ratio:.0f}x noisier than the central n(z). "
                    "The outlier fraction is recovered from a min-ratio statistic, so "
                    "pixel noise biases it low and inflates the prior width. Consider "
                    "smooth_realizations=<sigma in z units>, check "
                    "prior_stability_report(), or use nz_model='shift_stretch'."
                )
        if smooth:
            real = nzd.smooth_nz(self.z, real, smooth)
        self.realization_params = self._fit_realizations(real, decompose_kwargs)
        self.pz_prior_cov = nzd.photoz_prior_from_realizations(
            self.realization_params, mode=self.prior_mode)
        self._check_prior_centring(self.realization_params)

    def _setup_shift_stretch(self, decompose_kwargs):
        """The measured n(z) is the template; nothing is fitted."""
        if decompose_kwargs:
            warnings.warn(
                "decompose_kwargs is ignored for nz_model='shift_stretch': the "
                "shift and stretch are moments of n(z), not the output of a fit."
            )
        m = nzs.build_shift_stretch_model(self.z, self.nz_source)
        self.shift_stretch = m
        self.decomposition = None
        self.nz_source_fid = m.nz_fid
        self.pz_pivot = m.z_pivot
        # pz_mu is the redshift that dz shifts and pz_sigma the width that the
        # stretch scales, so both stay meaningful across the two models.
        self.pz_mu = m.z_pivot
        self.pz_sigma = m.sigma_fid
        self.pz_fout = np.zeros(self.n_source)
        self.nz_out = np.zeros((0, 0))  # no outlier template in this model
        self.pz_degenerate = np.zeros(self.n_source, dtype=bool)

    def _setup_no_uncertainty(self, decompose_kwargs):
        """The measured n(z) is taken as exactly known: no photo-z parameters.

        Shares the shift-and-stretch template machinery -- it is the same
        normalized n(z), just with nothing free to vary.  ``pz_mu`` and
        ``pz_sigma`` are still filled in with the moments of n(z) so that
        reporting code can read them, but no parameter is attached to either.
        """
        if decompose_kwargs:
            warnings.warn(
                "decompose_kwargs is ignored for nz_model='no_uncertainty': "
                "nothing is fitted."
            )
        m = nzs.build_shift_stretch_model(self.z, self.nz_source)
        self.shift_stretch = m
        self.decomposition = None
        self.nz_source_fid = m.nz_fid
        self.pz_pivot = m.z_pivot
        self.pz_mu = m.z_pivot
        self.pz_sigma = m.sigma_fid
        self.pz_fout = np.zeros(self.n_source)
        self.nz_out = np.zeros((0, 0))
        self.pz_degenerate = np.zeros(self.n_source, dtype=bool)

    def _setup_gaussian_outlier(self, decompose_kwargs):
        self._log(f"Decomposing {self.n_source} source bins ...")
        self.decomposition = nzd.decompose_tomography(
            self.z, self.nz_source, **decompose_kwargs)
        d = self.decomposition
        self.shift_stretch = None
        self.pz_mu = d.mu
        self.pz_sigma = d.sigma
        self.pz_pivot = d.mu
        self.pz_fout = d.f_out
        self.nz_out = d.nz_out
        self.nz_source_fid = d.nz_fid
        self.pz_degenerate = d.degenerate

    def _fit_realizations(self, real, decompose_kwargs):
        if self.nz_model == "shift_stretch":
            self._log(f"Measuring the shift and stretch of {real.shape[0]} "
                      f"realizations ...")
            return nzs.fit_shift_stretch(self.z, self.nz_source_fid, real)
        self._log(f"Fitting {real.shape[0]} realizations x {self.n_source} bins ...")
        return nzd.decompose_realizations(
            self.z, real,
            warm_start=np.column_stack([self.pz_mu, self.pz_sigma]),
            **decompose_kwargs)

    def _check_prior_centring(self, params):
        """Warn if the central n(z) is not the centroid of the realizations.

        The prior is a scatter about the fiducial n(z), so a mean shift much
        larger than that scatter means the two inputs describe different
        samples -- almost always a mix-up rather than a real feature.

        Only checked for ``'shift_stretch'``, where the estimator is a plain
        moment and the offset means exactly one thing.  The Gaussian fit's
        ``mu`` is a biased order statistic on noisy realizations, so the same
        comparison there would mix that bias with a genuine mismatch and fire
        when nothing is wrong.
        """
        if self.nz_model != "shift_stretch":
            self.pz_realization_offset = None
            return
        shift = params[:, :, 0]
        self.pz_realization_offset = np.nanmean(shift, axis=0)
        scatter = np.nanstd(shift, axis=0)
        bad = np.abs(self.pz_realization_offset) > np.where(scatter > 0, scatter, np.inf)
        if np.any(bad):
            warnings.warn(
                "the realizations are offset from the central n(z) by more than "
                "their own scatter in bins "
                f"{[int(i) + 1 for i in np.flatnonzero(bad)]} "
                f"(offset {np.round(self.pz_realization_offset[bad], 4).tolist()}, "
                f"scatter {np.round(scatter[bad], 4).tolist()}); check that "
                "nz_source really is the mean of nz_realizations."
            )

    def _resolve_ggl_pairs(self, spec, offset):
        z_src = np.array([np.trapz(self.z * n, self.z) / np.trapz(n, self.z)
                            for n in self.nz_source])
        self.z_mean_source = z_src
        if isinstance(spec, str):
            if spec == "z_offset":
                return fl.select_ggl_pairs(self.z_eff_lens, z_src, offset)
            if spec == "all":
                return [(l, s) for l in range(self.n_lens)
                        for s in range(self.n_source)]
            if spec == "a2z":
                accept = fl.A2Z_GGL_ACCEPT_Y1 if self.y1 else fl.A2Z_GGL_ACCEPT_Y10
                return sorted(accept)
            raise ValueError(f"unknown ggl_pairs {spec!r}.")
        return sorted(tuple(p) for p in spec)

    def _pz_param_names(self, i):
        """Photo-z parameter names of source bin ``i`` (0-based).

        The order matches the rows and columns of ``pz_prior_cov[i]``.  Empty
        for ``nz_model='no_uncertainty'``.
        """
        if self.nz_model == "no_uncertainty":
            return []
        if self.nz_model == "shift_stretch":
            return [f"zbias{i + 1}", f"zstretch{i + 1}"]
        return [f"zbias{i + 1}", f"zvariance{i + 1}", f"zoutlier{i + 1}"]

    def _setup_params(self):
        ns, nl = self.n_source, self.n_lens
        if self.nz_model == "no_uncertainty":
            pz_names, pz_labels = [], []
        elif self.nz_model == "shift_stretch":
            pz_names = ([f"zbias{i}" for i in range(1, ns + 1)]
                        + [f"zstretch{i}" for i in range(1, ns + 1)])
            pz_labels = ([rf"$\delta z_{{{i}}}$" for i in range(1, ns + 1)]
                         + [rf"$s_{{z,{i}}}$" for i in range(1, ns + 1)])
        else:
            pz_names = ([f"zbias{i}" for i in range(1, ns + 1)]
                        + [f"zvariance{i}" for i in range(1, ns + 1)]
                        + [f"zoutlier{i}" for i in range(1, ns + 1)])
            pz_labels = ([rf"$\delta z_{{{i}}}$" for i in range(1, ns + 1)]
                         + [rf"$\sigma_{{z,{i}}}$" for i in range(1, ns + 1)]
                         + [rf"$f_{{out,{i}}}$" for i in range(1, ns + 1)])

        self.param_order = (
            list(COSMO_PARAMS) + list(IA_PARAMS) + pz_names
            + [f"gbias{i}" for i in range(1, nl + 1)]
        )
        self.param_labels = (
            [r"$\Omega_m$", r"$\sigma_8$", r"$n_s$", r"$w_0$", r"$w_a$",
             r"$\Omega_b$", r"$h$", r"$A_0$", r"$\beta$", r"$\eta_l$", r"$\eta_h$"]
            + pz_labels
            + [rf"$b_g^{{{i}}}$" for i in range(1, nl + 1)]
        )

        omega_m = self.cosmo_kwargs["Omega_c"] + self.cosmo_kwargs["Omega_b"]
        self.fid = {
            "omega_m": omega_m,
            "sigma_8": self.cosmo_kwargs["sigma8"],
            "n_s": self.cosmo_kwargs["n_s"],
            "w_0": self.cosmo_kwargs["w0"],
            "w_a": self.cosmo_kwargs["wa"],
            "omega_b": self.cosmo_kwargs["Omega_b"],
            "h": self.cosmo_kwargs["h"],
        }
        self.fid.update(self.ia_params)
        for i in range(ns):
            if self.nz_model == "no_uncertainty":
                break
            self.fid[f"zbias{i + 1}"] = 0.0
            if self.nz_model == "shift_stretch":
                self.fid[f"zstretch{i + 1}"] = 1.0
            else:
                self.fid[f"zvariance{i + 1}"] = float(self.pz_sigma[i])
                self.fid[f"zoutlier{i + 1}"] = float(self.pz_fout[i])
        for i in range(nl):
            self.fid[f"gbias{i + 1}"] = float(self.gbias[i])

        # Priors.  Cosmology/IA/gbias are the paper's Gaussian priors; photo-z
        # comes from the realization scatter, kept as per-bin blocks.
        self.prior_sigma = {}
        self.prior_sigma.update(COSMO_PRIOR_SIGMA)
        self.prior_sigma.update(IA_PRIOR_SIGMA)
        for i in range(nl):
            self.prior_sigma[f"gbias{i + 1}"] = GBIAS_PRIOR_SIGMA
        for i in range(ns):
            sd = np.sqrt(np.diag(self.pz_prior_cov[i]))
            for k, name in enumerate(self._pz_param_names(i)):
                self.prior_sigma[name] = float(sd[k])

        self.param_step = {}
        for p in self.param_order:
            if p.startswith("zvariance"):
                h = max(1e-3, 0.02 * self.fid[p])
            elif p.startswith("zoutlier"):
                h = max(1e-4, 0.05 * self.fid[p])
            else:
                # Including zstretch: the template is interpolated linearly, so
                # the central difference is the same for any step below the
                # grid spacing (see nz_shift_stretch).
                h = self.step
            self.param_step[p] = float(h)

        # A negative outlier weight would make n(z) go negative, so use a
        # forward difference when the fiducial value sits too close to zero.
        # The shift-and-stretch model has no such parameter: dz is signed and
        # the stretch sits at 1.
        self.param_deriv_method = {p: None for p in self.param_order}
        if self.nz_model == "gaussian_outlier":
            for i in range(ns):
                p = f"zoutlier{i + 1}"
                if self.fid[p] < 2.0 * self.param_step[p]:
                    self.param_deriv_method[p] = "forward"

    def override_priors(self, priors):
        """Replace individual prior widths, ``{param: sigma}``."""
        for k, v in priors.items():
            if k not in self.prior_sigma:
                raise KeyError(f"unknown parameter {k!r}")
            self.prior_sigma[k] = float(v)
        return self

    @property
    def n_params(self):
        return len(self.param_order)

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    def _model(self):
        return _FlexModel(self._to_state())

    def _to_state(self):
        return dict(
            z=self.z,
            nz_model=self.nz_model,
            nz_source_fid=self.nz_source_fid,
            pz_pivot=self.pz_pivot,
            nz_out=self.nz_out,
            pz_mu=self.pz_mu,
            pz_sigma=self.pz_sigma,
            pz_fout=self.pz_fout,
            z_lens=self.z_lens,
            nz_lens=self.nz_lens,
            ell=self.ell,
            cosmo_kwargs=self.cosmo_kwargs,
            fid=self.fid,
            n_source=self.n_source,
            n_lens=self.n_lens,
            layout_kwargs=None,
            ia_lf_cosmo=self.ia_lf_cosmo,
            omega_b_convention=self.omega_b_convention,
            param_order=self.param_order,
        )

    def _layout_state(self):
        return dict(
            n_lens=self.layout.n_lens, n_source=self.layout.n_source,
            probe=self.layout.probe, leg_a=self.layout.leg_a,
            leg_b=self.layout.leg_b, bin_i=self.layout.bin_i,
            bin_j=self.layout.bin_j,
        )

    def compute(self, save=None, precompute=None, parallel=True, n_proc=None,
                deriv_method=None, nmodes_mode="edges", mp_context=None):
        """Evaluate the data vector, its derivatives and the covariance.

        This is the slow step.  Results are cached on the instance and can be
        written to ``save`` (an ``.npz``) and reloaded via ``precompute``.

        Parameters
        ----------
        parallel : bool
            Compute the per-parameter derivatives in a process pool.  Results
            are bitwise identical to the serial path: parent and children run
            the same module-level function over the same worker globals.
        n_proc : int, optional
            Defaults to the number of parameters, capped by the CPU affinity
            mask divided by ``OMP_NUM_THREADS``.
        mp_context : str, optional
            Multiprocessing start method; see :data:`DEFAULT_MP_CONTEXT`.
            ``'fork'`` is available but **deadlocks** once CCL has been used in
            the parent, which is always the case by the time the derivatives
            are computed.
        """
        if precompute is not None:
            self._load_npz(precompute)
            return self

        method = deriv_method or self.deriv_method
        state = self._to_state()
        layout_state = self._layout_state()
        overrides = {k: v for k, v in self.param_deriv_method.items() if v}
        init_args = (state, layout_state, self.fid, self.param_step, method,
                     overrides)
        _init_worker(*init_args)

        self._log("Computing the fiducial data vector and covariance ...")
        model = _WORKER["model"]
        cl_all = model.all_cl(self.fid)
        self.data_vector = np.array([
            cl_all[self.layout.leg_a[b], self.layout.leg_b[b]]
            for b in range(self.layout.n_blocks)
        ])
        noise = fl.noise_power(self.n_lens, self.n_source, self.neff_lens,
                               self.neff_source, self.sigma_e)
        nmodes = fl.n_modes(self.ell, self.ell_edges, self.fsky, mode=nmodes_mode)
        self.cov_blocks = fl.gaussian_covariance(cl_all, noise, self.layout, nmodes)

        self.chi_eff_lens_Mpc = ccl.comoving_radial_distance(
            self.cosmo, 1.0 / (1.0 + self.z_eff_lens))

        self._log(f"Computing derivatives for {self.n_params} parameters "
                  f"({'parallel' if parallel else 'serial'}) ...")
        params = list(self.param_order)
        if parallel:
            n_proc = n_proc or self._default_n_proc()
            pool = _make_pool(mp_context or DEFAULT_MP_CONTEXT, n_proc, init_args)
            try:
                results = pool.map(_deriv_one_param_dispatch, params)
            finally:
                pool.close()
                pool.join()
        else:
            results = [_deriv_one_param_dispatch(p) for p in params]

        self.deriv = np.stack(results)
        if not np.all(np.isfinite(self.deriv)):
            bad = [params[i] for i in range(len(params))
                   if not np.all(np.isfinite(self.deriv[i]))]
            warnings.warn(f"non-finite derivatives for: {bad}")
        self._computed = True
        self._log("Done.")

        if save:
            self.save(save)
        return self

    def _default_n_proc(self):
        try:
            avail = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - not linux
            avail = os.cpu_count() or 1
        threads = int(os.environ.get("OMP_NUM_THREADS", "1") or 1)
        return max(1, min(self.n_params, avail // max(1, threads)))

    def check_step_convergence(self, param, factors=(0.5, 1.0, 2.0)):
        """Recompute one derivative at scaled steps; returns max relative change."""
        base = self.param_step[param]
        state = self._to_state()
        out = {}
        for f in factors:
            steps = dict(self.param_step)
            steps[param] = base * f
            _init_worker(state, self._layout_state(), self.fid, steps,
                         self.deriv_method,
                         {k: v for k, v in self.param_deriv_method.items() if v})
            out[f] = _deriv_one_param_dispatch(param)
        ref = out[1.0]
        scale = np.max(np.abs(ref))
        return {f: float(np.max(np.abs(v - ref)) / scale) for f, v in out.items()}

    def prior_stability_report(self, realizations, smooth_grid=(0.0, 0.02, 0.05, 0.1),
                               n_max=200, realizations_axis=None):
        """Photo-z prior widths as a function of realization smoothing.

        A plateau across ``smooth_grid`` means the prior is set by the survey;
        a monotonic trend means it is being set by the binning of the
        realizations instead.

        This is a real hazard for ``nz_model='gaussian_outlier'``: the outlier
        fraction comes from a minimum-ratio statistic, so pixel noise biases it
        low and the prior width can change by factors of a few with the
        smoothing scale.  For ``'shift_stretch'`` the parameters are moments of
        n(z), which average pixel noise down rather than absorbing it, and the
        report should come out flat -- running it is the cheapest way to
        confirm that on your own realizations.

        Returns
        -------
        dict
            ``{smoothing: ndarray (n_tomo, n_pz_per_bin) of prior sigmas}``.
        """
        if self.nz_model == "no_uncertainty":
            raise RuntimeError(
                "nz_model='no_uncertainty' has no photo-z parameters and so no "
                "prior to report on."
            )
        real = nzd.orient_realizations(realizations, self.n_source, realizations_axis)
        real = real[:n_max]
        out = {}
        for s in smooth_grid:
            r = nzd.smooth_nz(self.z, real, s) if s > 0 else real
            p = self._fit_realizations(r, {})
            cov = nzd.photoz_prior_from_realizations(p, mode=self.prior_mode)
            out[float(s)] = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        return out

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path):
        """Write the computed derivatives, covariance and metadata to an npz."""
        if not self._computed:
            raise RuntimeError("call compute() before save().")
        slices = self.layout.probe_block_slices()
        np.savez_compressed(
            path,
            schema_version=SCHEMA_VERSION,
            created_utc=datetime.now(timezone.utc).isoformat(),
            mode=self.mode,
            nz_model=self.nz_model,
            deriv_method=self.deriv_method,
            # grids
            ell=self.ell, ell_edges=self.ell_edges, z_grid=self.z,
            # parameters
            param_names=np.array(self.param_order),
            param_labels=np.array(self.param_labels),
            param_fid=np.array([self.fid[p] for p in self.param_order]),
            param_step=np.array([self.param_step[p] for p in self.param_order]),
            param_prior_sigma=np.array(
                [self.prior_sigma[p] for p in self.param_order]),
            # layout
            block_probe=self.layout.probe,
            block_leg_a=self.layout.leg_a, block_leg_b=self.layout.leg_b,
            block_i=self.layout.bin_i, block_j=self.layout.bin_j,
            probe_names=np.array(fl.PROBES),
            probe_block_start=np.array([slices[p].start for p in fl.PROBES]),
            probe_block_stop=np.array([slices[p].stop for p in fl.PROBES]),
            # payload
            data_vector=self.data_vector, deriv=self.deriv,
            cov_blocks=self.cov_blocks,
            # survey
            fsky=self.fsky, sigma_e=self.sigma_e,
            neff_source=self.neff_source, neff_lens=self.neff_lens,
            n_tomo_source=self.n_source, n_lens=self.n_lens,
            # n(z) model
            nz_source_fid=self.nz_source_fid, nz_out_source=self.nz_out,
            z_lens=self.z_lens, nz_lens=self.nz_lens,
            pz_mu_fid=self.pz_mu, pz_sigma_fid=self.pz_sigma,
            pz_pivot=self.pz_pivot,
            pz_fout_fid=self.pz_fout, pz_prior_cov=self.pz_prior_cov,
            pz_degenerate=self.pz_degenerate, gbias_fid=self.gbias,
            # scale cuts
            z_eff_lens=self.z_eff_lens,
            chi_eff_lens_Mpc=self.chi_eff_lens_Mpc,
            h_fid=self.cosmo_kwargs["h"],
            # config
            cosmo_kwargs_json=json.dumps(self.cosmo_kwargs),
            config_json=json.dumps(dict(
                mode=self.mode, nz_model=self.nz_model, y1=self.y1, step=self.step,
                prior_mode=self.prior_mode, ia_lf_cosmo=self.ia_lf_cosmo,
                omega_b_convention=self.omega_b_convention,
                ggl_pairs=[list(p) for p in self.ggl_pairs],
            )),
        )
        self._log(f"Saved to {path}")
        return path

    def _load_npz(self, path, check_consistency=True):
        """Load a saved npz onto this instance.

        ``check_consistency`` compares the stored mode/n_tomo/z-grid against
        what ``__init__`` was given, which is how a stale npz gets caught.  It
        is switched off by :meth:`from_npz`, where there is no live
        configuration to compare against -- warning there would be pure noise
        and would train users to ignore the real warnings.
        """
        d = np.load(path, allow_pickle=False)
        v = int(d["schema_version"])
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema_version {v}, this fisherA2Z expects "
                f"{SCHEMA_VERSION}. Recompute with compute(save=...)."
            )
        on_disk_mode = str(d["mode"])
        on_disk_nz_model = str(d["nz_model"])
        if check_consistency:
            if on_disk_mode != self.mode:
                warnings.warn(
                    f"{path} was computed with mode={on_disk_mode!r} but this "
                    f"instance has mode={self.mode!r}; using the stored arrays."
                )
            # getattr: a caller may be loading onto an instance that has not
            # run the photo-z setup (the tests do exactly that), in which case
            # there is nothing to compare against and nothing to warn about.
            live_nz_model = getattr(self, "nz_model", None)
            if live_nz_model is not None and on_disk_nz_model != live_nz_model:
                warnings.warn(
                    f"{path} was computed with nz_model={on_disk_nz_model!r} but "
                    f"this instance has nz_model={live_nz_model!r}; the stored "
                    "derivatives are with respect to the stored model's parameters."
                )
            if int(d["n_tomo_source"]) != self.n_source:
                warnings.warn(
                    f"{path} has {int(d['n_tomo_source'])} source bins but this "
                    f"instance has {self.n_source}; the stored arrays win."
                )
            if (d["z_grid"].shape != self.z.shape
                    or not np.allclose(d["z_grid"], self.z)):
                warnings.warn(
                    f"{path} was computed on a different redshift grid; the stored "
                    "derivatives do not correspond to the n(z) given to __init__."
                )
        self.data_vector = d["data_vector"]
        self.deriv = d["deriv"]
        self.cov_blocks = d["cov_blocks"]
        self.chi_eff_lens_Mpc = d["chi_eff_lens_Mpc"]
        self.z_eff_lens = d["z_eff_lens"]
        self.ell = d["ell"]
        self.ell_edges = d["ell_edges"]
        self.param_order = [str(s) for s in d["param_names"]]
        self.param_labels = [str(s) for s in d["param_labels"]]
        self.fid = dict(zip(self.param_order, d["param_fid"].tolist()))
        self.param_step = dict(zip(self.param_order, d["param_step"].tolist()))
        self.prior_sigma = dict(
            zip(self.param_order, d["param_prior_sigma"].tolist()))
        self.pz_prior_cov = d["pz_prior_cov"]
        self.layout = fl.BlockLayout(
            n_lens=int(d["n_lens"]), n_source=int(d["n_tomo_source"]),
            probe=d["block_probe"], leg_a=d["block_leg_a"],
            leg_b=d["block_leg_b"], bin_i=d["block_i"], bin_j=d["block_j"],
        )
        self.mode = on_disk_mode
        # forecast() needs this to group the photo-z parameters for the prior.
        self.nz_model = on_disk_nz_model
        self.n_pz_per_bin = N_PZ_PER_BIN[on_disk_nz_model]
        self._computed = True
        self._log(f"Loaded precomputed derivatives and covariance from {path}")
        return self

    @classmethod
    def from_npz(cls, path, verbose=True):
        """Build a forecast-only instance from a saved npz.

        The returned object can call :meth:`forecast` but not :meth:`compute`;
        no n(z) decomposition is redone and no CCL evaluation happens.
        """
        obj = cls.__new__(cls)
        obj.verbose = verbose
        obj._load_npz(path, check_consistency=False)
        d = np.load(path, allow_pickle=False)
        obj.n_source = int(d["n_tomo_source"])
        obj.n_lens = int(d["n_lens"])
        obj.z = d["z_grid"]
        obj.fsky = float(d["fsky"])
        obj.sigma_e = float(d["sigma_e"])
        obj.neff_source = d["neff_source"]
        obj.neff_lens = d["neff_lens"]
        obj.pz_mu = d["pz_mu_fid"]
        obj.pz_sigma = d["pz_sigma_fid"]
        obj.pz_pivot = d["pz_pivot"]
        obj.pz_fout = d["pz_fout_fid"]
        obj.nz_out = d["nz_out_source"]
        obj.nz_source_fid = d["nz_source_fid"]
        obj.z_lens = d["z_lens"]
        obj.nz_lens = d["nz_lens"]
        obj.gbias = d["gbias_fid"]
        obj.cosmo_kwargs = json.loads(str(d["cosmo_kwargs_json"]))
        cfg = json.loads(str(d["config_json"]))
        obj.y1 = cfg["y1"]
        obj.step = cfg["step"]
        obj.prior_mode = cfg["prior_mode"]
        obj.ia_lf_cosmo = cfg["ia_lf_cosmo"]
        obj.omega_b_convention = cfg["omega_b_convention"]
        obj.ggl_pairs = [tuple(p) for p in cfg["ggl_pairs"]]
        obj.deriv_method = str(d["deriv_method"])
        obj.param_deriv_method = {p: None for p in obj.param_order}
        obj.cosmo = None
        return obj

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------

    def default_ell_cuts(self, k_max=0.3):
        """A2Z/SRD scale cuts: ell <= 3000 for shear, k <= 0.3 h/Mpc otherwise."""
        if self.chi_eff_lens_Mpc is None:
            raise RuntimeError("call compute() first; the k-cut needs chi(z).")
        lmax_k = fl.ell_max_kcut(self.chi_eff_lens_Mpc,
                                 float(self.cosmo_kwargs["h"]), k_max)
        return {"cs": (None, 3000.0), "ggl": (None, lmax_k), "gc": (None, lmax_k)}

    def forecast(self, ell_min_cs=None, ell_max_cs=None,
                 ell_min_ggl=None, ell_max_ggl=None,
                 ell_min_gc=None, ell_max_gc=None,
                 bin_pairs_cs=None, bin_pairs_ggl=None, bin_pairs_gc=None,
                 mask=None, probes=None, drop_params=None, k_max=0.3,
                 prior=True):
        """Assemble the Fisher matrix under a set of scale cuts.

        Fast: pure linear algebra on the cached arrays, no CCL.  Every ``ell_*``
        argument defaults to the A2Z/SRD cut (see :meth:`default_ell_cuts`);
        pass an explicit value to override, or ``mask=`` for full control.

        Parameters
        ----------
        bin_pairs_cs, bin_pairs_ggl, bin_pairs_gc : sequence of (i, j), optional
            Restrict each probe to these bin pairs.  Defaults to all available.
        drop_params : sequence of str, optional
            Parameters to remove from the forecast entirely (not marginalized
            -- fixed at their fiducial values).
        prior : bool
            Add the Gaussian priors.
        """
        if not self._computed:
            raise RuntimeError("call compute() (or compute(precompute=...)) first.")

        defaults = self.default_ell_cuts(k_max)
        if mask is None:
            cuts = {
                "cs": (ell_min_cs,
                       defaults["cs"][1] if ell_max_cs is None else ell_max_cs),
                "ggl": (ell_min_ggl,
                        defaults["ggl"][1] if ell_max_ggl is None else ell_max_ggl),
                "gc": (ell_min_gc,
                       defaults["gc"][1] if ell_max_gc is None else ell_max_gc),
            }
            block_select = {"cs": bin_pairs_cs, "ggl": bin_pairs_ggl,
                            "gc": bin_pairs_gc}
            mask = fl.build_mask(self.layout, self.ell, ell_cuts=cuts,
                                 block_select=block_select, probes=probes)
        mask = np.asarray(mask, dtype=bool)

        keep = [i for i, p in enumerate(self.param_order)
                if not (drop_params and p in drop_params)]
        deriv = self.deriv[keep]
        names = [self.param_order[i] for i in keep]
        labels = [self.param_labels[i] for i in keep]

        fisher, n_data = fl.fisher_from_derivs(deriv, self.cov_blocks, mask)

        prior_sigma = np.array([self.prior_sigma[p] for p in names])
        if prior:
            fisher = fisher + np.diag(1.0 / prior_sigma ** 2)
            if self.prior_mode == "full" and getattr(self, "pz_prior_cov", None) is not None:
                fisher = self._add_photoz_prior_correlations(fisher, names,
                                                             prior_sigma)

        return FisherFlexResult(
            fisher=fisher, param_order=names, param_labels=labels,
            param_fid={p: self.fid[p] for p in names}, mask=mask,
            n_data_used=n_data, prior_sigma=prior_sigma,
        )

    # ------------------------------------------------------------------
    # parameter bias from a wrong n(z)
    # ------------------------------------------------------------------

    def forecast_bias(self, nz_truth, forecast_params=None, z_truth=None,
                      model=None):
        """Parameter bias induced by analysing data drawn from ``nz_truth``.

        Forward-models the data vector twice -- once with the fiducial source
        n(z), once with ``nz_truth`` -- and propagates the residual through
        Eq. (13) of Zhang et al. (2025):

        ``delta_p = F^-1 . (dC_l/dp . Cov^-1 . (C_l^truth - C_l^fid))``

        Only the *source* n(z) is replaced; the lens sample stays fiducial.

        Parameters
        ----------
        nz_truth : ndarray, shape (n_source, n_z)
            The true source n(z).  Rows are renormalized to unit integral.
        forecast_params : dict, optional
            Passed verbatim to :meth:`forecast`, so the bias is evaluated under
            exactly the scale cuts, probes, priors and ``drop_params`` of the
            forecast it is quoted against.
        z_truth : ndarray, optional
            Redshift grid of ``nz_truth`` if it differs from ``self.z``; each
            row is interpolated onto ``self.z``.
        model : _FlexModel, optional
            A pre-built forward model to reuse.  Building one re-reads the
            k- and e-correction tables and rebuilds the luminosity function
            (~0.3 s), so pass it when scanning many ``nz_truth``.

        Returns
        -------
        FisherFlexBias

        Notes
        -----
        Two caveats worth knowing before quoting the number.  First, the paper
        notes this *overestimates* the realized bias: the Fisher posterior sits
        at the centre of the prior, while an MCMC lets the photo-z nuisances
        drift toward the truth and absorb part of the systematic.  Second,
        ``drop_params`` *fixes* parameters rather than marginalizing them, so
        dropped nuisances can no longer absorb anything and the bias on the
        survivors grows.
        """
        if not self._computed:
            raise RuntimeError("call compute() (or compute(precompute=...)) first.")

        nz_truth = np.atleast_2d(np.asarray(nz_truth, dtype=float))
        if nz_truth.shape[0] != self.n_source:
            raise ValueError(
                f"nz_truth has {nz_truth.shape[0]} tomographic bins but this "
                f"forecast has {self.n_source}."
            )
        if z_truth is not None:
            z_truth = np.asarray(z_truth, dtype=float)
            nz_truth = np.array([np.interp(self.z, z_truth, row, left=0.0,
                                           right=0.0) for row in nz_truth])
        elif nz_truth.shape[1] != self.z.size:
            raise ValueError(
                f"nz_truth has {nz_truth.shape[1]} redshift samples but the "
                f"grid has {self.z.size}; pass z_truth= to interpolate."
            )
        if np.any(nz_truth < 0):
            warnings.warn("nz_truth has negative values; clipping to zero.")
            nz_truth = np.clip(nz_truth, 0.0, None)
        #try:
        norm = np.trapz(nz_truth, self.z, axis=-1)
        if np.any(norm <= 0):
            raise ValueError("nz_truth has a bin with zero integral.")
        nz_truth = nz_truth / norm[:, None]

        res = self.forecast(**(forecast_params or {}))

        m = model if model is not None else self._model()
        d_cen = m.data_vector(self.fid, self.layout, nz_src=self.nz_source_fid)
        d_true = m.data_vector(self.fid, self.layout, nz_src=nz_truth)

        # The re-modelled baseline must reproduce the cached fiducial vector.
        # If it does not, the npz is stale or the cosmology has moved, and the
        # stored derivatives no longer describe this model.
        scale = np.maximum(np.abs(self.data_vector), 1e-300)
        drift = np.max(np.abs(d_cen - self.data_vector) / scale)
        if drift > 1e-8:
            warnings.warn(
                f"the re-modelled fiducial data vector differs from the cached "
                f"one by up to {drift:.2e} (relative); the stored derivatives "
                "may not correspond to this model."
            )

        delta = d_true - d_cen
        keep = [self.param_order.index(p) for p in res.param_order]
        bias_vec, n_used = fl.bias_vector_from_derivs(
            self.deriv[keep], self.cov_blocks, delta, res.mask)
        # res.cov rather than a fresh inverse, so the shifts stay exactly
        # consistent with the sigmas they get divided by.
        delta_p = res.cov @ bias_vec

        return FisherFlexBias(
            delta_p={p: float(v) for p, v in zip(res.param_order, delta_p)},
            bias_vec=bias_vec, result=res, delta_cl=delta,
            data_vector=self.data_vector, nz_truth=nz_truth,
            nz_fid=np.asarray(self.nz_source_fid), z=self.z,
            n_data_used=n_used,
        )

    def _add_photoz_prior_correlations(self, fisher, names, prior_sigma):
        """Replace the diagonal photo-z prior with the full per-bin block."""
        fisher = fisher.copy()
        for b in range(self.n_source):
            group = self._pz_param_names(b)
            if not group or not all(t in names for t in group):
                continue
            idx = [names.index(t) for t in group]
            cov = self.pz_prior_cov[b]
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                warnings.warn(f"photo-z prior covariance for bin {b} is singular; "
                              "using the diagonal.")
                continue
            # Undo the diagonal contribution added above, then add the block.
            for k, i in enumerate(idx):
                fisher[i, i] -= 1.0 / prior_sigma[i] ** 2
            fisher[np.ix_(idx, idx)] += inv
        return fisher


#: Default multiprocessing start method.
#:
#: It has to be ``forkserver`` or ``spawn``, not ``fork``.  CCL's GSL/OpenMP
#: internals hold locks that a bare ``fork`` inherits in a locked state, so a
#: forked child deadlocks the moment it calls into CCL -- and by the time the
#: derivatives are computed the parent has already evaluated the fiducial data
#: vector and the covariance.  ``forkserver`` forks from a clean, pre-CCL
#: server process, so it is immune.
DEFAULT_MP_CONTEXT = "forkserver"


@contextlib.contextmanager
def _suppress_main_fixup():
    """Stop ``spawn``/``forkserver`` children from re-importing ``__main__``.

    Both start methods normally re-execute the parent's ``__main__`` in every
    child.  That is unnecessary here -- the worker is a module-level function
    in an installed package, so the child only needs to import
    ``fisherA2Z.fisher_flex`` -- and it is actively harmful:

    - running from stdin or a notebook, ``__main__.__file__`` may be
      ``'<stdin>'``, and the child dies in a respawn loop on
      ``FileNotFoundError``;
    - running a plain script with no ``if __name__ == '__main__'`` guard, the
      child re-runs the whole script.

    Blanking ``__file__``/``__spec__`` for the duration of pool creation makes
    ``multiprocessing.spawn.get_preparation_data`` skip the fixup entirely.
    """
    main = sys.modules.get("__main__")
    if main is None:  # pragma: no cover - defensive
        yield
        return
    had_file = hasattr(main, "__file__")
    saved_file = getattr(main, "__file__", None)
    saved_spec = getattr(main, "__spec__", None)
    if had_file:
        del main.__file__
    main.__spec__ = None
    try:
        yield
    finally:
        if had_file:
            main.__file__ = saved_file
        main.__spec__ = saved_spec


def _make_pool(context, n_proc, init_args):
    """Create a worker pool that is safe to use after CCL has run."""
    ctx = mp.get_context(context)
    if context == "forkserver":
        try:
            ctx.set_forkserver_preload(["fisherA2Z.fisher_flex"])
        except Exception:  # pragma: no cover - preload is only an optimization
            pass
    with _suppress_main_fixup():
        return ctx.Pool(n_proc, initializer=_init_worker, initargs=init_args)


def _deriv_one_param_dispatch(param):
    """Apply the per-parameter method override, then differentiate."""
    override = _WORKER.get("overrides", {}).get(param)
    if override:
        saved = _WORKER["deriv_method"]
        _WORKER["deriv_method"] = override
        try:
            return _deriv_one_param(param)
        finally:
            _WORKER["deriv_method"] = saved
    return _deriv_one_param(param)


# --------------------------------------------------------------------------
# module helpers
# --------------------------------------------------------------------------


def _cosmo_to_kwargs(cosmo):
    """Extract the constructor kwargs we vary from a ``ccl.Cosmology``."""
    return dict(
        Omega_c=float(cosmo["Omega_c"]),
        Omega_b=float(cosmo["Omega_b"]),
        h=float(cosmo["h"]),
        sigma8=float(cosmo["sigma8"]),
        n_s=float(cosmo["n_s"]),
        w0=float(cosmo["w0"]),
        wa=float(cosmo["wa"]),
    )


def _default_lens_sample(n_lens, y1=False):
    """The A2Z LSST lens sample: SRD dN/dz convolved with a Gaussian core.

    Reproduces ``Fisher._makeLensPZ``: bins uniform in 0.2-1.2, photo-z scatter
    ``sigma_z (1+z)`` with ``sigma_z = 0.03`` (Y10) or ``0.06`` (Y1).  Note the
    ``(1+z)`` scaling applies on the *lens* side only -- the source photo-z
    parameters of FisherFlex are in absolute z units.
    """
    import pandas as pd

    fname = "/data/nzdist_y1.txt" if y1 else "/data/nzdist.txt"
    df = pd.read_csv(_PACKAGE_PATH + fname, sep=" ")
    z = np.array(df["zmid"], dtype=float)
    dneff = np.array(df["dneff"], dtype=float)
    sigma_z = 0.06 if y1 else 0.03

    edges = np.linspace(0.2, 1.2, n_lens + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = dneff / np.trapz(dneff, z)

    nz = np.empty((n_lens, z.size))
    weights = np.empty(n_lens)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        tomofilter = uniform.pdf(z, loc=lo, scale=hi - lo)
        joint = np.empty((z.size, z.size))
        for k in range(z.size):
            scale = sigma_z * (1.0 + z[k])
            core = norm.pdf((z - z[k]) / scale) / scale
            core = core / np.trapz(core, z)
            joint[:, k] = core * pdf[k] * tomofilter[k]
        row = np.trapz(joint, z, axis=1)
        weights[i] = np.trapz(row, z)
        nz[i] = row / weights[i]
    weights = weights / weights.sum()
    z_eff = centers
    return z, nz, z_eff, weights
