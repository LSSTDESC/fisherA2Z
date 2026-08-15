"""Data-vector layout, Gaussian covariance and Fisher algebra for FisherFlex.

Pure numpy/scipy: this module never imports CCL or :mod:`fisherA2Z.fisher`, so
the bulk of the FisherFlex test suite runs in milliseconds.

Conventions
-----------
*Legs* are the individual tracers.  Legs ``0 .. n_lens-1`` are the lens
(number-counts) bins; legs ``n_lens .. n_lens+n_source-1`` are the source
(weak-lensing) bins.

*Blocks* are the entries of the data vector, each carrying ``n_ell`` values.
They are ordered ``[cosmic shear, GGL, clustering]`` to match
``Fisher.makeFidCells``:

- ``cs``  -- every source pair ``i <= j``, ``i`` outer;
- ``ggl`` -- the selected lens x source pairs, lens outer;
- ``gc``  -- the lens auto-spectra only.

The Gaussian covariance is exactly block diagonal in ell, so it is stored as
``cov_blocks`` of shape ``(n_ell, n_blocks, n_blocks)``.  That is not just a
memory saving: any scale cut selects a subset of ``(block, ell)`` pairs, so the
*masked* covariance is still ell-block-diagonal and the Fisher matrix is a sum
of small independent Cholesky solves rather than one large ill-conditioned
inversion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

__all__ = [
    "BlockLayout",
    "ARCMIN2_PER_SR",
    "PROBES",
    "MODE_PROBES",
    "default_ell_edges",
    "ell_from_edges",
    "n_modes",
    "select_ggl_pairs",
    "A2Z_GGL_ACCEPT_Y10",
    "A2Z_GGL_ACCEPT_Y1",
    "build_layout",
    "noise_power",
    "gaussian_covariance",
    "ell_max_kcut",
    "build_mask",
    "fisher_from_derivs",
    "bias_vector_from_derivs",
]

#: Steradians per square arcminute, i.e. (180 * 60 / pi)^2.
ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2

PROBES = ("cs", "ggl", "gc")

MODE_PROBES = {
    "3x2pt": ("cs", "ggl", "gc"),
    "2x2pt": ("ggl", "gc"),
    "cosmic_shear": ("cs",),
}

#: The hard-coded lens x source pairs used by ``Fisher.makePosShearCells``.
A2Z_GGL_ACCEPT_Y10 = {
    (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4),
    (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (7, 4), (8, 4), (9, 4),
}
A2Z_GGL_ACCEPT_Y1 = {(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)}


# --------------------------------------------------------------------------
# multipole grid
# --------------------------------------------------------------------------


def default_ell_edges(ell_min=20.0, ell_max=15000.0, n_ell=20):
    """The A2Z multipole bin edges: 20 log-spaced bins from 20 to 15000."""
    return np.geomspace(ell_min, ell_max, n_ell + 1)


def ell_from_edges(edges):
    """Bin centres as the geometric mean of the edges.

    This reproduces ``data/ell-values.txt`` used by ``Fisher.getElls`` exactly.
    """
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[1:] * edges[:-1])


def n_modes(ell, edges, fsky, mode="edges"):
    """Number of independent modes per multipole bin, ``(2l+1) dl fsky``.

    ``mode='edges'`` uses the exact bin widths ``edges[1:] - edges[:-1]``.
    ``mode='exact_sum'`` uses ``sum_{l in bin} (2l+1)``, the correct discrete
    count.  The rail reference implementation uses ``ell * dlog(ell)``, which
    differs from the exact width by ~0.5% and shifts every forecast sigma by
    ~0.25%; it is deliberately not offered here.
    """
    ell = np.asarray(ell, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if mode == "edges":
        d_ell = edges[1:] - edges[:-1]
        return (2.0 * ell + 1.0) * d_ell * fsky
    if mode == "exact_sum":
        out = np.empty(ell.size)
        for i in range(ell.size):
            lo = int(np.ceil(edges[i]))
            hi = int(np.floor(edges[i + 1]))
            ells = np.arange(lo, hi + 1)
            out[i] = np.sum(2.0 * ells + 1.0) * fsky
        return out
    raise ValueError(f"unknown n_modes mode {mode!r}; use 'edges' or 'exact_sum'.")


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def select_ggl_pairs(z_mean_lens, z_mean_source, offset=0.3):
    """Lens-source pairs with ``mean(z)_source > mean(z)_lens + offset``."""
    z_mean_lens = np.atleast_1d(np.asarray(z_mean_lens, dtype=float))
    z_mean_source = np.atleast_1d(np.asarray(z_mean_source, dtype=float))
    return [
        (l, s)
        for l in range(z_mean_lens.size)
        for s in range(z_mean_source.size)
        if z_mean_source[s] > z_mean_lens[l] + offset
    ]


@dataclass
class BlockLayout:
    """Description of the data-vector blocks.

    Attributes
    ----------
    n_lens, n_source : int
    probe : ndarray of str, shape (n_blocks,)
        One of ``'cs'``, ``'ggl'``, ``'gc'``.
    leg_a, leg_b : ndarray of int, shape (n_blocks,)
        Global leg indices of the two tracers in each block.
    bin_i, bin_j : ndarray of int, shape (n_blocks,)
        Per-probe bin indices.  For ``ggl`` these are ``(lens, source)``.
    """

    n_lens: int
    n_source: int
    probe: np.ndarray
    leg_a: np.ndarray
    leg_b: np.ndarray
    bin_i: np.ndarray
    bin_j: np.ndarray

    @property
    def n_blocks(self) -> int:
        return len(self.probe)

    @property
    def n_legs(self) -> int:
        return self.n_lens + self.n_source

    def probe_slice(self, probe):
        """Contiguous ``slice`` of blocks belonging to ``probe``."""
        idx = np.flatnonzero(self.probe == probe)
        if idx.size == 0:
            return slice(0, 0)
        return slice(int(idx[0]), int(idx[-1]) + 1)

    def probe_block_slices(self):
        return {p: self.probe_slice(p) for p in PROBES}

    def pairs(self, probe):
        """List of ``(i, j)`` bin pairs for ``probe``, in block order."""
        sl = self.probe_slice(probe)
        return list(zip(self.bin_i[sl].tolist(), self.bin_j[sl].tolist()))

    def is_lens_leg(self, leg):
        return np.asarray(leg) < self.n_lens

    def lens_bin_of_block(self):
        """Lens bin index per block, or -1 for cosmic-shear blocks."""
        out = np.full(self.n_blocks, -1, dtype=int)
        for b in range(self.n_blocks):
            if self.probe[b] in ("ggl", "gc"):
                out[b] = self.bin_i[b]
        return out


def build_layout(n_lens, n_source, ggl_pairs, mode="3x2pt"):
    """Build the ``[cs, ggl, gc]`` block layout.

    Parameters
    ----------
    ggl_pairs : sequence of (lens, source)
        Accepted lens-source pairs.  Sorted into lens-major order to match
        ``Fisher.makePosShearCells``.
    mode : {'3x2pt', '2x2pt', 'cosmic_shear'}
        Which probes to include.
    """
    if mode not in MODE_PROBES:
        raise ValueError(f"unknown mode {mode!r}; use one of {sorted(MODE_PROBES)}.")
    probes = MODE_PROBES[mode]

    probe, leg_a, leg_b, bin_i, bin_j = [], [], [], [], []

    if "cs" in probes:
        for i in range(n_source):
            for j in range(i, n_source):
                probe.append("cs")
                leg_a.append(n_lens + i)
                leg_b.append(n_lens + j)
                bin_i.append(i)
                bin_j.append(j)

    if "ggl" in probes:
        for (l, s) in sorted(ggl_pairs):
            if not (0 <= l < n_lens and 0 <= s < n_source):
                raise ValueError(f"ggl pair {(l, s)} out of range for "
                                 f"n_lens={n_lens}, n_source={n_source}.")
            probe.append("ggl")
            leg_a.append(l)
            leg_b.append(n_lens + s)
            bin_i.append(l)
            bin_j.append(s)

    if "gc" in probes:
        for l in range(n_lens):
            probe.append("gc")
            leg_a.append(l)
            leg_b.append(l)
            bin_i.append(l)
            bin_j.append(l)

    return BlockLayout(
        n_lens=int(n_lens),
        n_source=int(n_source),
        probe=np.array(probe, dtype="<U3"),
        leg_a=np.array(leg_a, dtype=int),
        leg_b=np.array(leg_b, dtype=int),
        bin_i=np.array(bin_i, dtype=int),
        bin_j=np.array(bin_j, dtype=int),
    )


# --------------------------------------------------------------------------
# covariance
# --------------------------------------------------------------------------


def noise_power(n_lens, n_source, neff_lens, neff_source, sigma_e):
    """Per-leg shot/shape noise power spectrum (flat in ell).

    Lens legs get ``1 / n_l``; source legs get ``sigma_e**2 / n_s``, with the
    number densities converted from arcmin^-2 to sr^-1.  ``sigma_e`` is the
    *per-component* ellipticity dispersion, matching the LSST SRD and A2Z --
    i.e. the noise is ``sigma_e^2 / n``, not ``sigma_e^2 / (2 n)`` and not
    ``2 sigma_e^2 / n``.

    Lens and source samples are assumed disjoint, so there is no lens-source
    noise cross term.
    """
    neff_lens = np.atleast_1d(np.asarray(neff_lens, dtype=float))
    neff_source = np.atleast_1d(np.asarray(neff_source, dtype=float))
    if neff_lens.size != n_lens:
        raise ValueError(f"neff_lens has {neff_lens.size} entries, expected {n_lens}.")
    if neff_source.size != n_source:
        raise ValueError(f"neff_source has {neff_source.size} entries, expected {n_source}.")
    noise = np.zeros(n_lens + n_source)
    noise[:n_lens] = 1.0 / (neff_lens * ARCMIN2_PER_SR)
    noise[n_lens:] = sigma_e ** 2 / (neff_source * ARCMIN2_PER_SR)
    return noise


def gaussian_covariance(cl_all, noise, layout, nmodes):
    """Gaussian (Wick) covariance of the block data vector.

    .. math::
        \\mathrm{Cov}(C_\\ell^{AB}, C_\\ell^{CD}) =
        \\frac{\\tilde C_\\ell^{AC}\\tilde C_\\ell^{BD}
             + \\tilde C_\\ell^{AD}\\tilde C_\\ell^{BC}}
             {(2\\ell+1)\\,\\Delta\\ell\\,f_\\mathrm{sky}}

    with :math:`\\tilde C = C + N\\delta_{AB}`.

    Parameters
    ----------
    cl_all : ndarray, shape (n_legs, n_legs, n_ell)
        Signal spectra for *all* leg pairs -- including lens cross-bin spectra
        and rejected lens-source pairs, which enter the Wick contractions even
        though they are not in the data vector.
    noise : ndarray, shape (n_legs,)
        Per-leg noise power (see :func:`noise_power`).
    nmodes : ndarray, shape (n_ell,)

    Returns
    -------
    cov_blocks : ndarray, shape (n_ell, n_blocks, n_blocks)
    """
    cl_all = np.asarray(cl_all, dtype=float)
    n_legs = layout.n_legs
    if cl_all.shape[:2] != (n_legs, n_legs):
        raise ValueError(
            f"cl_all has shape {cl_all.shape}, expected ({n_legs}, {n_legs}, n_ell)."
        )
    ct = cl_all.copy()
    idx = np.arange(n_legs)
    ct[idx, idx, :] += np.asarray(noise, dtype=float)[:, None]
    ct = 0.5 * (ct + ct.transpose(1, 0, 2))  # enforce leg symmetry exactly

    ctl = np.ascontiguousarray(ct.transpose(2, 0, 1))  # (n_ell, n_legs, n_legs)
    a = layout.leg_a
    b = layout.leg_b
    cov = (
        ctl[:, a[:, None], a[None, :]] * ctl[:, b[:, None], b[None, :]]
        + ctl[:, a[:, None], b[None, :]] * ctl[:, b[:, None], a[None, :]]
    ) / np.asarray(nmodes, dtype=float)[:, None, None]
    return 0.5 * (cov + cov.transpose(0, 2, 1))


# --------------------------------------------------------------------------
# scale cuts
# --------------------------------------------------------------------------


def ell_max_kcut(chi_Mpc, h, k_max=0.3):
    """``ell_max`` from a comoving wavenumber cut, Limber-style.

    ``k_max`` is in h/Mpc and ``chi_Mpc`` in Mpc, so the conversion needs an
    explicit factor of ``h``::

        ell_max = k_max * (chi_Mpc * h) - 0.5

    Dropping that ``h`` inflates ell_max by 1/h ~ 1.49 and spuriously tightens
    the clustering and GGL constraints by roughly 20%.
    """
    return k_max * (np.asarray(chi_Mpc, dtype=float) * h) - 0.5


def build_mask(layout, ell, ell_cuts=None, block_select=None, probes=None):
    """Boolean ``(n_blocks, n_ell)`` scale-cut mask.

    Parameters
    ----------
    ell_cuts : dict, optional
        ``{'cs': (lmin, lmax), 'ggl': ..., 'gc': ...}``.  Either bound may be
        ``None``.  ``lmax`` (or ``lmin``) may also be an array indexed by the
        block's *lens* bin, which is how a per-lens-bin k_max cut is expressed
        with no special-case code.
    block_select : dict, optional
        ``{'cs': [(i, j), ...], ...}`` restricting which bin pairs are used.
    probes : sequence of str, optional
        Probes to keep at all; everything else is masked out entirely.
    """
    ell = np.asarray(ell, dtype=float)
    mask = np.ones((layout.n_blocks, ell.size), dtype=bool)
    lens_bin = layout.lens_bin_of_block()

    if probes is not None:
        keep = np.isin(layout.probe, list(probes))
        mask &= keep[:, None]

    if block_select:
        for probe, pairs in block_select.items():
            if pairs is None:
                continue
            allowed = {tuple(p) for p in pairs}
            for b in range(layout.n_blocks):
                if layout.probe[b] == probe:
                    if (int(layout.bin_i[b]), int(layout.bin_j[b])) not in allowed:
                        mask[b] = False

    if ell_cuts:
        for probe, bounds in ell_cuts.items():
            if bounds is None:
                continue
            lmin, lmax = bounds
            for b in range(layout.n_blocks):
                if layout.probe[b] != probe:
                    continue
                lo = _resolve_bound(lmin, lens_bin[b])
                hi = _resolve_bound(lmax, lens_bin[b])
                if lo is not None:
                    mask[b] &= ell >= lo
                if hi is not None:
                    mask[b] &= ell <= hi
    return mask


def _resolve_bound(bound, lens_bin):
    if bound is None:
        return None
    arr = np.atleast_1d(bound)
    if arr.size == 1:
        return float(arr[0])
    if lens_bin < 0:
        raise ValueError(
            "a per-lens-bin ell cut was given for a probe with no lens bin "
            "(cosmic shear); pass a scalar instead."
        )
    return float(arr[lens_bin])


# --------------------------------------------------------------------------
# Fisher algebra
# --------------------------------------------------------------------------


def fisher_from_derivs(deriv, cov_blocks, mask=None, jitter=0.0):
    """Assemble the Fisher matrix by summing independent per-ell solves.

    ``F = sum_l  D_l[mask_l]^T  Cov_l[mask_l, mask_l]^{-1}  D_l[mask_l]``

    Parameters
    ----------
    deriv : ndarray, shape (n_par, n_blocks, n_ell)
    cov_blocks : ndarray, shape (n_ell, n_blocks, n_ell) -> (n_ell, n_blocks, n_blocks)
    mask : ndarray of bool, shape (n_blocks, n_ell), optional
        ``None`` means use everything.

    Returns
    -------
    fisher : ndarray, shape (n_par, n_par)
    n_data : int
        Number of data-vector elements actually used.
    """
    deriv = np.asarray(deriv, dtype=float)
    cov_blocks = np.asarray(cov_blocks, dtype=float)
    n_par, n_blocks, n_ell = deriv.shape
    if cov_blocks.shape != (n_ell, n_blocks, n_blocks):
        raise ValueError(
            f"cov_blocks has shape {cov_blocks.shape}, expected "
            f"({n_ell}, {n_blocks}, {n_blocks})."
        )
    if mask is None:
        mask = np.ones((n_blocks, n_ell), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    fisher = np.zeros((n_par, n_par))
    n_data = 0
    for l in range(n_ell):
        sel = np.flatnonzero(mask[:, l])
        if sel.size == 0:
            continue
        n_data += sel.size
        d = deriv[:, sel, l].T  # (n_sel, n_par)
        x = _solve_block(cov_blocks[l][np.ix_(sel, sel)], d, l, jitter)
        fisher += d.T @ x
    return 0.5 * (fisher + fisher.T), n_data


def _solve_block(c, rhs, l, jitter=0.0):
    """``Cov_l^{-1} rhs`` for one ell block, with a warned pinv fallback.

    Shared by :func:`fisher_from_derivs` and :func:`bias_vector_from_derivs` so
    the two cannot drift apart in how they treat an ill-conditioned block.
    """
    if jitter:
        c = c + jitter * np.diag(np.diag(c))
    try:
        cf = cho_factor(c, lower=True, check_finite=False)
        return cho_solve(cf, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"covariance block at ell index {l} is not positive definite; "
            "falling back to a pseudo-inverse."
        )
        return np.linalg.pinv(c) @ rhs


def bias_vector_from_derivs(deriv, cov_blocks, delta, mask=None, jitter=0.0):
    """Numerator of the Fisher-bias formula, summed over independent ell blocks.

    ``B_a = sum_l  D_l[mask_l, a]^T  Cov_l[mask_l, mask_l]^{-1}  delta_l[mask_l]``

    This is the bracketed term of Eq. (13) of Zhang et al. (2025); the parameter
    bias itself is ``F^{-1} B`` with ``F`` the *same* Fisher matrix returned by
    :func:`fisher_from_derivs` under the *same* ``mask``, plus its priors.

    Parameters
    ----------
    deriv : ndarray, shape (n_par, n_blocks, n_ell)
    cov_blocks : ndarray, shape (n_ell, n_blocks, n_blocks)
    delta : ndarray, shape (n_blocks, n_ell)
        Residual data vector, ``C_ell^biased - C_ell^fiducial``.
    mask : ndarray of bool, shape (n_blocks, n_ell), optional
        ``None`` means use everything.

    Returns
    -------
    bias_vec : ndarray, shape (n_par,)
    n_data : int
        Number of data-vector elements actually used.
    """
    deriv = np.asarray(deriv, dtype=float)
    cov_blocks = np.asarray(cov_blocks, dtype=float)
    delta = np.asarray(delta, dtype=float)
    n_par, n_blocks, n_ell = deriv.shape
    if cov_blocks.shape != (n_ell, n_blocks, n_blocks):
        raise ValueError(
            f"cov_blocks has shape {cov_blocks.shape}, expected "
            f"({n_ell}, {n_blocks}, {n_blocks})."
        )
    if delta.shape != (n_blocks, n_ell):
        raise ValueError(
            f"delta has shape {delta.shape}, expected ({n_blocks}, {n_ell})."
        )
    if mask is None:
        mask = np.ones((n_blocks, n_ell), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    bias_vec = np.zeros(n_par)
    n_data = 0
    for l in range(n_ell):
        sel = np.flatnonzero(mask[:, l])
        if sel.size == 0:
            continue
        n_data += sel.size
        d = deriv[:, sel, l].T  # (n_sel, n_par)
        x = _solve_block(cov_blocks[l][np.ix_(sel, sel)], delta[sel, l], l, jitter)
        bias_vec += d.T @ x
    return bias_vec, n_data


def dense_covariance(cov_blocks, mask=None):
    """Expand ``cov_blocks`` into a dense ``(n_data, n_data)`` matrix.

    Only useful for tests and for exporting to other tools -- the block form is
    what the forecast actually uses.  The flat data-vector ordering is
    block-major (all ell of block 0, then block 1, ...), matching
    ``Fisher.makeFidCells``.
    """
    cov_blocks = np.asarray(cov_blocks, dtype=float)
    n_ell, n_blocks, _ = cov_blocks.shape
    if mask is None:
        mask = np.ones((n_blocks, n_ell), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    flat = [(b, l) for b in range(n_blocks) for l in range(n_ell) if mask[b, l]]
    out = np.zeros((len(flat), len(flat)))
    for p, (b1, l1) in enumerate(flat):
        for q, (b2, l2) in enumerate(flat):
            if l1 == l2:
                out[p, q] = cov_blocks[l1, b1, b2]
    return out
