"""Tests for the analytic Gaussian covariance.

The reference implementation used here is a literal transcription of the triple
loop in ``rail/evaluation/metrics/cosmic_shear_snr.py``, which is what the
vectorized version replaced.
"""

import numpy as np
import pytest

from fisherA2Z import flex_layout as fl


def _reference_gaussian_covariance(cl_all, noise, layout, nmodes):
    """Explicit-loop Wick covariance, written the slow obvious way."""
    n_blocks, n_ell = layout.n_blocks, len(nmodes)
    ct = cl_all.copy()
    for a in range(layout.n_legs):
        ct[a, a, :] += noise[a]

    cov = np.zeros((n_ell, n_blocks, n_blocks))
    for l in range(n_ell):
        for p in range(n_blocks):
            a, b = layout.leg_a[p], layout.leg_b[p]
            for q in range(n_blocks):
                c, d = layout.leg_a[q], layout.leg_b[q]
                cov[l, p, q] = (
                    ct[a, c, l] * ct[b, d, l] + ct[a, d, l] * ct[b, c, l]
                ) / nmodes[l]
    return cov


def test_matches_the_explicit_loop_reference(toy_cl_all, toy_layout, toy_ell):
    noise = fl.noise_power(3, 2, np.full(3, 4.0), np.full(2, 10.0), 0.26)
    nmodes = np.array([1e4, 5e4, 2e5])

    fast = fl.gaussian_covariance(toy_cl_all, noise, toy_layout, nmodes)
    slow = _reference_gaussian_covariance(toy_cl_all, noise, toy_layout, nmodes)

    assert fast.shape == (3, toy_layout.n_blocks, toy_layout.n_blocks)
    assert np.allclose(fast, slow, rtol=1e-14, atol=0.0)


def test_covariance_is_symmetric_and_cholesky_factorable(toy_cov):
    for l in range(toy_cov.shape[0]):
        assert np.array_equal(toy_cov[l], toy_cov[l].T), "not bitwise symmetric"
        np.linalg.cholesky(toy_cov[l])  # raises if not positive definite


def test_single_auto_block_variance_is_twice_ct_squared():
    """For one auto-spectrum, Cov = 2 (C + N)^2 / nmodes -- the textbook case."""
    layout = fl.build_layout(0, 1, [], mode="cosmic_shear")
    cl = np.full((1, 1, 2), 3e-9)
    noise = fl.noise_power(0, 1, np.array([]), np.array([10.0]), 0.26)
    nmodes = np.array([1e4, 1e5])

    cov = fl.gaussian_covariance(cl, noise, layout, nmodes)
    ct = 3e-9 + noise[0]

    assert cov.shape == (2, 1, 1)
    assert cov[0, 0, 0] == pytest.approx(2 * ct**2 / 1e4, rel=1e-14)
    assert cov[1, 0, 0] == pytest.approx(2 * ct**2 / 1e5, rel=1e-14)


def test_noise_only_limit():
    """With zero signal the covariance is pure shot/shape noise."""
    layout = fl.build_layout(0, 1, [], mode="cosmic_shear")
    cl = np.zeros((1, 1, 1))
    noise = fl.noise_power(0, 1, np.array([]), np.array([10.0]), 0.26)
    nmodes = np.array([1e4])

    cov = fl.gaussian_covariance(cl, noise, layout, nmodes)
    assert cov[0, 0, 0] == pytest.approx(2 * noise[0] ** 2 / 1e4, rel=1e-14)


def test_noise_power_conventions():
    """``sigma_e`` is per-component: N = sigma_e^2 / n, not sigma_e^2 / (2n)."""
    noise = fl.noise_power(2, 2, np.array([4.0, 6.0]), np.array([10.0, 5.0]), 0.26)

    assert noise.shape == (4,)
    # Lens legs first, then source legs.
    assert noise[0] == pytest.approx(1.0 / (4.0 * fl.ARCMIN2_PER_SR))
    assert noise[1] == pytest.approx(1.0 / (6.0 * fl.ARCMIN2_PER_SR))
    assert noise[2] == pytest.approx(0.26**2 / (10.0 * fl.ARCMIN2_PER_SR))
    assert noise[3] == pytest.approx(0.26**2 / (5.0 * fl.ARCMIN2_PER_SR))


def test_arcmin2_per_sr_value():
    assert fl.ARCMIN2_PER_SR == pytest.approx(1.1818103e7, rel=1e-6)


def test_covariance_scales_inversely_with_fsky(toy_cl_all, toy_layout, toy_ell):
    noise = fl.noise_power(3, 2, np.full(3, 4.0), np.full(2, 10.0), 0.26)
    edges = np.array([50.0, 200.0, 1000.0, 4000.0])

    cov1 = fl.gaussian_covariance(
        toy_cl_all, noise, toy_layout, fl.n_modes(toy_ell, edges, 0.1))
    cov2 = fl.gaussian_covariance(
        toy_cl_all, noise, toy_layout, fl.n_modes(toy_ell, edges, 0.4))

    assert np.allclose(cov1, 4.0 * cov2, rtol=1e-13)


def test_dense_expansion_is_exactly_ell_block_diagonal(toy_cov):
    """Off-diagonal-in-ell entries must be identically zero, not just small."""
    dense = fl.dense_covariance(toy_cov)
    n_blocks, n_ell = toy_cov.shape[1], toy_cov.shape[0]

    assert dense.shape == (n_blocks * n_ell, n_blocks * n_ell)
    flat_ell = np.tile(np.arange(n_ell), n_blocks)
    off = flat_ell[:, None] != flat_ell[None, :]
    assert np.all(dense[off] == 0.0)


def test_dense_expansion_respects_the_mask(toy_cov, toy_layout, toy_ell):
    mask = fl.build_mask(toy_layout, toy_ell, ell_cuts={"cs": (None, 1000.0)})
    dense = fl.dense_covariance(toy_cov, mask=mask)

    assert dense.shape == (int(mask.sum()), int(mask.sum()))
    assert np.allclose(dense, dense.T)


def test_covariance_needs_leg_pairs_absent_from_the_data_vector(toy_layout):
    """The Wick contractions couple blocks through spectra that are not observables.

    Lens-lens *cross*-bin spectra never appear in the data vector -- clustering
    uses auto-correlations only -- but they enter ``Cov(C^gg_ii, C^gg_jj)``.
    Zeroing one must therefore change the covariance.

    Note the ``atol=0``: these covariances are ~1e-20, so the default
    ``np.allclose`` tolerance would call any two of them equal.
    """
    rng = np.random.default_rng(11)
    n = toy_layout.n_legs
    amp = rng.uniform(0.5, 1.5, n)
    cl = amp[:, None, None] * amp[None, :, None] * np.ones((1, 1, 1)) * 1e-8
    noise = fl.noise_power(3, 2, np.full(3, 4.0), np.full(2, 10.0), 0.26)
    nmodes = np.array([1e4])

    full = fl.gaussian_covariance(cl, noise, toy_layout, nmodes)

    stripped = cl.copy()
    stripped[0, 1, :] = stripped[1, 0, :] = 0.0  # lens bin 0 x lens bin 1
    partial = fl.gaussian_covariance(stripped, noise, toy_layout, nmodes)

    gc = toy_layout.probe_slice("gc")
    assert not np.allclose(full, partial, rtol=1e-10, atol=0.0)
    # Specifically, the Cov(gc_00, gc_11) element is built from ct[0, 1]^2.
    b0, b1 = gc.start, gc.start + 1
    assert full[0, b0, b1] == pytest.approx(2 * cl[0, 1, 0] ** 2 / 1e4, rel=1e-13)
    assert partial[0, b0, b1] == 0.0
