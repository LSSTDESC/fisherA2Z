"""Tests for the n(z) -> (mu, sigma, f_out) decomposition.

Pure numpy/scipy: neither CCL nor :mod:`fisherA2Z.fisher` is imported here.
"""

import numpy as np
import pytest

from fisherA2Z import nz_decomposition as nzd


def test_recovers_a_pure_gaussian(z_grid):
    """A bin with no outliers must come back with f_out clamped to zero."""
    nz = nzd.gaussian(z_grid, 1.0, 0.2)
    fit = nzd.decompose_nz(z_grid, nz)

    assert fit.mu == pytest.approx(1.0, abs=2e-3)
    assert fit.sigma == pytest.approx(0.2, abs=2e-3)
    assert fit.f_out < nzd.F_OUT_MIN
    assert fit.degenerate


def test_recovers_a_known_mixture(z_grid):
    truth = dict(mu=1.0, sigma=0.15, f_out=0.15)
    nz = (
        (1 - truth["f_out"]) * nzd.gaussian(z_grid, truth["mu"], truth["sigma"])
        + truth["f_out"] * nzd.gaussian(z_grid, 2.2, 0.30)
    )
    fit = nzd.decompose_nz(z_grid, nz / np.trapz(nz, z_grid))

    assert fit.mu == pytest.approx(truth["mu"], abs=2e-3)
    assert fit.sigma == pytest.approx(truth["sigma"], abs=2e-3)
    assert fit.f_out == pytest.approx(truth["f_out"], abs=5e-3)


def test_hard_zeros_do_not_collapse_the_amplitude(z_grid):
    """Regression guard for the ``A -> 0, f_out -> 1`` trap.

    Real tomographic n(z) is exactly zero outside its support, and a naive
    ``min_z n/G`` would then return zero for *any* (mu, sigma).
    """
    nz = 0.85 * nzd.gaussian(z_grid, 1.0, 0.15) + 0.15 * nzd.gaussian(z_grid, 2.2, 0.30)
    nz = np.where((z_grid > 0.35) & (z_grid < 2.9), nz, 0.0)

    fit = nzd.decompose_nz(z_grid, nz / np.trapz(nz, z_grid))

    assert fit.amplitude > 0.1
    assert fit.f_out < 0.5
    assert fit.mu == pytest.approx(1.0, abs=0.02)


def test_residual_is_non_negative_and_normalized(z_grid, toy_nz):
    """The outlier component is a probability density by construction."""
    dec = nzd.decompose_tomography(z_grid, toy_nz)

    assert np.all(dec.nz_out >= 0.0)
    assert np.allclose(np.trapz(dec.nz_out, z_grid, axis=1), 1.0, atol=1e-10)


def test_reconstruction_closes(z_grid, toy_nz):
    """(1-f) core + f residual must reproduce the input n(z) in L1."""
    dec = nzd.decompose_tomography(z_grid, toy_nz)

    for b in range(dec.n_tomo):
        model = nzd.build_model_nz(
            z_grid, dec.mu[b], dec.sigma[b], dec.f_out[b], dec.nz_out[b]
        )
        l1 = np.trapz(np.abs(model - dec.nz_fid[b]), z_grid)
        assert l1 < 1e-3, f"bin {b} reconstruction L1 error {l1}"


def test_model_nz_is_normalized(z_grid, toy_nz):
    """Including under shifted parameters -- the truncation deficit moves with dz."""
    dec = nzd.decompose_tomography(z_grid, toy_nz)

    for dz in (-0.05, 0.0, 0.05):
        for b in range(dec.n_tomo):
            model = dec.bins[b].model(dz=dz)
            assert np.trapz(model, z_grid) == pytest.approx(1.0, rel=1e-12)


def test_degenerate_bin_is_flagged_without_nans(z_grid):
    nz = np.array([nzd.gaussian(z_grid, 1.0, 0.2),
                   0.9 * nzd.gaussian(z_grid, 1.2, 0.15)
                   + 0.1 * nzd.gaussian(z_grid, 2.5, 0.2)])
    nz /= np.trapz(nz, z_grid, axis=1)[:, None]
    dec = nzd.decompose_tomography(z_grid, nz)

    assert dec.degenerate[0] and not dec.degenerate[1]
    assert np.all(np.isfinite(dec.params))
    assert np.all(np.isfinite(dec.nz_out))
    # A degenerate bin still needs a usable outlier shape for the f derivative.
    assert np.trapz(dec.nz_out[0], z_grid) == pytest.approx(1.0, rel=1e-10)


def test_warm_start_matches_cold_start(z_grid, toy_nz):
    cold = nzd.decompose_nz(z_grid, toy_nz[0])
    warm = nzd.decompose_nz(z_grid, toy_nz[0], x0=(cold.mu, cold.sigma))

    assert warm.params == pytest.approx(cold.params, abs=1e-4)


def test_priors_recover_injected_jitter(z_grid, toy_nz, toy_realizations):
    """The realizations carry sigma(mu)=0.01 and sigma(f_out)=0.01 by design."""
    dec = nzd.decompose_tomography(z_grid, toy_nz)
    warm = np.column_stack([dec.mu, dec.sigma])
    params = nzd.decompose_realizations(z_grid, toy_realizations, warm_start=warm)
    cov = nzd.photoz_prior_from_realizations(params)

    assert cov.shape == (2, 3, 3)
    for b in range(2):
        assert np.allclose(cov[b], cov[b].T)
        assert np.all(np.linalg.eigvalsh(cov[b]) > 0)
        assert np.sqrt(cov[b][0, 0]) == pytest.approx(0.01, rel=0.2)
        assert np.sqrt(cov[b][2, 2]) == pytest.approx(0.01, rel=0.5)


def test_prior_diag_mode_zeroes_off_diagonals(z_grid, toy_nz, toy_realizations):
    dec = nzd.decompose_tomography(z_grid, toy_nz)
    warm = np.column_stack([dec.mu, dec.sigma])
    params = nzd.decompose_realizations(z_grid, toy_realizations, warm_start=warm)

    full = nzd.photoz_prior_from_realizations(params, mode="full")
    diag = nzd.photoz_prior_from_realizations(params, mode="diag")

    assert np.allclose(diag, np.stack([np.diag(np.diag(c)) for c in full]))


def test_realizations_axis_disambiguation():
    """A square (n, n, n_z) input must raise rather than silently guess."""
    rng = np.random.default_rng(0)
    square = rng.random((4, 4, 20))

    with pytest.raises(ValueError, match="ambiguous"):
        nzd.orient_realizations(square, n_tomo=4)

    assert nzd.orient_realizations(square, 4, realizations_axis=0).shape == (4, 4, 20)
    assert np.array_equal(
        nzd.orient_realizations(square, 4, realizations_axis=1),
        square.transpose(1, 0, 2),
    )


def test_realizations_axis_inferred_both_ways():
    rng = np.random.default_rng(1)
    real_first = rng.random((7, 3, 20))
    tomo_first = real_first.transpose(1, 0, 2)

    assert nzd.orient_realizations(real_first, 3).shape == (7, 3, 20)
    assert nzd.orient_realizations(tomo_first, 3).shape == (7, 3, 20)
    with pytest.raises(ValueError, match="n_tomo"):
        nzd.orient_realizations(real_first, 5)


def test_profile_amplitude_is_the_largest_feasible_one(z_grid):
    """A*G <= n on the support, and nudging A up violates it."""
    nz = 0.8 * nzd.gaussian(z_grid, 1.0, 0.2) + 0.2 * nzd.gaussian(z_grid, 2.0, 0.3)
    A, G = nzd.profile_amplitude(1.0, 0.2, z_grid, nz, eta_rel=0.0)

    support = G > nzd.TAU * G.max()
    assert np.all(A * G[support] <= nz[support] * (1 + 1e-9))
    assert np.any(1.001 * A * G[support] > nz[support] * (1 + 1e-9))


def test_smoothing_preserves_normalization_and_positivity(z_grid, toy_realizations):
    smoothed = nzd.smooth_nz(z_grid, toy_realizations, 0.05)

    assert smoothed.shape == toy_realizations.shape
    assert np.all(smoothed >= 0.0)
    assert np.allclose(np.trapz(smoothed, z_grid, axis=-1),
                       np.trapz(toy_realizations, z_grid, axis=-1), rtol=1e-2)


def test_noise_ratio_grows_with_injected_noise(z_grid, toy_nz, toy_realizations):
    """The diagnostic is only meaningful as a monotone function of pixel noise.

    Its absolute scale depends on the grid spacing, so no threshold is asserted.
    """
    rng = np.random.default_rng(3)
    ratios = [nzd.noise_ratio(z_grid, toy_nz, toy_realizations)]
    for level in (0.05, 0.2, 0.5):
        noisy = np.clip(
            toy_realizations * (1 + rng.normal(0, level, toy_realizations.shape)),
            0, None,
        )
        ratios.append(nzd.noise_ratio(z_grid, toy_nz, noisy))

    assert np.all(np.diff(ratios) > 0), ratios
    assert ratios[0] == pytest.approx(1.0, abs=0.5)
