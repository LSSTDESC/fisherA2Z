"""Unit tests for the shift-and-stretch photo-z model.

Pure numpy -- no CCL, no ``fisher.py``, so the whole file runs in well under a
second.  The two cases worth reading are
``test_the_model_is_continuous_across_the_fiducial_point`` (a regression guard
for a derivative bug that got *worse* as the step shrank) and
``test_moments_survive_pixel_noise_that_wrecks_the_gaussian_fit`` (the reason
this model exists at all).
"""

import numpy as np
import pytest

from fisherA2Z import nz_shift_stretch as nzs


Z = np.linspace(0.0, 3.0, 300)


def _gauss(z, mu, sigma):
    return np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _skewed(z, mu=0.9, sigma=0.22, skew=0.6):
    """A smooth, mildly skewed n(z) -- HSC-like, with no separable island."""
    n = _gauss(z, mu, sigma) * (1.0 + skew * np.tanh((z - mu) / sigma))
    n = np.clip(n, 0.0, None)
    return n / np.trapz(n, z)


# --------------------------------------------------------------------------
# nz_moments
# --------------------------------------------------------------------------


def test_moments_of_a_known_gaussian():
    z = np.linspace(-2.0, 6.0, 4001)
    mean, sigma = nzs.nz_moments(z, _gauss(z, 1.3, 0.25))

    assert mean == pytest.approx(1.3, abs=1e-6)
    assert sigma == pytest.approx(0.25, rel=1e-5)


def test_moments_broadcast_over_leading_axes():
    stack = np.array([[_gauss(Z, 0.8, 0.2), _gauss(Z, 1.4, 0.3)]] * 5)
    mean, sigma = nzs.nz_moments(Z, stack)

    assert mean.shape == (5, 2) and sigma.shape == (5, 2)
    assert np.allclose(mean[:, 0], mean[0, 0])
    assert mean[0, 1] > mean[0, 0]


def test_moments_reject_a_mismatched_grid():
    with pytest.raises(ValueError, match="redshift columns"):
        nzs.nz_moments(Z, np.ones(Z.size + 1))


def test_moments_reject_an_empty_nz():
    with pytest.raises(ValueError, match="no moments"):
        nzs.nz_moments(Z, np.zeros_like(Z))


# --------------------------------------------------------------------------
# shift_stretch_nz
# --------------------------------------------------------------------------


def test_the_fiducial_point_returns_the_template_itself():
    """The defining property: no fit stands between the data and the model."""
    n = _skewed(Z)
    out = nzs.shift_stretch_nz(Z, n, 0.0, 1.0)

    assert np.allclose(out, n, rtol=1e-14, atol=0.0)


def test_a_shift_moves_the_mean_by_exactly_that_much():
    n = _skewed(Z)
    mean0 = nzs.nz_moments(Z, n)[0]

    for dz in (-0.15, -0.02, 0.05, 0.2):
        mean = nzs.nz_moments(Z, nzs.shift_stretch_nz(Z, n, dz, 1.0))[0]
        assert mean - mean0 == pytest.approx(dz, abs=2e-3), dz


def test_a_stretch_scales_the_width_and_leaves_the_mean_alone():
    """Stretches stay modest on purpose.

    The template sits at z = 0.9 with sigma = 0.22 on a grid starting at 0, so
    a stretch beyond ~1.4 pushes the left tail off the grid and the recovered
    width comes back low (1.57 instead of 1.6 at s = 1.6).  That is grid
    truncation, not a modelling error, and the forecast never goes near it --
    the stretch prior is a few percent wide.
    """
    n = _skewed(Z)
    mean0, sigma0 = nzs.nz_moments(Z, n)

    for s in (0.75, 0.9, 1.1, 1.3):
        mean, sigma = nzs.nz_moments(Z, nzs.shift_stretch_nz(Z, n, 0.0, s))
        assert sigma / sigma0 == pytest.approx(s, rel=5e-3), s
        assert mean == pytest.approx(mean0, abs=2e-3), s


def test_the_output_is_a_non_negative_normalized_density():
    n = _skewed(Z)
    for dz, s in [(0.0, 1.0), (0.3, 1.4), (-0.25, 0.75), (0.1, 1.0)]:
        out = nzs.shift_stretch_nz(Z, n, dz, s)
        assert np.all(out >= 0.0)
        assert np.trapz(out, Z) == pytest.approx(1.0, rel=1e-10)


def test_renormalize_false_leaves_the_amplitude_alone():
    n = _skewed(Z)
    s = 1.25
    out = nzs.shift_stretch_nz(Z, n, 0.0, s, renormalize=False)

    # Stretching by s spreads the same mass over s times the width, so the
    # explicit 1/s in the model already conserves the integral -- no rescaling
    # needed, up to whatever mass the grid edge swallows.
    assert np.trapz(out, Z) == pytest.approx(1.0, rel=5e-3)
    assert out.max() == pytest.approx(n.max() / s, rel=1e-2)


def test_the_stretch_is_clamped_away_from_zero():
    """A derivative step must never be able to reverse or collapse the template."""
    n = _skewed(Z)
    out = nzs.shift_stretch_nz(Z, n, 0.0, -1.0)

    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.allclose(out, nzs.shift_stretch_nz(Z, n, 0.0, nzs.MIN_STRETCH))


def test_the_model_is_continuous_across_the_fiducial_point():
    """Regression guard for the ``np.interp`` boundary jump.

    With ``left=0``/``right=0`` fill the edge pixel dropped from ``n_0`` to
    zero the instant ``dz`` changed sign.  A central difference across a jump
    of size ``eps`` returns ``eps / 2h``, so the derivative error *grew* as the
    step shrank -- 17% at ``h = 2.5e-4``.  Padding the template with one zero
    cell at each end removes the jump.

    The template here deliberately does *not* vanish at the grid edges; on an
    n(z) that has already decayed to zero the bug is invisible.
    """
    z = np.linspace(0.5, 1.3, 120)
    n = _skewed(z, 0.9, 0.35, skew=0.6)
    assert n[0] > 0.1 * n.max() and n[-1] > 0.1 * n.max(), (n[0], n[-1], n.max())

    # All steps stay well below the grid spacing (0.0067), where the
    # piecewise-linear derivative is exactly step-independent.
    ratios = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        plus = nzs.shift_stretch_nz(z, n, h, 1.0)
        minus = nzs.shift_stretch_nz(z, n, -h, 1.0)
        ratios.append(np.abs((plus - minus) / (2 * h)).max())

    # A jump would make this blow up like 1/h -- a 1000x rise across the scan.
    assert max(ratios) / min(ratios) < 1.01, ratios


def test_the_derivative_matches_minus_the_gradient_of_the_template():
    """``dn/d(dz) = -n'(z)``, which is what the padding is there to preserve."""
    z = np.linspace(0.4, 1.6, 120)
    n = _skewed(z)
    h = 1e-4

    got = (nzs.shift_stretch_nz(z, n, h, 1.0)
           - nzs.shift_stretch_nz(z, n, -h, 1.0)) / (2 * h)
    expected = -np.gradient(n, z)

    interior = slice(2, -2)
    assert np.allclose(got[interior], expected[interior],
                       rtol=0.0, atol=0.02 * np.abs(expected).max())


def test_a_short_grid_is_rejected():
    # z_pivot is given explicitly so the failure comes from the template
    # padding rather than from taking moments of a one-point grid.
    with pytest.raises(ValueError, match="at least two points"):
        nzs.shift_stretch_nz(np.array([1.0]), np.array([1.0]), 0.1,
                             z_pivot=1.0)


# --------------------------------------------------------------------------
# build_shift_stretch_model
# --------------------------------------------------------------------------


def test_building_the_model_normalizes_and_records_the_moments():
    raw = np.array([_skewed(Z, 0.7, 0.18) * 37.0, _skewed(Z, 1.2, 0.26) * 0.004])
    m = nzs.build_shift_stretch_model(Z, raw)

    assert m.n_tomo == 2
    assert np.allclose(np.trapz(m.nz_fid, Z, axis=1), 1.0, rtol=1e-12)
    assert m.z_pivot[1] > m.z_pivot[0]
    assert np.all(m.sigma_fid > 0)
    # The pivot is the mean, which is what decouples the shift from the stretch.
    assert np.allclose(m.z_pivot, nzs.nz_moments(Z, m.nz_fid)[0])


def test_the_model_method_round_trips_through_shift_stretch_nz():
    m = nzs.build_shift_stretch_model(Z, np.array([_skewed(Z)]))

    assert np.allclose(m.model(0), m.nz_fid[0], rtol=1e-14)
    assert np.allclose(
        m.model(0, 0.1, 1.2),
        nzs.shift_stretch_nz(Z, m.nz_fid[0], 0.1, 1.2, m.z_pivot[0]))


def test_negative_bins_are_clipped_with_a_warning():
    bad = np.array([_skewed(Z)])
    bad[0, 5] = -1e-3

    with pytest.warns(UserWarning, match="negative"):
        m = nzs.build_shift_stretch_model(Z, bad)
    assert np.all(m.nz_fid >= 0)


def test_an_empty_bin_is_rejected():
    with pytest.raises(ValueError, match="positive, finite integral"):
        nzs.build_shift_stretch_model(Z, np.zeros((1, Z.size)))


def test_a_mismatched_grid_is_rejected():
    with pytest.raises(ValueError, match="redshift columns"):
        nzs.build_shift_stretch_model(Z, np.ones((2, Z.size + 3)))


# --------------------------------------------------------------------------
# fit_shift_stretch -- the estimator
# --------------------------------------------------------------------------


def test_injected_shifts_and_stretches_are_recovered():
    """The estimator is exact for a location-scale family, so this is tight."""
    ref = np.array([_skewed(Z, 0.8, 0.20), _skewed(Z, 1.3, 0.28)])
    truth = [(0.06, 1.10), (-0.09, 0.88)]

    reals = np.array([[nzs.shift_stretch_nz(Z, ref[b], dz, s)
                       for b, (dz, s) in enumerate(truth)]])
    got = nzs.fit_shift_stretch(Z, ref, reals)

    assert got.shape == (1, 2, 2)
    for b, (dz, s) in enumerate(truth):
        assert got[0, b, 0] == pytest.approx(dz, abs=1e-3)
        assert got[0, b, 1] == pytest.approx(s, rel=2e-3)


def test_the_reference_fits_itself_at_the_fiducial_values():
    ref = np.array([_skewed(Z, 0.8, 0.20), _skewed(Z, 1.3, 0.28)])
    got = nzs.fit_shift_stretch(Z, ref, ref[None, :, :])

    assert np.allclose(got[0, :, 0], 0.0, atol=1e-14)
    assert np.allclose(got[0, :, 1], 1.0, rtol=1e-14)


def test_the_prior_matches_the_injected_scatter():
    from fisherA2Z.nz_decomposition import photoz_prior_from_realizations

    rng = np.random.default_rng(11)
    ref = np.array([_skewed(Z, 0.9, 0.22)])
    sd_dz, sd_s = 0.012, 0.035

    n_real = 400
    reals = np.empty((n_real, 1, Z.size))
    for r in range(n_real):
        reals[r, 0] = nzs.shift_stretch_nz(
            Z, ref[0], rng.normal(0, sd_dz), rng.normal(1.0, sd_s))

    cov = photoz_prior_from_realizations(
        nzs.fit_shift_stretch(Z, ref, reals))

    assert cov.shape == (1, 2, 2)
    assert np.sqrt(cov[0, 0, 0]) == pytest.approx(sd_dz, rel=0.2)
    assert np.sqrt(cov[0, 1, 1]) == pytest.approx(sd_s, rel=0.2)
    # Shift and stretch are independent here; the pivot is what keeps them so.
    corr = cov[0, 0, 1] / np.sqrt(cov[0, 0, 0] * cov[0, 1, 1])
    assert abs(corr) < 0.2


def test_moments_survive_pixel_noise_that_wrecks_the_gaussian_fit():
    """Why this model replaced the decomposition as the default.

    ``f_out`` comes from ``min_z n/G``, a downward-biased order statistic: one
    low-fluctuating pixel drags it, so its scatter across realizations tracks
    the pixel noise rather than the photo-z calibration.  The moments are
    integrals, so the same noise averages down.
    """
    from fisherA2Z import nz_decomposition as nzd

    rng = np.random.default_rng(3)
    ref = np.array([_skewed(Z, 0.9, 0.22)])

    n_real = 24
    clean = np.repeat(ref[None, :, :], n_real, axis=0)
    noisy = np.clip(clean + rng.normal(0, 0.02 * ref.max(), clean.shape),
                    1e-12, None)
    noisy /= np.trapz(noisy, Z, axis=-1)[..., None]

    ss = nzs.fit_shift_stretch(Z, ref, noisy)
    dz_scatter = ss[:, 0, 0].std(ddof=1)
    # Pure pixel noise, no real photo-z error: the shift should barely move.
    assert dz_scatter < 0.1 * 0.22, dz_scatter

    fout = np.array([nzd.decompose_nz(Z, noisy[r, 0]).f_out
                     for r in range(n_real)])
    fout_ref = nzd.decompose_nz(Z, ref[0]).f_out
    # The same noise moves f_out by a sizeable fraction of its own value.
    assert fout.std(ddof=1) > 0.02 or abs(fout.mean() - fout_ref) > 0.05, (
        f"f_out scatter {fout.std(ddof=1):.4g}, bias "
        f"{fout.mean() - fout_ref:+.4g}"
    )


def test_a_skewed_nz_gets_an_outlier_component_that_is_not_an_outlier():
    """The user's finding, pinned down: no tail means no meaningful split.

    The decomposition *reconstructs* any n(z) essentially exactly -- the
    residual is defined as ``n - A G``, so core plus residual is the data by
    construction, to 1e-13.  Reconstruction error is therefore no test of
    anything.  What breaks on a smooth skewed n(z) is the *interpretation*:
    there is no separable outlier population, so the fit hands back a large
    ``f_out`` whose "outlier" component sits right on top of the core.  It is
    the non-Gaussian shoulders, relabelled.

    A genuinely bimodal n(z) is included as the control, so this test fails if
    the decomposition ever stops finding real islands.
    """
    from fisherA2Z import nz_decomposition as nzd

    smooth = _skewed(Z, 0.9, 0.22, skew=0.8)
    dec = nzd.decompose_nz(Z, smooth)

    # Exact reconstruction, as advertised -- and hence uninformative.
    assert np.abs(dec.model() - smooth).max() / smooth.max() < 1e-4

    # A pure Gaussian gets f_out ~ 0; mere skewness already buys a few percent
    # of spurious "outliers", and real n(z) with more structure buys far more.
    pure = nzd.decompose_nz(Z, _gauss(Z, 0.9, 0.22))
    assert dec.f_out > 20 * max(pure.f_out, 1e-4), (dec.f_out, pure.f_out)

    # And they are not outliers: the component sits on top of the core.
    out_mean = nzs.nz_moments(Z, dec.nz_out)[0]
    assert abs(out_mean - dec.mu) < 2.0 * dec.sigma, (
        f"outlier component at {out_mean:.3f} vs core {dec.mu:.3f} "
        f"+- {dec.sigma:.3f}"
    )

    # Control: a real outlier island is found where it actually is.
    island = 0.85 * _gauss(Z, 0.9, 0.20) + 0.15 * _gauss(Z, 2.2, 0.25)
    island /= np.trapz(island, Z)
    dec_i = nzd.decompose_nz(Z, island)
    assert abs(nzs.nz_moments(Z, dec_i.nz_out)[0] - dec_i.mu) > 3.0 * dec_i.sigma

    # The shift-and-stretch template needs no such split in either case.
    assert np.allclose(nzs.shift_stretch_nz(Z, smooth, 0.0, 1.0), smooth,
                       rtol=1e-14)


def test_realizations_of_the_wrong_shape_are_rejected():
    ref = np.array([_skewed(Z)])
    with pytest.raises(ValueError, match="realizations must have shape"):
        nzs.fit_shift_stretch(Z, ref, np.ones((4, 3, Z.size)))
    with pytest.raises(ValueError, match="realizations must have shape"):
        nzs.fit_shift_stretch(Z, ref, np.ones((3, Z.size)))
