"""Tests for the Fisher-bias machinery: Eq. (13) of Zhang et al. (2025).

The linear-response test is the one that matters.  If the residual is built to
be *exactly* ``eps * dC/dp`` for a single parameter, then

    F^-1 . D^T Cov^-1 . (eps * D e_p) = eps * e_p

algebraically, with no priors and nothing dropped.  Recovering ``eps`` therefore
validates the mask consistency, the parameter ordering, the per-ell Cholesky
solve and the sign convention in one shot.
"""

import numpy as np
import pytest

from fisherA2Z import flex_layout as fl
from fisherA2Z.fisher_flex import FisherFlex


# --------------------------------------------------------------------------
# pure linear algebra -- no CCL
# --------------------------------------------------------------------------


def test_bias_vector_matches_the_dense_reference(toy_layout, toy_ell, toy_cov):
    """The per-ell solve must equal a dense ``D^T C^-1 delta``."""
    rng = np.random.default_rng(3)
    n_blocks, n_ell = toy_layout.n_blocks, toy_ell.size
    deriv = rng.normal(size=(4, n_blocks, n_ell)) * 1e-9
    delta = rng.normal(size=(n_blocks, n_ell)) * 1e-10

    got, n_data = fl.bias_vector_from_derivs(deriv, toy_cov, delta)

    dense = fl.dense_covariance(toy_cov)
    inv = np.linalg.inv(dense)
    d_flat = deriv.reshape(4, -1)
    want = d_flat @ inv @ delta.reshape(-1)

    assert n_data == n_blocks * n_ell
    assert np.allclose(got, want, rtol=1e-9, atol=0)


def test_bias_vector_respects_the_mask(toy_layout, toy_ell, toy_cov):
    """A masked element must not contribute, whatever the residual holds."""
    rng = np.random.default_rng(11)
    n_blocks, n_ell = toy_layout.n_blocks, toy_ell.size
    deriv = rng.normal(size=(3, n_blocks, n_ell)) * 1e-9
    delta = rng.normal(size=(n_blocks, n_ell)) * 1e-10
    mask = np.ones((n_blocks, n_ell), dtype=bool)
    mask[:, -1] = False

    got, n_data = fl.bias_vector_from_derivs(deriv, toy_cov, delta, mask)

    poisoned = delta.copy()
    poisoned[:, -1] = 1e30
    got2, _ = fl.bias_vector_from_derivs(deriv, toy_cov, poisoned, mask)

    assert n_data == n_blocks * (n_ell - 1)
    assert np.array_equal(got, got2)


def test_bias_vector_rejects_a_mis_shaped_residual(toy_layout, toy_ell, toy_cov):
    deriv = np.zeros((2, toy_layout.n_blocks, toy_ell.size))
    with pytest.raises(ValueError, match="delta has shape"):
        fl.bias_vector_from_derivs(deriv, toy_cov, np.zeros((3, 3)))


def test_fisher_bias_identity_is_exact(toy_layout, toy_ell, toy_cov):
    """``delta = eps * D_p`` must give back exactly ``delta_p = eps e_p``.

    This is the algebraic core of Eq. (13), free of any CCL or grid effect:
    ``B = D^T Cov^-1 (eps D e_p) = eps F e_p``, so ``F^-1 B = eps e_p``.  The
    end-to-end CCL test below is the same statement with a real forward model,
    where the recovery is only as good as the numerical derivative.
    """
    rng = np.random.default_rng(19)
    n_blocks, n_ell = toy_layout.n_blocks, toy_ell.size
    deriv = rng.normal(size=(4, n_blocks, n_ell)) * 1e-9
    mask = np.ones((n_blocks, n_ell), dtype=bool)
    mask[2, 1] = False  # exercise the masked path too

    eps = 3e-3
    p = 2
    delta = eps * deriv[p]

    fisher, _ = fl.fisher_from_derivs(deriv, toy_cov, mask)
    bias_vec, _ = fl.bias_vector_from_derivs(deriv, toy_cov, delta, mask)
    delta_p = np.linalg.solve(fisher, bias_vec)

    want = np.zeros(4)
    want[p] = eps
    assert np.allclose(delta_p, want, rtol=1e-8, atol=1e-12)


# --------------------------------------------------------------------------
# end to end -- small CCL evaluations
# --------------------------------------------------------------------------

#: 150 points, i.e. dz = 0.016.  The photo-z shift is applied by interpolating
#: n(z) on this grid, so a coarse grid makes the numerical dC/dzbias disagree
#: with the true local response: measured |C(eps) - C(0)| / |eps dC/dz| is 0.969
#: at dz = 0.05, 0.991 at dz = 0.016 and 0.997 at dz = 0.008, essentially
#: independent of eps.  That is a property of the forward model, not of the bias
#: formula (see ``test_fisher_bias_identity_is_exact``), but it sets the
#: tolerance the end-to-end test can honestly claim.
Z_GRID = np.linspace(0.05, 2.5, 150)
BIN_MU = (0.6, 1.1)
BIN_SIGMA = (0.15, 0.18)
BIN_OUT_MU = (1.8, 2.1)


def _toy_inputs():
    """12 realizations and their mean, mirroring ``test_ccl_smoke``."""
    import fisherA2Z.nz_decomposition as nzd

    z = Z_GRID
    rng = np.random.default_rng(5)
    reals = np.empty((12, 2, z.size))
    for r in range(12):
        for b in range(2):
            reals[r, b] = nzd.build_model_nz(
                z, BIN_MU[b] + rng.normal(0, 0.01), BIN_SIGMA[b],
                0.1 + rng.normal(0, 0.01),
                nzd.gaussian(z, BIN_OUT_MU[b], 0.25),
            )
    return z, reals.mean(axis=0), reals


#: Parameters kept for the prior-free linear-response tests.
#:
#: The full 17-parameter Fisher matrix of this toy setup is rank-deficient
#: without priors -- 34 data points cannot constrain 17 parameters, and
#: ``cond(F) ~ 2e10``.  Inverting it would amplify the O(eps^2) nonlinearity
#: into noise.  Restricting to these three gives ``cond(F) ~ 17``, so the
#: algebraic identity is actually testable.  Dropping parameters does not weaken
#: the identity: with ``delta = eps * D_zbias1`` the bias vector restricted to
#: any subset containing ``zbias1`` is still ``eps * F_sub[:, zbias1]``.
LINEAR_KEEP = ["omega_m", "sigma_8", "zbias1"]


@pytest.fixture(scope="module")
def small_flex():
    z, nz, reals = _toy_inputs()
    flex = FisherFlex(
        nz_source=nz, nz_realizations=reals, z_grid=z,
        neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26,
        n_lens_bins=2, ell_edges=np.geomspace(100.0, 3000.0, 9),
        verbose=False,
    )
    return flex.compute(parallel=False)


def _linear_params(flex, **extra):
    drop = [p for p in flex.param_order if p not in LINEAR_KEEP]
    return dict(prior=False, drop_params=drop, **extra)


@pytest.mark.ccl
def test_identical_nz_gives_zero_bias(small_flex):
    """The only residual allowed is the roundoff of renormalizing by ~1.0."""
    bias = small_flex.forecast_bias(nz_truth=small_flex.nz_source_fid)

    assert np.max(np.abs(bias.delta_cl) / small_flex.data_vector) < 1e-12
    res = small_flex.forecast()
    for p in bias.param_order:
        assert abs(bias.shift(p)) < 1e-10 * res.sigma(p)
    assert bias.mafe() < 1e-12


@pytest.mark.ccl
def test_linear_response_recovers_the_injected_shift(small_flex):
    """Truth = the model at ``zbias1 = eps`` must yield ``delta_p = eps e_zbias1``.

    The 2% tolerance is set by the forward model, not the bias formula: on this
    grid ``|C(eps) - C(0)|`` is 0.991 of ``|eps dC/dzbias1|`` because the shift
    is applied by interpolation (see the ``Z_GRID`` note).  The formula itself
    is checked exactly in ``test_fisher_bias_identity_is_exact``.
    """
    eps = 2e-3
    model = small_flex._model()
    vals = dict(small_flex.fid)
    vals["zbias1"] = eps
    nz_truth = model.source_nz(vals)

    bias = small_flex.forecast_bias(
        nz_truth=nz_truth, forecast_params=_linear_params(small_flex),
        model=model)

    assert bias.shift("zbias1") == pytest.approx(eps, rel=0.02)
    for p in bias.param_order:
        if p == "zbias1":
            continue
        assert abs(bias.shift(p)) < 1e-3 * bias.result.sigma(p)


@pytest.mark.ccl
def test_bias_sign_follows_the_injected_sign(small_flex):
    model = small_flex._model()
    out = {}
    for eps in (+2e-3, -2e-3):
        vals = dict(small_flex.fid)
        vals["zbias1"] = eps
        out[eps] = small_flex.forecast_bias(
            nz_truth=model.source_nz(vals),
            forecast_params=_linear_params(small_flex),
            model=model).shift("zbias1")
    assert out[+2e-3] > 0 > out[-2e-3]
    assert out[+2e-3] == pytest.approx(-out[-2e-3], rel=0.1)


@pytest.mark.ccl
def test_nz_truth_is_renormalized_and_grid_checked(small_flex):
    """A rescaled n(z) is the same physical distribution, so the bias is too."""
    base = small_flex.forecast_bias(nz_truth=small_flex.nz_source_fid * 3.0)
    res = small_flex.forecast()
    for p in base.param_order:
        assert abs(base.shift(p)) < 1e-10 * res.sigma(p)

    with pytest.raises(ValueError, match="tomographic bins"):
        small_flex.forecast_bias(nz_truth=np.ones((3, Z_GRID.size)))
    with pytest.raises(ValueError, match="redshift samples"):
        small_flex.forecast_bias(nz_truth=np.ones((2, 7)))


@pytest.mark.ccl
def test_z_truth_interpolation(small_flex):
    """Handing the same n(z) on a finer grid must reproduce it closely."""
    fine = np.linspace(Z_GRID[0], Z_GRID[-1], 4 * Z_GRID.size)
    nz_fine = np.array([np.interp(fine, Z_GRID, row)
                        for row in small_flex.nz_source_fid])
    bias = small_flex.forecast_bias(nz_truth=nz_fine, z_truth=fine)
    res = small_flex.forecast()
    for p in ("omega_m", "sigma_8"):
        assert abs(bias.shift(p)) < 0.02 * res.sigma(p)


@pytest.mark.ccl
def test_forecast_params_are_plumbed_through(small_flex):
    model = small_flex._model()
    vals = dict(small_flex.fid)
    vals["zbias1"] = 5e-3
    nz_truth = model.source_nz(vals)

    full = small_flex.forecast_bias(nz_truth=nz_truth, model=model)
    dropped = small_flex.forecast_bias(
        nz_truth=nz_truth, forecast_params=dict(drop_params=["zbias1"]),
        model=model)
    cs_only = small_flex.forecast_bias(
        nz_truth=nz_truth, forecast_params=dict(probes=["cs"]), model=model)

    assert "zbias1" in full.param_order
    assert "zbias1" not in dropped.param_order
    assert dropped.param_order == dropped.result.param_order
    # Fixing the nuisance stops it absorbing the systematic, so more of the
    # residual has to be taken up by the cosmology.
    assert abs(dropped.shift("omega_m")) > abs(full.shift("omega_m"))
    assert cs_only.n_data_used < full.n_data_used


@pytest.mark.ccl
def test_result_object_accessors_are_consistent(small_flex):
    model = small_flex._model()
    vals = dict(small_flex.fid)
    vals["zbias1"] = 5e-3
    bias = small_flex.forecast_bias(nz_truth=model.source_nz(vals), model=model)

    for p in ("omega_m", "sigma_8"):
        assert bias.n_sigma(p) == pytest.approx(
            bias.shift(p) / bias.result.sigma(p))
        assert bias.shifted_fid(p) == pytest.approx(
            bias.result.param_fid[p] + bias.shift(p))
    assert bias.shift(0) == bias.shift(bias.param_order[0])

    s8_fid, s8_shift, d, n = bias.s8()
    assert d == pytest.approx(s8_shift - s8_fid)
    assert n == pytest.approx(d / bias.result.s8()[1])
    assert 0.0 < bias.mafe() < 1.0
    assert "bias/sigma" in bias.summary(["omega_m", "sigma_8"])


@pytest.mark.ccl
def test_bias_2d_is_consistent_with_the_1d_numbers(small_flex):
    model = small_flex._model()
    vals = dict(small_flex.fid)
    vals["zbias1"] = 5e-3
    bias = small_flex.forecast_bias(nz_truth=model.source_nz(vals), model=model)

    b2 = bias.bias_2d("omega_m", "sigma_8")

    assert b2.params == ("omega_m", "sigma_8")
    assert b2.delta == pytest.approx(
        [bias.shift("omega_m"), bias.shift("sigma_8")])
    assert b2.shifted == pytest.approx(b2.fid + b2.delta)
    # The marginal significances must reproduce the 1-D accessors exactly.
    assert b2.marginal_n_sigma() == pytest.approx(
        (bias.n_sigma("omega_m"), bias.n_sigma("sigma_8")))
    # cov must be the marginalized 2x2 block, i.e. the inverse of marginalize().
    assert np.allclose(np.linalg.inv(b2.cov),
                       bias.result.marginalize("omega_m", "sigma_8"))


@pytest.mark.ccl
def test_bias_2d_chi2_matches_the_quadratic_form(small_flex):
    model = small_flex._model()
    vals = dict(small_flex.fid)
    vals["zbias1"] = 5e-3
    bias = small_flex.forecast_bias(nz_truth=model.source_nz(vals), model=model)
    b2 = bias.bias_2d("w_0", "w_a")

    want = b2.delta @ np.linalg.inv(b2.cov) @ b2.delta
    assert b2.chi2 == pytest.approx(want, rel=1e-10)
    assert b2.n_sigma == pytest.approx(np.sqrt(b2.chi2))
    assert b2.confidence == pytest.approx(1.0 - np.exp(-0.5 * b2.chi2))
    assert 0.0 <= b2.confidence <= 1.0


def test_bias_2d_containment_uses_the_contour_convention():
    """``inside(s)`` must agree with the ellipse ``contour()`` would draw."""
    from fisherA2Z.fisher_flex import Bias2D, CHI2_2D

    cov = np.eye(2)
    for s in (1, 2, 3):
        just_in = Bias2D(("a", "b"), ("a", "b"), np.zeros(2), np.zeros(2),
                         np.zeros(2), cov, CHI2_2D[s] - 1e-9)
        just_out = Bias2D(("a", "b"), ("a", "b"), np.zeros(2), np.zeros(2),
                          np.zeros(2), cov, CHI2_2D[s] + 1e-9)
        assert just_in.inside(s) and not just_out.inside(s)
    with pytest.raises(KeyError):
        just_in.inside(4)


def test_bias_2d_geometry_on_a_known_case():
    """A unit-covariance plane makes every quantity checkable by hand."""
    from fisherA2Z.fisher_flex import Bias2D

    b2 = Bias2D(("a", "b"), ("a", "b"), np.zeros(2), np.array([3.0, 4.0]),
                np.array([3.0, 4.0]), np.eye(2), 25.0)
    assert b2.n_sigma == pytest.approx(5.0)
    assert b2.angle == pytest.approx(np.degrees(np.arctan2(4.0, 3.0)))
    assert b2.marginal_n_sigma() == pytest.approx((3.0, 4.0))
    assert not b2.inside(3)
    assert "joint" in str(b2)


@pytest.mark.ccl
def test_bias_2d_detects_a_shift_across_the_degeneracy(small_flex):
    """The joint significance can exceed both marginal ones.

    That is the whole reason the 2-D number is worth having: a shift running
    across a degeneracy is small in each marginal error bar yet far outside the
    ellipse.  Constructed here directly from the covariance, so it is a property
    of ``Bias2D`` rather than of any particular systematic.
    """
    from fisherA2Z.fisher_flex import Bias2D

    res = small_flex.forecast()
    i, j = res.index("omega_m"), res.index("sigma_8")
    cov = res.cov[np.ix_([i, j], [i, j])]
    # Step along the *tightest* principal direction: small in both marginals.
    evals, evecs = np.linalg.eigh(cov)
    delta = 3.0 * np.sqrt(evals[0]) * evecs[:, 0]

    b2 = Bias2D(("omega_m", "sigma_8"), ("a", "b"), np.zeros(2), delta, delta,
                cov, float(delta @ np.linalg.inv(cov) @ delta))
    assert b2.n_sigma == pytest.approx(3.0)
    assert max(abs(np.array(b2.marginal_n_sigma()))) < b2.n_sigma
    assert not b2.inside(2)


@pytest.mark.ccl
def test_plotting_smoke(small_flex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = small_flex._model()
    vals = dict(small_flex.fid)
    vals["zbias1"] = 5e-3
    bias = small_flex.forecast_bias(nz_truth=model.source_nz(vals), model=model)
    params = ["omega_m", "sigma_8", "w_0"]

    _, ax = plt.subplots()
    bias.result.contour("omega_m", "sigma_8", ax=ax)
    n_before = len(ax.patches)
    bias.arrow("omega_m", "sigma_8", ax=ax, shifted_contour=True)
    assert len(ax.patches) > n_before

    fig = bias.result.corner(params)
    assert bias.corner_arrows(params, fig=fig) is fig
    plt.close("all")


@pytest.mark.ccl
def test_contour_centre_and_autolim_are_backwards_compatible(small_flex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = small_flex.forecast()
    _, ax = plt.subplots()
    res.contour("omega_m", "sigma_8", ax=ax)
    default_lims = ax.get_xlim(), ax.get_ylim()

    # autolim=False must leave whatever limits are already there.
    ax.set_xlim(0.0, 1.0)
    res.contour("omega_m", "sigma_8", ax=ax, autolim=False)
    assert ax.get_xlim() == (0.0, 1.0)

    # centre= moves the ellipse and, with autolim, the window with it.
    _, ax2 = plt.subplots()
    res.contour("omega_m", "sigma_8", ax=ax2, centre=(0.9, 0.9))
    assert ax2.get_xlim() != default_lims[0]
    plt.close("all")
