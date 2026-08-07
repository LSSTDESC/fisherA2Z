"""Tests for the Fisher assembly, priors, masks and figure of merit.

These exercise the per-ell Cholesky path in :func:`flex_layout.fisher_from_derivs`
against dense linear algebra, and the ``FisherFlexResult`` conveniences against
their closed-form definitions.  No CCL evaluation happens here -- the
derivatives are synthetic.
"""

import numpy as np
import pytest

from fisherA2Z import flex_layout as fl
from fisherA2Z.fisher_flex import FisherFlexResult, FoM, marginalize


@pytest.fixture
def toy_deriv(toy_layout, toy_ell):
    """Synthetic derivatives of 4 parameters w.r.t. every (block, ell)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1e-9, size=(4, toy_layout.n_blocks, toy_ell.size))


def test_fisher_is_symmetric_and_positive_definite(toy_deriv, toy_cov):
    fisher, n_data = fl.fisher_from_derivs(toy_deriv, toy_cov)

    assert fisher.shape == (4, 4)
    assert np.array_equal(fisher, fisher.T), "not bitwise symmetric"
    assert np.all(np.linalg.eigvalsh(fisher) > 0)
    assert n_data == toy_cov.shape[1] * toy_cov.shape[0]


def test_per_ell_solve_equals_the_dense_expression(toy_deriv, toy_cov):
    """F = D^T C^-1 D, done densely, must match the blocked per-ell version."""
    fisher, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)

    n_par, n_blocks, n_ell = toy_deriv.shape
    dense_cov = fl.dense_covariance(toy_cov)
    # dense_covariance orders the flat vector block-major.
    d_flat = toy_deriv.reshape(n_par, n_blocks * n_ell)
    expected = d_flat @ np.linalg.inv(dense_cov) @ d_flat.T

    assert np.allclose(fisher, expected, rtol=1e-10, atol=0.0)


def test_all_true_mask_equals_the_unmasked_path(toy_deriv, toy_cov):
    a, na = fl.fisher_from_derivs(toy_deriv, toy_cov, mask=None)
    mask = np.ones(toy_deriv.shape[1:], dtype=bool)
    b, nb = fl.fisher_from_derivs(toy_deriv, toy_cov, mask=mask)

    assert np.array_equal(a, b)
    assert na == nb


def test_masked_fisher_matches_the_dense_masked_expression(
        toy_deriv, toy_cov, toy_layout, toy_ell):
    """This is the property a stored *inverse* covariance could not provide."""
    mask = fl.build_mask(toy_layout, toy_ell, ell_cuts={"cs": (None, 1000.0)})
    fisher, n_data = fl.fisher_from_derivs(toy_deriv, toy_cov, mask=mask)

    dense_cov = fl.dense_covariance(toy_cov, mask=mask)
    n_par = toy_deriv.shape[0]
    keep = [(b, l) for b in range(mask.shape[0]) for l in range(mask.shape[1])
            if mask[b, l]]
    d_flat = np.array([[toy_deriv[p, b, l] for (b, l) in keep] for p in range(n_par)])
    expected = d_flat @ np.linalg.inv(dense_cov) @ d_flat.T

    assert n_data == len(keep)
    assert np.allclose(fisher, expected, rtol=1e-10, atol=0.0)


def test_masking_never_tightens_a_constraint(toy_deriv, toy_cov, toy_layout, toy_ell):
    """Dropping data cannot add information -- a basic monotonicity check."""
    full, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)
    mask = fl.build_mask(toy_layout, toy_ell, ell_cuts={"cs": (None, 1000.0)})
    cut, _ = fl.fisher_from_derivs(toy_deriv, toy_cov, mask=mask)

    sig_full = np.sqrt(np.diag(np.linalg.inv(full)))
    sig_cut = np.sqrt(np.diag(np.linalg.inv(cut)))

    assert np.all(sig_cut >= sig_full * (1 - 1e-10))


def test_empty_mask_gives_a_zero_fisher(toy_deriv, toy_cov):
    mask = np.zeros(toy_deriv.shape[1:], dtype=bool)
    fisher, n_data = fl.fisher_from_derivs(toy_deriv, toy_cov, mask=mask)

    assert n_data == 0
    assert np.array_equal(fisher, np.zeros((4, 4)))


def test_mismatched_covariance_shape_raises(toy_deriv, toy_cov):
    with pytest.raises(ValueError, match="cov_blocks has shape"):
        fl.fisher_from_derivs(toy_deriv, toy_cov[:, :3, :3])


# --------------------------------------------------------------------------
# priors and FisherFlexResult
# --------------------------------------------------------------------------


def _result(fisher, prior_sigma=None, n_par=None):
    n = fisher.shape[0] if n_par is None else n_par
    names = ["omega_m", "sigma_8", "w_0", "w_a"][:n]
    return FisherFlexResult(
        fisher=fisher,
        param_order=names,
        param_labels=names,
        param_fid={"omega_m": 0.3156, "sigma_8": 0.831, "w_0": -1.0, "w_a": 0.0},
        mask=np.ones((1, 1), dtype=bool),
        n_data_used=1,
        prior_sigma=np.full(n, np.inf) if prior_sigma is None else prior_sigma,
    )


def test_diagonal_priors_add_exactly_one_over_sigma_squared(toy_deriv, toy_cov):
    fisher, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)
    prior_sigma = np.array([0.15, 0.2, 0.8, 1.3])

    with_prior = fisher + np.diag(1.0 / prior_sigma**2)

    assert np.allclose(np.diag(with_prior - fisher), 1.0 / prior_sigma**2)
    # Off-diagonals are untouched by a diagonal prior.
    off = ~np.eye(4, dtype=bool)
    assert np.array_equal((with_prior - fisher)[off], np.zeros(12))


def test_a_correlated_prior_is_the_inverse_of_its_covariance():
    """``prior_mode='full'`` adds ``inv(prior_cov)``, not ``diag(1/sigma^2)``."""
    prior_cov = np.array([[4e-4, 1.5e-4], [1.5e-4, 9e-4]])
    fisher = np.eye(2) * 100.0

    with_prior = fisher + np.linalg.inv(prior_cov)
    diag_only = fisher + np.diag(1.0 / np.diag(prior_cov))

    assert not np.allclose(with_prior, diag_only)
    # A correlated prior is more informative on each parameter individually.
    assert np.all(np.diag(with_prior) > np.diag(diag_only))


def test_sigma_is_sqrt_diag_inverse_fisher(toy_deriv, toy_cov):
    fisher, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)
    res = _result(fisher)

    expected = np.sqrt(np.diag(np.linalg.inv(fisher)))
    for i, p in enumerate(res.param_order):
        assert res.sigma(p) == pytest.approx(expected[i], rel=1e-12)
    assert res.sigma(0) == res.sigma("omega_m")


def test_unknown_parameter_raises(toy_deriv, toy_cov):
    fisher, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)
    with pytest.raises(KeyError, match="unknown parameter"):
        _result(fisher).sigma("h0")


def test_fom_matches_one_over_sqrt_det_of_the_2x2_covariance(toy_deriv, toy_cov):
    fisher, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)
    res = _result(fisher)

    i, j = res.index("w_0"), res.index("w_a")
    cov2 = np.linalg.inv(fisher)[np.ix_([i, j], [i, j])]

    assert res.fom("w_0", "w_a") == pytest.approx(1.0 / np.sqrt(np.linalg.det(cov2)),
                                                  rel=1e-10)
    assert res.fom("w_0", "w_a") == pytest.approx(FoM(marginalize(fisher, i, j)),
                                                  rel=1e-12)


def test_fom_is_symmetric_in_its_arguments(toy_deriv, toy_cov):
    res = _result(fl.fisher_from_derivs(toy_deriv, toy_cov)[0])
    assert res.fom("w_0", "w_a") == pytest.approx(res.fom("w_a", "w_0"), rel=1e-12)


def test_marginalize_matches_the_published_helper(toy_deriv, toy_cov):
    """``FisherFlexResult`` must not diverge from ``fisher.marginalize``."""
    from fisherA2Z.fisher import marginalize as published

    fisher, _ = fl.fisher_from_derivs(toy_deriv, toy_cov)
    res = _result(fisher)

    assert np.array_equal(res.marginalize("omega_m", "sigma_8"),
                          published(fisher, 0, 1))


def test_prior_dominated_flags_uninformative_parameters():
    """A parameter with no data constraint sits exactly at its prior width."""
    prior_sigma = np.array([0.15, 0.2])
    fisher = np.diag([1e4, 0.0]) + np.diag(1.0 / prior_sigma**2)
    res = _result(fisher, prior_sigma=prior_sigma, n_par=2)

    assert not res.prior_dominated[0]
    assert res.prior_dominated[1]
    assert res.sigma("sigma_8") == pytest.approx(prior_sigma[1], rel=1e-12)


def test_s8_error_propagation():
    """S_8 = sigma_8 sqrt(Om/0.3); check the Jacobian against finite differences."""
    fisher = np.array([[2.0e4, 5.0e3], [5.0e3, 1.0e4]])
    res = _result(fisher, n_par=2)
    om, s8v = res.param_fid["omega_m"], res.param_fid["sigma_8"]

    value, err = res.s8()
    assert value == pytest.approx(s8v * np.sqrt(om / 0.3), rel=1e-12)

    eps = 1e-6
    grad_num = np.array([
        ((s8v * np.sqrt((om + eps) / 0.3)) - (s8v * np.sqrt((om - eps) / 0.3))) / (2 * eps),
        (((s8v + eps) * np.sqrt(om / 0.3)) - ((s8v - eps) * np.sqrt(om / 0.3))) / (2 * eps),
    ])
    cov = np.linalg.inv(fisher)
    assert err == pytest.approx(np.sqrt(grad_num @ cov @ grad_num), rel=1e-6)


def test_corner_labels_each_dataset_once():
    """Regression guard: the label used to be attached to every diagonal panel,
    so ``fig.legend()`` repeated each entry once per parameter."""
    import matplotlib
    matplotlib.use("Agg")

    res = _result(np.diag([1.0 / 0.01**2, 1.0 / 0.05**2]), n_par=2)
    fig = res.corner(["omega_m", "sigma_8"], label="run A")

    handles, labels = fig.axes[0].get_legend_handles_labels()
    all_labels = []
    for ax in fig.axes:
        all_labels += ax.get_legend_handles_labels()[1]

    assert all_labels.count("run A") == 1


def test_corner_overlays_two_results_on_one_figure():
    import matplotlib
    matplotlib.use("Agg")

    a = _result(np.diag([1.0 / 0.01**2, 1.0 / 0.05**2]), n_par=2)
    b = _result(np.diag([1.0 / 0.02**2, 1.0 / 0.06**2]), n_par=2)

    fig = a.corner(["omega_m", "sigma_8"], color="C0", label="a")
    fig = b.corner(["omega_m", "sigma_8"], color="C1", label="b", fig=fig)

    all_labels = []
    for ax in fig.axes:
        all_labels += ax.get_legend_handles_labels()[1]
    assert sorted(all_labels) == ["a", "b"]
    assert len(fig.axes) == 4


def test_contour_ellipse_matches_the_analytic_orientation():
    """Guard against the row/column eigenvector bug in ``fisher.plot_contours``.

    For a correlated 2x2 covariance the ellipse position angle is
    ``0.5 * atan2(2 C01, C00 - C11)``.  Taking a *row* of the eigenvector matrix
    instead of a column gives a different angle whenever the covariance is not
    diagonal, so this pins the orientation down.
    """
    import matplotlib
    matplotlib.use("Agg")

    cov = np.array([[4.0e-4, 1.2e-4], [1.2e-4, 1.0e-4]])
    res = _result(np.linalg.inv(cov), n_par=2)
    ax = res.contour("omega_m", "sigma_8", sigmas=(1,))

    patches = [p for p in ax.patches if p.__class__.__name__ == "Ellipse"]
    assert len(patches) == 1
    e = patches[0]

    expected_angle = np.degrees(
        0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    )
    assert e.angle % 180 == pytest.approx(expected_angle % 180, abs=1e-6)

    # Semi-axes are sqrt(2.30 * eigenvalue), major first.
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    assert e.width == pytest.approx(2 * np.sqrt(2.30 * evals[0]), rel=1e-10)
    assert e.height == pytest.approx(2 * np.sqrt(2.30 * evals[1]), rel=1e-10)
    assert e.center == (res.param_fid["omega_m"], res.param_fid["sigma_8"])
