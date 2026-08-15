"""``nz_model='no_uncertainty'``: n(z) exactly known, no photo-z parameters.

The defining property is the equivalence test: this model must be bitwise the
same as ``'shift_stretch'`` with every photo-z parameter fixed.  Both share the
same template, so the fiducial n(z), the data vector and the cosmology
derivatives are identical, and the only difference is which parameters are
carried.  Anything else would mean one of the two paths is doing extra work to
the n(z).
"""

import numpy as np
import pytest

from fisherA2Z.fisher_flex import FisherFlex, N_PZ_PER_BIN, NZ_MODELS


Z_GRID = np.linspace(0.05, 2.5, 50)
BIN_MU = (0.6, 1.1)
BIN_SIGMA = (0.15, 0.18)
BIN_OUT_MU = (1.8, 2.1)

SURVEY = dict(z_grid=Z_GRID, neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26,
              n_lens_bins=2, ell_edges=np.geomspace(100.0, 3000.0, 9),
              verbose=False)


def _toy_inputs():
    import fisherA2Z.nz_decomposition as nzd

    rng = np.random.default_rng(5)
    reals = np.empty((12, 2, Z_GRID.size))
    for r in range(12):
        for b in range(2):
            reals[r, b] = nzd.build_model_nz(
                Z_GRID, BIN_MU[b] + rng.normal(0, 0.01), BIN_SIGMA[b],
                0.1 + rng.normal(0, 0.01),
                nzd.gaussian(Z_GRID, BIN_OUT_MU[b], 0.25),
            )
    return reals.mean(axis=0), reals


# --------------------------------------------------------------------------
# construction -- no CCL
# --------------------------------------------------------------------------


def test_model_is_registered_with_zero_parameters_per_bin():
    assert "no_uncertainty" in NZ_MODELS
    assert N_PZ_PER_BIN["no_uncertainty"] == 0


def test_realizations_may_be_omitted_entirely():
    """The whole point: no second argument, no photoz_prior_cov."""
    nz, _ = _toy_inputs()
    flex = FisherFlex(nz_source=nz, nz_model="no_uncertainty", **SURVEY)

    assert flex.n_pz_per_bin == 0
    assert flex.pz_prior_cov.shape == (2, 0, 0)
    assert flex.realization_params is None
    for i in range(flex.n_source):
        assert flex._pz_param_names(i) == []


def test_other_models_still_demand_a_prior_source():
    """Making the argument optional must not silently weaken the other models."""
    nz, _ = _toy_inputs()
    for model in ("shift_stretch", "gaussian_outlier"):
        with pytest.raises(ValueError, match="nz_realizations or photoz_prior_cov"):
            FisherFlex(nz_source=nz, nz_model=model, **SURVEY)


def test_missing_survey_arguments_still_raise():
    """The None defaults exist only to free up nz_realizations."""
    nz, _ = _toy_inputs()
    with pytest.raises(TypeError, match="fsky"):
        FisherFlex(nz_source=nz, z_grid=Z_GRID, neff_source=[2.0, 2.0],
                   sigma_e=0.26, nz_model="no_uncertainty", verbose=False)
    with pytest.raises(TypeError, match="z_grid, neff_source"):
        FisherFlex(nz_source=nz, fsky=0.1, sigma_e=0.26,
                   nz_model="no_uncertainty", verbose=False)


def test_positional_call_order_is_unchanged():
    """Existing positional callers must keep working."""
    nz, reals = _toy_inputs()
    flex = FisherFlex(nz, reals, Z_GRID, [2.0, 2.0], 0.1, 0.26,
                      n_lens_bins=2, verbose=False)
    assert flex.nz_model == "shift_stretch"
    assert flex.n_pz_per_bin == 2


def test_unused_prior_inputs_warn():
    nz, reals = _toy_inputs()
    with pytest.warns(UserWarning, match="no photo-z parameters"):
        FisherFlex(nz_source=nz, nz_realizations=reals,
                   nz_model="no_uncertainty", **SURVEY)


def test_parameter_vector_has_no_photoz_entries():
    nz, _ = _toy_inputs()
    flex = FisherFlex(nz_source=nz, nz_model="no_uncertainty", **SURVEY)

    assert flex.n_params == 7 + 4 + 2          # cosmology + IA + 2 gbias
    assert not [p for p in flex.param_order
                if p.startswith(("zbias", "zstretch", "zvariance", "zoutlier"))]
    for p in ("omega_m", "sigma_8", "w_0", "w_a", "A0", "gbias1", "gbias2"):
        assert p in flex.param_order
    assert set(flex.fid) == set(flex.param_order)
    assert set(flex.prior_sigma) >= set(flex.param_order)


def test_prior_stability_report_is_refused():
    nz, reals = _toy_inputs()
    flex = FisherFlex(nz_source=nz, nz_model="no_uncertainty", **SURVEY)
    with pytest.raises(RuntimeError, match="no photo-z parameters"):
        flex.prior_stability_report(reals)


# --------------------------------------------------------------------------
# end to end -- small CCL evaluations
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair():
    """The same setup under 'no_uncertainty' and 'shift_stretch'."""
    nz, reals = _toy_inputs()
    a = FisherFlex(nz_source=nz, nz_model="no_uncertainty",
                   **SURVEY).compute(parallel=False)
    b = FisherFlex(nz_source=nz, nz_realizations=reals,
                   nz_model="shift_stretch", **SURVEY).compute(parallel=False)
    return a, b


@pytest.mark.ccl
def test_matches_shift_stretch_with_photoz_fixed(pair):
    """The equivalence that defines the model, asserted exactly."""
    a, b = pair
    pz = [p for i in range(b.n_source) for p in b._pz_param_names(i)]

    assert np.array_equal(a.nz_source_fid, b.nz_source_fid)
    assert np.array_equal(a.data_vector, b.data_vector)
    assert np.array_equal(a.cov_blocks, b.cov_blocks)

    ra, rb = a.forecast(), b.forecast(drop_params=pz)
    assert ra.param_order == rb.param_order
    assert np.array_equal(ra.fisher, rb.fisher)


@pytest.mark.ccl
def test_the_nz_is_frozen_against_every_parameter(pair):
    """No parameter may move n(z), so no derivative can act through it."""
    a, _ = pair
    model = a._model()
    base = model.source_nz(a.fid)
    # The template is renormalized on the way out, which moves the last bit;
    # normalization is not observable, since CCL normalizes n(z) internally.
    assert np.allclose(base, a.nz_source_fid, rtol=1e-12, atol=0)
    for p in ("omega_m", "gbias1", "A0"):
        vals = dict(a.fid)
        vals[p] = a.fid[p] + 0.1
        assert np.array_equal(model.source_nz(vals), base)


@pytest.mark.ccl
def test_it_is_the_optimistic_bound(pair):
    """Knowing n(z) exactly can never be worse than marginalizing over it."""
    a, b = pair
    ra, rb = a.forecast(), b.forecast()
    assert ra.n_data_used == rb.n_data_used
    for p in ra.param_order:
        assert ra.sigma(p) <= rb.sigma(p) * (1 + 1e-12)
    assert ra.fom("omega_m", "sigma_8") >= rb.fom("omega_m", "sigma_8")


@pytest.mark.ccl
def test_npz_roundtrip(pair, tmp_path):
    a, _ = pair
    path = tmp_path / "no_uncertainty.npz"
    a.save(path)

    g = FisherFlex.from_npz(path, verbose=False)
    assert g.nz_model == "no_uncertainty"
    assert g.n_pz_per_bin == 0
    assert g.param_order == a.param_order
    assert np.array_equal(g.forecast().fisher, a.forecast().fisher)


@pytest.mark.ccl
def test_forecast_bias_still_works(pair):
    """n(z) being fixed does not make the forecast insensitive to it.

    It makes the bias *worse*: nothing is free to absorb the systematic.
    """
    a, b = pair
    vals = dict(b.fid)
    vals["zbias1"] = 5e-3
    nz_truth = b._model().source_nz(vals)

    bias_a = a.forecast_bias(nz_truth=nz_truth)
    bias_b = b.forecast_bias(nz_truth=nz_truth)

    assert not [p for p in bias_a.param_order if p.startswith("z")]
    assert bias_a.mafe() > 0
    assert abs(bias_a.shift("omega_m")) > 0
    # Same residual, but the shift_stretch run can partly absorb it, so its
    # cosmology has to move less.
    assert abs(bias_a.shift("omega_m")) >= abs(bias_b.shift("omega_m"))
    assert bias_a.bias_2d("omega_m", "sigma_8").chi2 >= 0
