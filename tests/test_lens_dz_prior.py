"""Lens mean-shift nuisance parameters (``lens_dz_prior``)."""

import numpy as np
import pytest

from fisherA2Z.fisher_flex import FisherFlex

pytestmark = pytest.mark.ccl


def _inputs():
    import fisherA2Z.nz_decomposition as nzd

    z = np.linspace(0.05, 2.5, 40)
    nz = np.array([nzd.gaussian(z, 0.6, 0.15), nzd.gaussian(z, 1.1, 0.18)])
    nz /= np.trapz(nz, z, axis=1)[:, None]
    cov = np.array([np.diag([0.01 ** 2, 0.02 ** 2])] * 2)
    return dict(nz_source=nz, photoz_prior_cov=cov, z_grid=z,
                neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26, n_lens_bins=2,
                ell_edges=np.array([100.0, 300.0, 900.0, 2700.0]), verbose=False)


@pytest.fixture(scope="module")
def pair():
    off = FisherFlex(**_inputs()).compute(parallel=False)
    on = FisherFlex(**_inputs(), lens_dz_prior=[0.01, 0.02]).compute(parallel=False)
    return off, on


def test_default_unchanged(pair):
    off, _ = pair
    assert off.lens_dz_prior is None
    assert not any(p.startswith("lzbias") for p in off.param_order)


def test_params_and_priors(pair):
    _, on = pair
    assert [p for p in on.param_order if p.startswith("lzbias")] == ["lzbias1", "lzbias2"]
    assert on.prior_sigma["lzbias1"] == 0.01 and on.prior_sigma["lzbias2"] == 0.02
    assert on.fid["lzbias1"] == 0.0


def test_fiducial_data_vector_unchanged(pair):
    off, on = pair
    np.testing.assert_allclose(on.data_vector, off.data_vector, rtol=1e-12)  # renormalization round-off


def test_lens_shift_moves_only_clustering_legs(pair):
    _, on = pair
    d = on.deriv[on.param_order.index("lzbias1")]
    probe = np.asarray(on.layout.probe)
    assert np.all(d[probe == "cs"] == 0)
    assert np.any(d[probe != "cs"] != 0)


def test_lens_prior_costs_constraining_power(pair):
    off, on = pair
    s_off = off.forecast().s8()[1]
    s_on = on.forecast().s8()[1]
    assert s_on >= s_off
    # a vanishing lens prior recovers the lens-known forecast
    tight = on.override_priors({"lzbias1": 1e-7, "lzbias2": 1e-7}).forecast().s8()[1]
    np.testing.assert_allclose(tight, s_off, rtol=1e-6)


def test_cosmic_shear_ignores_lens_params():
    off = FisherFlex(**_inputs(), mode="cosmic_shear").compute(parallel=False)
    on = FisherFlex(**_inputs(), mode="cosmic_shear",
                    lens_dz_prior=0.01).compute(parallel=False)
    np.testing.assert_allclose(on.forecast().s8()[1], off.forecast().s8()[1], rtol=1e-10)


def test_roundtrip(pair, tmp_path):
    _, on = pair
    path = str(tmp_path / "lz.npz")
    on.save(path)
    back = FisherFlex.from_npz(path)
    np.testing.assert_array_equal(back.lens_dz_prior, on.lens_dz_prior)
    assert back.forecast().s8()[1] == on.forecast().s8()[1]
