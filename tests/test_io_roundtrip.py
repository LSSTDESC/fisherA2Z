"""npz save/load round-trip.

The contract is strict: a forecast run from a reloaded file must be *bitwise*
identical to the in-memory one, and the file must load with
``allow_pickle=False`` so it is safe to distribute.
"""

import json

import numpy as np
import pytest

from fisherA2Z.fisher_flex import SCHEMA_VERSION, FisherFlex

pytestmark = pytest.mark.ccl


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """Compute once, save, and hand back ``(flex, path)``."""
    import fisherA2Z.nz_decomposition as nzd

    z = np.linspace(0.05, 2.5, 40)
    nz = np.array([
        0.9 * nzd.gaussian(z, 0.6, 0.15) + 0.1 * nzd.gaussian(z, 1.8, 0.25),
        0.9 * nzd.gaussian(z, 1.1, 0.18) + 0.1 * nzd.gaussian(z, 2.1, 0.25),
    ])
    nz /= np.trapz(nz, z, axis=1)[:, None]

    rng = np.random.default_rng(9)
    reals = np.stack([
        nz * (1 + rng.normal(0, 0.02, nz.shape)) for _ in range(10)
    ])
    reals = np.clip(reals, 0, None)
    reals /= np.trapz(reals, z, axis=-1)[..., None]

    flex = FisherFlex(
        nz_source=nz, nz_realizations=reals, z_grid=z,
        neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26, n_lens_bins=2,
        ell_edges=np.array([100.0, 300.0, 900.0, 2700.0]), verbose=False,
    )
    flex.compute(parallel=False)
    path = tmp_path_factory.mktemp("npz") / "flex.npz"
    flex.save(str(path))
    return flex, str(path)


def test_file_loads_without_pickle(saved):
    """A pickled npz would be an arbitrary-code-execution hazard to share."""
    _, path = saved
    d = np.load(path, allow_pickle=False)

    assert int(d["schema_version"]) == SCHEMA_VERSION
    for key in ("deriv", "cov_blocks", "data_vector", "ell", "ell_edges",
                "param_names", "param_fid", "param_step", "param_prior_sigma",
                "pz_prior_cov", "chi_eff_lens_Mpc", "z_eff_lens",
                "block_probe", "block_leg_a", "block_leg_b", "fsky", "sigma_e"):
        assert key in d, f"missing key {key}"


def test_payload_arrays_round_trip_exactly(saved):
    flex, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    for attr in ("deriv", "cov_blocks", "data_vector", "ell", "ell_edges",
                 "chi_eff_lens_Mpc", "z_eff_lens"):
        assert np.array_equal(getattr(reloaded, attr), getattr(flex, attr)), attr


def test_deriv_stays_three_dimensional(saved):
    """Flattening would destroy the (block, ell) factorization the solve needs."""
    flex, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    assert reloaded.deriv.ndim == 3
    assert reloaded.deriv.shape == (flex.n_params, flex.layout.n_blocks,
                                    flex.ell.size)


def test_forecast_from_disk_is_bitwise_identical(saved):
    flex, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    a = flex.forecast()
    b = reloaded.forecast()

    assert np.array_equal(a.fisher, b.fisher)
    assert np.array_equal(a.mask, b.mask)
    assert a.n_data_used == b.n_data_used
    assert a.param_order == b.param_order
    assert a.sigmas() == b.sigmas()
    assert a.fom("w_0", "w_a") == b.fom("w_0", "w_a")


def test_precompute_kwarg_matches_from_npz(saved):
    """``compute(precompute=...)`` is the in-place equivalent of ``from_npz``."""
    flex, path = saved
    other = FisherFlex.__new__(FisherFlex)
    other.verbose = False
    other.mode = flex.mode
    other.n_source = flex.n_source
    other.z = flex.z
    other._load_npz(path)

    assert np.array_equal(other.deriv, flex.deriv)
    assert np.array_equal(other.cov_blocks, flex.cov_blocks)


def test_layout_round_trips(saved):
    flex, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    assert np.array_equal(reloaded.layout.probe, flex.layout.probe)
    assert np.array_equal(reloaded.layout.leg_a, flex.layout.leg_a)
    assert np.array_equal(reloaded.layout.leg_b, flex.layout.leg_b)
    assert reloaded.layout.n_blocks == flex.layout.n_blocks
    assert reloaded.layout.probe_block_slices() == flex.layout.probe_block_slices()


def test_parameter_tables_round_trip(saved):
    flex, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    assert reloaded.param_order == flex.param_order
    assert reloaded.param_labels == flex.param_labels
    assert reloaded.fid == pytest.approx(flex.fid)
    assert reloaded.param_step == pytest.approx(flex.param_step)
    assert reloaded.prior_sigma == pytest.approx(flex.prior_sigma)
    assert np.allclose(reloaded.pz_prior_cov, flex.pz_prior_cov)


def test_survey_properties_round_trip(saved):
    flex, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    assert reloaded.fsky == flex.fsky
    assert reloaded.sigma_e == flex.sigma_e
    assert np.array_equal(reloaded.neff_source, flex.neff_source)
    assert np.array_equal(reloaded.neff_lens, flex.neff_lens)
    assert reloaded.ggl_pairs == flex.ggl_pairs


def test_cosmo_kwargs_json_reparses(saved):
    flex, path = saved
    d = np.load(path, allow_pickle=False)

    parsed = json.loads(str(d["cosmo_kwargs_json"]))
    assert parsed == flex.cosmo_kwargs
    assert parsed["transfer_function"] == "eisenstein_hu"


def test_from_npz_needs_no_ccl_evaluation(saved):
    """The forecast-only instance carries no live cosmology object."""
    _, path = saved
    reloaded = FisherFlex.from_npz(path, verbose=False)

    assert reloaded.cosmo is None
    reloaded.forecast()  # must still work: chi_eff is stored, not recomputed


def test_from_npz_is_warning_free(saved):
    """``from_npz`` has no live config to compare against, so it must stay quiet.

    Regression guard: it used to seed placeholder ``n_source=0`` / ``z=[]``
    values before loading, which tripped the staleness warnings on every single
    call and so trained users to ignore them.
    """
    import warnings as _warnings

    _, path = saved
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        FisherFlex.from_npz(path, verbose=False)


def test_tampered_schema_version_raises(saved, tmp_path):
    flex, path = saved
    d = dict(np.load(path, allow_pickle=False))
    d["schema_version"] = np.array(SCHEMA_VERSION + 1)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(str(bad), **d)

    with pytest.raises(ValueError, match="schema_version"):
        FisherFlex.from_npz(str(bad), verbose=False)


def test_mismatched_z_grid_warns(saved):
    """A stale npz must never silently forecast the wrong n(z)."""
    flex, path = saved
    other = FisherFlex.__new__(FisherFlex)
    other.verbose = False
    other.mode = flex.mode
    other.n_source = flex.n_source
    other.z = np.linspace(0.05, 2.5, 37)  # different grid

    with pytest.warns(UserWarning, match="different redshift grid"):
        other._load_npz(path)


def test_mismatched_n_tomo_warns(saved):
    flex, path = saved
    other = FisherFlex.__new__(FisherFlex)
    other.verbose = False
    other.mode = flex.mode
    other.n_source = 5           # npz has 2
    other.z = flex.z

    with pytest.warns(UserWarning, match="source bins"):
        other._load_npz(path)


def test_the_photoz_model_round_trips(saved):
    """``nz_model`` and its pivot have to survive, or ``forecast`` mis-groups.

    ``forecast`` needs ``nz_model`` to know how many photo-z parameters each
    bin has before it can apply the per-bin prior block, and a reloaded
    instance never runs ``__init__``.
    """
    flex, path = saved
    d = np.load(path, allow_pickle=False)

    assert str(d["nz_model"]) == flex.nz_model == "shift_stretch"
    assert json.loads(str(d["config_json"]))["nz_model"] == flex.nz_model
    assert np.array_equal(d["pz_pivot"], flex.pz_pivot)
    assert np.array_equal(d["nz_source_fid"], flex.nz_source_fid)
    assert d["pz_prior_cov"].shape == (flex.n_source, 2, 2)

    reloaded = FisherFlex.from_npz(path, verbose=False)
    assert reloaded.nz_model == flex.nz_model
    assert reloaded.n_pz_per_bin == 2
    assert np.array_equal(reloaded.pz_pivot, flex.pz_pivot)


def test_mismatched_nz_model_warns(saved):
    """Loading shift-stretch derivatives into a Gaussian-model instance."""
    flex, path = saved
    other = FisherFlex.__new__(FisherFlex)
    other.verbose = False
    other.mode = flex.mode
    other.n_source = flex.n_source
    other.z = flex.z
    other.nz_model = "gaussian_outlier"

    with pytest.warns(UserWarning, match="nz_model"):
        other._load_npz(path)

    # The stored model wins, so the prior grouping stays consistent with the
    # derivatives that are actually on disk.
    assert other.nz_model == "shift_stretch"
    assert other.n_pz_per_bin == 2
