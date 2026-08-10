"""End-to-end smoke tests for ``FisherFlex.compute`` and ``forecast``.

Kept fast by using 3 multipoles, 2 source bins, 2 lens bins and the
Eisenstein-Hu transfer function -- no Boltzmann code is ever called.  The
``test_gbias_derivative_matches_the_closed_form`` case is the important one: it
validates the whole numerical-differentiation machinery against an analytic
result.
"""

import numpy as np
import pytest

from fisherA2Z import flex_layout as fl
from fisherA2Z.fisher_flex import FisherFlex

pytestmark = pytest.mark.ccl


Z_GRID = np.linspace(0.05, 2.5, 50)
BIN_MU = (0.6, 1.1)
BIN_SIGMA = (0.15, 0.18)
BIN_OUT_MU = (1.8, 2.1)


def _toy_inputs():
    """12 realizations, and their mean as the central n(z).

    Taking the mean is both what the real workflow does (the tutorial's central
    n(z) is the average over Dirichlet draws) and the only way to get the
    central n(z) exactly centred on the realizations.  Building it instead at
    the mean *parameters* leaves it off-centre by the sampling error of 12
    draws -- measured at 2.1x the realization scatter in bin 2 -- which
    ``FisherFlex`` rightly warns about.  Every realization integrates to 1, so
    the mean does too and the mean shift is zero by construction.
    """
    import fisherA2Z.nz_decomposition as nzd

    z = Z_GRID

    def build(b, mu, f_out):
        return nzd.build_model_nz(z, mu, BIN_SIGMA[b], f_out,
                                  nzd.gaussian(z, BIN_OUT_MU[b], 0.25))

    rng = np.random.default_rng(5)
    reals = np.empty((12, 2, z.size))
    for r in range(12):
        for b in range(2):
            reals[r, b] = build(b, BIN_MU[b] + rng.normal(0, 0.01),
                                0.1 + rng.normal(0, 0.01))
    return z, reals.mean(axis=0), reals


def _make_flex(**kwargs):
    z, nz, reals = _toy_inputs()
    flex = FisherFlex(
        nz_source=nz, nz_realizations=reals, z_grid=z,
        neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26,
        n_lens_bins=2, ell_edges=np.array([100.0, 300.0, 900.0, 2700.0]),
        verbose=False, **kwargs,
    )
    return flex.compute(parallel=False)


@pytest.fixture(scope="module")
def small_flex():
    """A minimal but complete 3x2pt setup, computed serially once per module.

    Uses the default ``nz_model='shift_stretch'``.
    """
    return _make_flex()


@pytest.fixture(scope="module")
def small_flex_gaussian():
    """The same setup under the A2Z Gaussian-core-plus-outlier photo-z model."""
    return _make_flex(nz_model="gaussian_outlier")


def test_compute_produces_correctly_shaped_finite_arrays(small_flex):
    n_par = small_flex.n_params
    n_blocks = small_flex.layout.n_blocks
    n_ell = small_flex.ell.size

    assert small_flex.data_vector.shape == (n_blocks, n_ell)
    assert small_flex.deriv.shape == (n_par, n_blocks, n_ell)
    assert small_flex.cov_blocks.shape == (n_ell, n_blocks, n_blocks)
    assert np.all(np.isfinite(small_flex.deriv))
    assert np.all(np.isfinite(small_flex.data_vector))
    assert np.all(small_flex.data_vector > 0)


def test_parameter_vector_has_the_expected_composition(small_flex):
    """7 cosmological + 4 IA + 2 per source bin + 1 per lens bin.

    The shift-and-stretch model carries two photo-z nuisances per bin; there is
    no outlier fraction, which is the same thing as fixing it at zero with a
    delta-function prior.
    """
    assert small_flex.n_params == 7 + 4 + 2 * 2 + 2
    assert len(small_flex.param_order) == small_flex.n_params
    assert len(set(small_flex.param_order)) == small_flex.n_params
    for p in ("omega_m", "sigma_8", "w_0", "w_a", "n_s", "omega_b", "h"):
        assert p in small_flex.param_order
    for p in ("A0", "beta", "etal", "etah"):
        assert p in small_flex.param_order
    for b in (1, 2):
        assert f"zbias{b}" in small_flex.param_order
        assert f"zstretch{b}" in small_flex.param_order
        assert f"zoutlier{b}" not in small_flex.param_order


def test_the_gaussian_model_keeps_its_three_photoz_parameters(small_flex_gaussian):
    """The older A2Z-style model is still selectable and still has ``f_out``."""
    flex = small_flex_gaussian
    assert flex.n_params == 7 + 4 + 3 * 2 + 2
    for b in (1, 2):
        assert f"zbias{b}" in flex.param_order
        assert f"zvariance{b}" in flex.param_order
        assert f"zoutlier{b}" in flex.param_order


def test_the_fiducial_source_nz_is_the_tabulated_data(small_flex,
                                                      small_flex_gaussian):
    """The whole point of the shift-and-stretch model.

    At ``(dz, s) = (0, 1)`` the model returns the measured n(z) itself, so the
    data vector, the covariance and the derivatives are all built on the data
    rather than on a fit to it.  The agreement is to round-off rather than
    bitwise only because the model renormalizes an input that already
    integrates to 1 to within an ulp.

    ``gaussian_outlier`` goes through a fit and so lands somewhere else.  On
    this fixture it lands only 1e-5 away, because the toy n(z) really is a
    Gaussian plus a well-separated outlier island -- the one case the
    decomposition was built for.  The size of that gap on a *realistic* n(z) is
    tested without CCL in ``test_nz_shift_stretch.py``; all that is checked
    here is that the two models are genuinely different code paths.
    """
    from fisherA2Z.fisher_flex import _FlexModel

    data = small_flex.nz_source_fid
    got = _FlexModel(small_flex._to_state()).source_nz(small_flex.fid)
    assert np.allclose(got, data, rtol=1e-12, atol=0.0)

    fit = _FlexModel(small_flex_gaussian._to_state()).source_nz(
        small_flex_gaussian.fid)
    assert np.abs(fit - data).max() / data.max() > 1e-9


def test_gbias_derivative_matches_the_closed_form(small_flex):
    """``C^gg_ii ~ b_i^2``, so ``dC/db_i = 2 C / b_i`` exactly.

    This validates the numerical differentiation, the parameter plumbing and the
    block layout in one shot -- a wrong step size, a mislabelled parameter or a
    mis-indexed block all break it.
    """
    gc = small_flex.layout.probe_slice("gc")
    for b in range(small_flex.n_lens):
        name = f"gbias{b + 1}"
        i = small_flex.param_order.index(name)
        block = gc.start + b

        cl = small_flex.data_vector[block]
        expected = 2.0 * cl / small_flex.fid[name]
        got = small_flex.deriv[i, block]

        assert np.allclose(got, expected, rtol=1e-6, atol=0.0), (
            f"{name}: got {got}, expected {expected}"
        )


def test_gbias_only_affects_its_own_lens_bin(small_flex):
    """Linear bias is per-bin; the clustering auto-spectra must not mix."""
    gc = small_flex.layout.probe_slice("gc")
    i = small_flex.param_order.index("gbias1")

    assert np.all(small_flex.deriv[i, gc.start] != 0)
    assert np.all(small_flex.deriv[i, gc.start + 1] == 0)
    # ... and it leaves cosmic shear alone entirely.
    cs = small_flex.layout.probe_slice("cs")
    assert np.all(small_flex.deriv[i, cs] == 0)


def test_shear_power_increases_with_sigma8(small_flex):
    """A sign check that catches a flipped derivative or a swapped parameter."""
    i = small_flex.param_order.index("sigma_8")
    cs = small_flex.layout.probe_slice("cs")

    assert np.all(small_flex.deriv[i, cs] > 0)


def test_chi_eff_lens_matches_a_direct_ccl_call(small_flex):
    import pyccl as ccl

    expected = ccl.comoving_radial_distance(
        small_flex.cosmo, 1.0 / (1.0 + small_flex.z_eff_lens))

    assert np.allclose(small_flex.chi_eff_lens_Mpc, expected, rtol=1e-14, atol=0.0)
    assert np.all(np.diff(small_flex.chi_eff_lens_Mpc) > 0)


def test_the_default_lens_neff_is_a_total_split_across_bins():
    """48 arcmin^-2 is the SRD *gold sample*, not the density of each bin."""
    z, nz, reals = _toy_inputs()
    flex = FisherFlex(
        nz_source=nz, nz_realizations=reals, z_grid=z, neff_source=[2.0, 2.0],
        fsky=0.1, sigma_e=0.26, n_lens_bins=10, verbose=False,
    )

    assert flex.neff_lens.size == 10
    assert flex.neff_lens.sum() == pytest.approx(48.0, rel=1e-9)
    assert np.all(flex.neff_lens < 10.0)


def test_a_scalar_lens_neff_warns_that_it_is_per_bin():
    """Broadcasting a scalar multiplies the sample by n_lens; say so."""
    z, nz, reals = _toy_inputs()

    with pytest.warns(UserWarning, match="per \\*bin\\*"):
        flex = FisherFlex(
            nz_source=nz, nz_realizations=reals, z_grid=z,
            neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26,
            n_lens_bins=4, lens_neff=12.0, verbose=False,
        )

    assert np.allclose(flex.neff_lens, 12.0)
    assert flex.neff_lens.sum() == pytest.approx(48.0)


def test_a_per_bin_lens_neff_array_is_taken_verbatim(recwarn):
    z, nz, reals = _toy_inputs()
    want = [3.0, 4.0, 5.0, 6.0]
    flex = FisherFlex(
        nz_source=nz, nz_realizations=reals, z_grid=z, neff_source=[2.0, 2.0],
        fsky=0.1, sigma_e=0.26, n_lens_bins=4, lens_neff=want, verbose=False,
    )

    assert np.allclose(flex.neff_lens, want)
    assert not [w for w in recwarn if "per *bin*" in str(w.message)]


def test_lensing_kernel_spline_integration_is_disabled():
    """Process-global CCL setting that A2Z relies on; must be set on import."""
    import pyccl as ccl

    assert ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION is False


def test_default_ell_cuts_apply_the_k_cut(small_flex):
    cuts = small_flex.default_ell_cuts()

    assert cuts["cs"] == (None, 3000.0)
    expected = fl.ell_max_kcut(small_flex.chi_eff_lens_Mpc,
                               float(small_flex.cosmo_kwargs["h"]), 0.3)
    assert np.allclose(cuts["gc"][1], expected)
    assert np.allclose(cuts["ggl"][1], expected)


def test_default_ell_cuts_require_compute(small_flex):
    flex = FisherFlex.__new__(FisherFlex)
    flex.chi_eff_lens_Mpc = None
    with pytest.raises(RuntimeError, match="call compute"):
        flex.default_ell_cuts()


# --------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------


def test_forecast_gives_a_positive_definite_fisher(small_flex):
    res = small_flex.forecast()

    assert res.fisher.shape == (small_flex.n_params, small_flex.n_params)
    assert np.all(np.linalg.eigvalsh(res.fisher) > 0)
    assert res.n_data_used > 0
    assert res.n_data_used == int(res.mask.sum())
    assert np.all(np.isfinite(list(res.sigmas().values())))


def test_forecast_before_compute_raises():
    flex = FisherFlex.__new__(FisherFlex)
    flex._computed = False
    with pytest.raises(RuntimeError, match="call compute"):
        flex.forecast()


def test_tighter_scale_cuts_never_tighten_a_constraint(small_flex):
    """The headline monotonicity check: less data cannot mean more information."""
    loose = small_flex.forecast()
    tight = small_flex.forecast(ell_max_cs=400.0)

    assert tight.n_data_used < loose.n_data_used
    for p in small_flex.param_order:
        assert tight.sigma(p) >= loose.sigma(p) * (1 - 1e-9), p


def test_dropping_a_probe_loosens_constraints(small_flex):
    full = small_flex.forecast()
    shear_only = small_flex.forecast(probes=["cs"])

    assert shear_only.n_data_used < full.n_data_used
    assert shear_only.sigma("sigma_8") >= full.sigma("sigma_8")


def test_galaxy_bias_is_prior_dominated_in_a_shear_only_forecast(small_flex):
    """With no clustering data the gbias parameters carry zero derivative."""
    res = small_flex.forecast(probes=["cs"])
    i = res.index("gbias1")

    assert res.prior_dominated[i]
    assert res.sigma("gbias1") == pytest.approx(res.prior_sigma[i], rel=1e-6)


def test_priors_can_be_switched_off(small_flex):
    """``prior=False`` removes exactly the prior and nothing else.

    Stated on the Fisher matrices, not on the sigmas: this toy setup has 3
    multipoles and 13 data points against 17 parameters, so the prior-free
    Fisher is singular *by construction* and its sigmas are NaN.  An earlier
    version of this test compared sigmas and passed only because the
    pseudo-inverse of a singular matrix happened to return a finite number.
    """
    with_prior = small_flex.forecast(prior=True)
    without = small_flex.forecast(prior=False)

    names = with_prior.param_order
    assert names == without.param_order

    # Build the prior matrix independently: diagonal everywhere, except the
    # per-bin photo-z blocks, which carry the full realization covariance.
    expected = np.diag(1.0 / np.array(
        [small_flex.prior_sigma[p] for p in names]) ** 2)
    for b in range(small_flex.n_source):
        idx = [names.index(t) for t in small_flex._pz_param_names(b)]
        expected[np.ix_(idx, idx)] = np.linalg.inv(small_flex.pz_prior_cov[b])

    assert np.allclose(with_prior.fisher - without.fisher, expected,
                       rtol=1e-10, atol=0.0)


def test_dropping_the_prior_loosens_an_invertible_sub_forecast(small_flex):
    """The monotonicity claim, made where it is actually well posed.

    Fixing the parameters that the 13-point toy data vector cannot constrain
    leaves a 4-parameter problem whose prior-free Fisher is invertible
    (condition ratio ~7e-4), and there the prior can only tighten.
    """
    fixed = ["A0", "beta", "etal", "etah", "zbias1", "zbias2",
             "zstretch1", "zstretch2", "n_s", "omega_b", "h", "w_0", "w_a"]
    with_prior = small_flex.forecast(drop_params=fixed, prior=True)
    without = small_flex.forecast(drop_params=fixed, prior=False)

    assert without.param_order == ["omega_m", "sigma_8", "gbias1", "gbias2"]
    assert np.all(np.isfinite(list(without.sigmas().values())))
    for p in without.param_order:
        assert without.sigma(p) >= with_prior.sigma(p) * (1 - 1e-9), p


def test_drop_params_fixes_rather_than_marginalizes(small_flex):
    """A fixed parameter carries no uncertainty, so the others get tighter."""
    full = small_flex.forecast()
    dropped = small_flex.forecast(drop_params=["w_0", "w_a"])

    assert "w_0" not in dropped.param_order and "w_a" not in dropped.param_order
    assert len(dropped.param_order) == len(full.param_order) - 2
    assert dropped.sigma("omega_m") <= full.sigma("omega_m") * (1 + 1e-9)


def test_restricting_bin_pairs_reduces_the_data(small_flex):
    res = small_flex.forecast(bin_pairs_cs=[(0, 0)])

    full = small_flex.forecast()
    assert res.n_data_used < full.n_data_used


def test_figures_of_merit_are_positive_and_finite(small_flex):
    res = small_flex.forecast()

    for pair in (("omega_m", "sigma_8"), ("w_0", "w_a")):
        fom = res.fom(*pair)
        assert np.isfinite(fom) and fom > 0


def test_more_sky_never_loosens_a_constraint(small_flex):
    """fsky enters only through nmodes, so quadrupling it quadruples the Fisher.

    Stated on the Fisher matrix rather than on the sigmas: this toy setup has 19
    parameters and only 3 multipoles, so the prior-free Fisher is rank deficient
    and cannot be inverted.  The scaling law is the substantive claim anyway.
    """
    fisher_small, _ = fl.fisher_from_derivs(small_flex.deriv, small_flex.cov_blocks)
    fisher_big, _ = fl.fisher_from_derivs(small_flex.deriv, small_flex.cov_blocks / 4.0)

    assert np.allclose(fisher_big, 4.0 * fisher_small, rtol=1e-10, atol=0.0)

    # With priors added the matrix is invertible and the sigmas do shrink.
    prior = np.diag(1.0 / np.array(
        [small_flex.prior_sigma[p] for p in small_flex.param_order]) ** 2)
    sig_small = np.sqrt(np.diag(np.linalg.inv(fisher_small + prior)))
    sig_big = np.sqrt(np.diag(np.linalg.inv(fisher_big + prior)))
    assert np.all(sig_big <= sig_small * (1 + 1e-12))
