"""Serial-vs-parallel equivalence for the derivative step.

The claim being tested is strict *bitwise* equality, not approximate agreement.
It holds by construction: parent and children run the same module-level
function over the same worker globals, ``pool.map`` preserves order, and the
results are stacked in a fixed parameter order, so the reduction order in
``D^T C^-1 D`` is identical.
"""

import multiprocessing as mp

import numpy as np
import pytest

from fisherA2Z import fisher_flex as ff
from fisherA2Z.fisher_flex import FisherFlex

pytestmark = pytest.mark.ccl


@pytest.fixture(scope="module")
def flex():
    import fisherA2Z.nz_decomposition as nzd

    z = np.linspace(0.05, 2.5, 40)
    nz = np.array([
        0.9 * nzd.gaussian(z, 0.6, 0.15) + 0.1 * nzd.gaussian(z, 1.8, 0.25),
        0.9 * nzd.gaussian(z, 1.1, 0.18) + 0.1 * nzd.gaussian(z, 2.1, 0.25),
    ])
    nz /= np.trapz(nz, z, axis=1)[:, None]

    rng = np.random.default_rng(21)
    reals = np.clip(
        np.stack([nz * (1 + rng.normal(0, 0.02, nz.shape)) for _ in range(8)]),
        0, None)
    reals /= np.trapz(reals, z, axis=-1)[..., None]

    return FisherFlex(
        nz_source=nz, nz_realizations=reals, z_grid=z,
        neff_source=[2.0, 2.0], fsky=0.1, sigma_e=0.26, n_lens_bins=2,
        ell_edges=np.array([100.0, 600.0, 3000.0]), verbose=False,
    )


@pytest.fixture(scope="module")
def computed(flex):
    """Run ``compute`` exactly once each way; every test below reuses these.

    The serial pass also asserts that ``parallel=False`` never reaches
    multiprocessing, which would otherwise need a third full compute.
    """
    def boom(*a, **kw):
        raise AssertionError("multiprocessing was used despite parallel=False")

    original = ff._make_pool
    ff._make_pool = boom
    try:
        flex.compute(parallel=False)
    finally:
        ff._make_pool = original
    serial_deriv = flex.deriv.copy()
    serial_fisher = flex.forecast().fisher.copy()

    flex.compute(parallel=True, n_proc=2)
    parallel_deriv = flex.deriv.copy()
    parallel_fisher = flex.forecast().fisher.copy()

    return flex, serial_deriv, parallel_deriv, serial_fisher, parallel_fisher


def test_serial_and_parallel_derivatives_are_bitwise_equal(computed):
    _, serial, parallel, _, _ = computed

    assert np.array_equal(serial, parallel), (
        f"max abs diff {np.max(np.abs(serial - parallel))}"
    )


def test_serial_and_parallel_fisher_matrices_are_bitwise_equal(computed):
    _, _, _, f_serial, f_parallel = computed

    assert np.array_equal(f_serial, f_parallel)


def test_serial_derivatives_are_finite(computed):
    _, serial, _, _, _ = computed
    assert np.all(np.isfinite(serial))


def test_pool_map_preserves_parameter_order(computed):
    """A stray ``imap_unordered`` would silently permute the Fisher matrix."""
    flex = computed[0]
    gc = flex.layout.probe_slice("gc")

    # The closed-form gbias derivative only lands on the right row if the
    # results came back in param_order.
    for b in range(flex.n_lens):
        i = flex.param_order.index(f"gbias{b + 1}")
        expected = 2.0 * flex.data_vector[gc.start + b] / flex.fid[f"gbias{b + 1}"]
        assert np.allclose(flex.deriv[i, gc.start + b], expected, rtol=1e-6)


def test_default_context_is_not_fork():
    """``fork`` deadlocks once CCL has run in the parent.

    Verified experimentally: ``fork`` with no prior CCL work completes in 0.3 s,
    but after ~20 ``ccl.angular_cl`` calls the children hang at 0% CPU forever,
    because CCL's GSL/OpenMP internals hold locks that a bare ``fork`` inherits
    in the locked state.  By the time derivatives are computed the parent has
    always used CCL, so ``fork`` must never be the default.
    """
    assert ff.DEFAULT_MP_CONTEXT != "fork"
    assert ff.DEFAULT_MP_CONTEXT in mp.get_all_start_methods()


def test_suppress_main_fixup_restores_module_state():
    """The context manager must leave ``__main__`` exactly as it found it.

    It exists because ``spawn``/``forkserver`` re-import ``__main__`` in the
    children, which fails outright when the parent was started from stdin or a
    notebook (``FileNotFoundError: '<stdin>'``).
    """
    import sys

    main = sys.modules["__main__"]
    had_file = hasattr(main, "__file__")
    before_file = getattr(main, "__file__", None)
    before_spec = getattr(main, "__spec__", None)

    with ff._suppress_main_fixup():
        assert not hasattr(main, "__file__")
        assert main.__spec__ is None

    assert hasattr(main, "__file__") == had_file
    assert getattr(main, "__file__", None) == before_file
    assert getattr(main, "__spec__", None) is before_spec


def test_default_n_proc_respects_the_affinity_mask(flex):
    """``os.cpu_count()`` over-reports on a shared node; affinity is the truth."""
    import os

    n = flex._default_n_proc()

    assert 1 <= n <= flex.n_params
    try:
        assert n <= len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not linux
        pass
