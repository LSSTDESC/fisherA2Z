"""Intrinsic-alignment helpers, checked against ``fisherA2Z.fisher``.

These are the pieces of A2Z that were *reimplemented* rather than imported, so
each one is pinned to the published behaviour.
"""

import numpy as np
import pytest

from fisherA2Z import fisher_flex as ff

pytestmark = pytest.mark.ccl


def test_get_phi_never_touches_self(tiny_cosmo):
    """``Fisher.get_phi`` is self-free; if that changes, this fails loudly.

    ``fisher_flex._get_phi`` is a transcription rather than a bound method, so a
    future refactor that starts reading ``self`` would silently desynchronize.
    """
    from fisherA2Z.fisher import Fisher

    z = np.linspace(0.1, 1.5, 8)
    # A2Z's signature has no defaults; these are the values its callers pass.
    L_ref, phi_ref = Fisher.get_phi(None, z, tiny_cosmo,
                                    -20.70, 1.23, -1.23, 0.0094, -0.3, 25.3)
    L, phi = ff._get_phi(z, tiny_cosmo)

    assert len(L) == len(L_ref) == z.size
    for i in range(z.size):
        assert np.allclose(L[i], L_ref[i], rtol=1e-14, atol=0.0)
        assert np.allclose(phi[i], phi_ref[i], rtol=1e-14, atol=0.0)


def test_ai_for_nz_matches_fisher_getai(tiny_cosmo):
    """The cached path must reproduce ``Fisher.getAi`` exactly.

    ``getAi`` ignores the z-grid of the ``dNdz`` it is handed and uses
    ``self.zmid_source`` instead, and it passes ``self.cosmo`` rather than its
    own ``cosmo`` argument.  The comparison is therefore only meaningful when
    both are pinned to the same values, which is what this sets up.
    """
    from fisherA2Z.fisher import Fisher

    z = np.linspace(0.05, 3.0, 120)
    nz = np.exp(-0.5 * ((z - 0.9) / 0.25) ** 2)
    beta = 1.0

    fisher = Fisher.__new__(Fisher)      # bypass the heavy __init__
    fisher.zmid_source = z
    fisher.cosmo = tiny_cosmo
    ref = fisher.getAi(beta, tiny_cosmo, nz)

    cache = ff._AiCache()
    ai_z = cache.ai_of_z(z, tiny_cosmo, "test", beta)
    got = ff._ai_for_nz(z, nz, ai_z)

    assert got == pytest.approx(float(np.atleast_1d(ref)[0]), rel=1e-12)


def test_ai_cache_hits(tiny_cosmo):
    z = np.linspace(0.05, 3.0, 60)
    cache = ff._AiCache()

    cache.ai_of_z(z, tiny_cosmo, "k0", 1.0)
    assert (cache.misses, cache.hits) == (1, 0)

    cache.ai_of_z(z, tiny_cosmo, "k0", 1.0)
    assert (cache.misses, cache.hits) == (1, 1)

    # A different beta reuses the cached phi but needs a new integral.
    cache.ai_of_z(z, tiny_cosmo, "k0", 1.2)
    assert (cache.misses, cache.hits) == (2, 1)
    assert len(cache._phi) == 1

    # A different cosmology must not be served from the cache.
    cache.ai_of_z(z, tiny_cosmo, "k1", 1.0)
    assert (cache.misses, cache.hits) == (3, 1)
    assert len(cache._phi) == 2


def test_ai_of_z_depends_on_beta(tiny_cosmo):
    z = np.linspace(0.05, 3.0, 40)
    cache = ff._AiCache()

    a1 = cache.ai_of_z(z, tiny_cosmo, "k", 1.0)
    a2 = cache.ai_of_z(z, tiny_cosmo, "k", 1.3)

    assert a1.shape == z.shape
    assert np.all(np.isfinite(a1))
    assert not np.allclose(a1, a2, rtol=1e-6, atol=0.0)


def test_vectorized_a_l_matches_the_scalar_loop():
    z = np.linspace(0.0, 3.0, 50)
    etal = 1.5
    scalar = np.array([((1 + zi) / 1.62) ** etal for zi in z])

    assert np.array_equal(ff._a_l(z, etal), scalar)


def test_vectorized_a_h_matches_the_scalar_loop():
    """A2Z's ``A_h`` is scalar-only (``if z > 0.75``); the branch must survive."""
    z = np.linspace(0.0, 3.0, 50)
    etah = 0.5
    scalar = np.array([((1 + zi) / 1.75) ** etah if zi > 0.75 else 1.0 for zi in z])

    assert np.array_equal(ff._a_h(z, etah), scalar)


def test_a_h_is_continuous_only_where_a2z_says_it_is():
    """Documenting a real discontinuity in the published model, not a bug here."""
    below = ff._a_h(np.array([0.7499]), 0.5)[0]
    above = ff._a_h(np.array([0.7501]), 0.5)[0]

    assert below == 1.0
    assert above == pytest.approx((1.75 / 1.75) ** 0.5, abs=1e-4)
    # At z = 0.75 the two branches happen to meet, so the kink is in the
    # derivative rather than the value.
    assert above == pytest.approx(below, abs=1e-4)
