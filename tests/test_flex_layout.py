"""Tests for the block layout, GGL pair selection and scale-cut masks."""

import numpy as np
import pytest

from fisherA2Z import flex_layout as fl


def test_block_order_is_cs_ggl_gc(toy_layout):
    """The order matches ``Fisher.makeFidCells`` so derivatives stay comparable."""
    probes = toy_layout.probe.tolist()
    assert probes == ["cs"] * 3 + ["ggl"] * 6 + ["gc"] * 3
    assert toy_layout.n_blocks == 12
    assert toy_layout.n_legs == 5


def test_probe_slices_are_contiguous_and_partition_the_blocks(toy_layout):
    slices = toy_layout.probe_block_slices()

    covered = np.zeros(toy_layout.n_blocks, dtype=int)
    for probe, sl in slices.items():
        assert np.all(toy_layout.probe[sl] == probe)
        covered[sl] += 1
    assert np.all(covered == 1)


def test_legs_are_lens_then_source(toy_layout):
    cs = toy_layout.probe_slice("cs")
    gc = toy_layout.probe_slice("gc")

    assert np.all(toy_layout.leg_a[cs] >= toy_layout.n_lens)
    assert np.all(toy_layout.leg_a[gc] < toy_layout.n_lens)
    # GGL always pairs a lens leg with a source leg, in that order.
    ggl = toy_layout.probe_slice("ggl")
    assert np.all(toy_layout.leg_a[ggl] < toy_layout.n_lens)
    assert np.all(toy_layout.leg_b[ggl] >= toy_layout.n_lens)


def test_cosmic_shear_blocks_are_the_upper_triangle(toy_layout):
    assert toy_layout.pairs("cs") == [(0, 0), (0, 1), (1, 1)]


def test_clustering_blocks_are_autocorrelations_only(toy_layout):
    assert toy_layout.pairs("gc") == [(0, 0), (1, 1), (2, 2)]
    gc = toy_layout.probe_slice("gc")
    assert np.array_equal(toy_layout.leg_a[gc], toy_layout.leg_b[gc])


def test_ggl_pair_selection_matches_a_hand_computed_set():
    z_lens = np.array([0.25, 0.55, 0.95])
    z_source = np.array([0.4, 0.9, 1.5])
    # source > lens + 0.3:
    #   lens 0 (0.25 -> 0.55): sources 0.9, 1.5
    #   lens 1 (0.55 -> 0.85): sources 0.9, 1.5
    #   lens 2 (0.95 -> 1.25): source 1.5
    assert fl.select_ggl_pairs(z_lens, z_source) == [(0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_ggl_offset_is_honoured():
    z_lens = np.array([0.5])
    z_source = np.array([0.6, 1.0])

    assert fl.select_ggl_pairs(z_lens, z_source, offset=0.3) == [(0, 1)]
    assert fl.select_ggl_pairs(z_lens, z_source, offset=0.05) == [(0, 0), (0, 1)]
    assert fl.select_ggl_pairs(z_lens, z_source, offset=2.0) == []


def test_mode_restricts_the_probes():
    pairs = [(l, s) for l in range(3) for s in range(2)]

    full = fl.build_layout(3, 2, pairs, mode="3x2pt")
    two = fl.build_layout(3, 2, pairs, mode="2x2pt")
    shear = fl.build_layout(3, 2, pairs, mode="cosmic_shear")

    assert set(full.probe) == {"cs", "ggl", "gc"}
    assert set(two.probe) == {"ggl", "gc"}
    assert set(shear.probe) == {"cs"}
    assert shear.n_blocks == 3

    with pytest.raises(ValueError, match="unknown mode"):
        fl.build_layout(3, 2, pairs, mode="1x2pt")


def test_a2z_accept_sets_have_the_documented_sizes():
    assert len(fl.A2Z_GGL_ACCEPT_Y10) == 25
    assert len(fl.A2Z_GGL_ACCEPT_Y1) == 7


def test_out_of_range_ggl_pair_raises():
    with pytest.raises(ValueError, match="out of range"):
        fl.build_layout(3, 2, [(0, 5)], mode="3x2pt")


# --------------------------------------------------------------------------
# ell grid
# --------------------------------------------------------------------------


def test_default_ell_grid_matches_the_a2z_values():
    edges = fl.default_ell_edges()
    ell = fl.ell_from_edges(edges)

    assert edges.size == 21 and ell.size == 20
    assert ell[0] == pytest.approx(np.sqrt(edges[0] * edges[1]))
    # A2Z's data/ell-values.txt spans this range.
    assert ell[0] == pytest.approx(23.0, abs=1.0)
    assert ell[-1] == pytest.approx(13000.0, rel=0.05)


def test_ell_cut_3000_keeps_15_of_20_bins():
    ell = fl.ell_from_edges(fl.default_ell_edges())
    assert int(np.sum(ell <= 3000)) == 15


def test_n_modes_uses_exact_bin_widths():
    edges = fl.default_ell_edges()
    ell = fl.ell_from_edges(edges)
    fsky = 0.4

    nm = fl.n_modes(ell, edges, fsky)
    expected = (2 * ell + 1) * np.diff(edges) * fsky
    assert np.allclose(nm, expected)


def test_n_modes_edges_vs_gradient_differ_by_the_documented_amount():
    """The rail reference uses ``ell * grad(log ell)``; it is not the same thing."""
    edges = fl.default_ell_edges()
    ell = fl.ell_from_edges(edges)

    exact = np.diff(edges)
    approx = ell * np.gradient(np.log(ell))
    rel = np.abs(approx - exact) / exact
    assert np.median(rel) == pytest.approx(0.0045, abs=0.002)


# --------------------------------------------------------------------------
# scale cuts
# --------------------------------------------------------------------------


def test_ell_max_kcut_keeps_the_factor_of_h():
    """Hard-coded guard: dropping the ``h`` inflates ell_max by 1/h ~ 1.49."""
    assert fl.ell_max_kcut(1000.0, 0.6727, k_max=0.3) == pytest.approx(0.3 * 672.7 - 0.5)
    # The two sanity values quoted in the plan.
    assert fl.ell_max_kcut(1000.0, 0.6727) == pytest.approx(201.31, abs=0.01)


def test_ell_max_kcut_is_vectorized():
    chi = np.array([1000.0, 2000.0])
    out = fl.ell_max_kcut(chi, 0.6727)
    assert out.shape == (2,)
    assert out[1] == pytest.approx(2 * (out[0] + 0.5) - 0.5)


def test_mask_defaults_to_everything(toy_layout, toy_ell):
    mask = fl.build_mask(toy_layout, toy_ell)

    assert mask.shape == (toy_layout.n_blocks, toy_ell.size)
    assert mask.dtype == bool
    assert mask.all()


def test_mask_applies_per_probe_ell_cuts(toy_layout, toy_ell):
    mask = fl.build_mask(toy_layout, toy_ell, ell_cuts={"cs": (None, 1000.0)})

    cs = toy_layout.probe_slice("cs")
    assert np.array_equal(mask[cs][0], toy_ell <= 1000.0)
    # Other probes untouched.
    assert mask[toy_layout.probe_slice("gc")].all()


def test_mask_accepts_a_per_lens_bin_ell_max(toy_layout, toy_ell):
    """This is how the k_max cut is expressed with no special-case code."""
    lmax = np.array([200.0, 800.0, 3000.0])
    mask = fl.build_mask(toy_layout, toy_ell, ell_cuts={"gc": (None, lmax)})

    gc = toy_layout.probe_slice("gc")
    for b, lens_bin in enumerate(range(3)):
        assert np.array_equal(mask[gc][b], toy_ell <= lmax[lens_bin])


def test_per_lens_bin_cut_on_cosmic_shear_raises(toy_layout, toy_ell):
    with pytest.raises(ValueError, match="no lens bin"):
        fl.build_mask(toy_layout, toy_ell,
                      ell_cuts={"cs": (None, np.array([100.0, 200.0, 300.0]))})


def test_mask_block_select_restricts_bin_pairs(toy_layout, toy_ell):
    mask = fl.build_mask(toy_layout, toy_ell, block_select={"cs": [(0, 0)]})

    cs = toy_layout.probe_slice("cs")
    assert mask[cs][0].all()          # (0, 0) kept
    assert not mask[cs][1].any()      # (0, 1) dropped
    assert not mask[cs][2].any()      # (1, 1) dropped


def test_mask_probes_filter_removes_whole_probes(toy_layout, toy_ell):
    mask = fl.build_mask(toy_layout, toy_ell, probes=["cs"])

    assert mask[toy_layout.probe_slice("cs")].all()
    assert not mask[toy_layout.probe_slice("ggl")].any()
    assert not mask[toy_layout.probe_slice("gc")].any()


def test_lens_bin_of_block_is_minus_one_for_shear(toy_layout):
    lens_bin = toy_layout.lens_bin_of_block()

    assert np.all(lens_bin[toy_layout.probe_slice("cs")] == -1)
    assert np.all(lens_bin[toy_layout.probe_slice("gc")] == np.arange(3))
