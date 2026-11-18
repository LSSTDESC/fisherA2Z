"""Survey-strategy lookup table: area x depth x photo-z prior -> FoM and sigma(S8).

Same infrastructure as ../wfd_footprint/wfd_yearly_forecast.ipynb:
Cardinal 1yr source n(z) shape, per-bin n_eff rescaled with depth from counts in the
Cardinal gold mock, FisherFlex Y1 lens preset with its n_eff rescaled the same way,
sigma_e = 0.26, 300 < ell < 1800 for shear.

Grid
----
area   1000 ... 18000 deg^2, step 1000                       (18)
i_lim  23.5 ... 25.5, step 0.1  (sample limiting magnitude;   (21)
       m5 = i_lim + 1)
prior  sigma(dz_src) = (0.01, 0.01, 0.01, 0.02, 0.02) * k     (3: k = 1, 1/2, 1/4)
       sigma(dz_lens) = 0.01 * k for every lens bin
       source stretch prior: the Cardinal-realization value, the same in every scenario

At fixed n_eff the covariance is exactly proportional to 1/f_sky and the derivatives
do not depend on area, so each depth is computed once per probe and the 18 areas are
obtained by rescaling the covariance (checked against full recomputations below).
The prior enters only in forecast(), so the 3 scenarios cost nothing extra.

Usage:  python run_lookup.py OUT.csv [--quick]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import healpy as hp
import pyarrow.parquet as pq
import qp
import tables_io

from fisherA2Z.fisher_flex import FisherFlex

HERE = os.path.dirname(os.path.abspath(__file__))
NZDC = os.path.join(HERE, "..", "nzdc_example")
MOCK = os.path.join(HERE, "..", "data", "25", "part-0.pq")

FULL_SKY_DEG2 = 4 * np.pi * (180.0 / np.pi) ** 2
AREAS = np.arange(1000.0, 18000.1, 1000.0)
I_LIMS = np.round(np.arange(23.5, 25.5001, 0.1), 2)
M5_TO_ILIM = 1.0
AREA_REF = 10000.0               # area each depth is computed at, then rescaled

M_SRC_FID = 24.1                 # depth the Cardinal 1yr n_eff corresponds to
NEFF_LENS_TOTAL = 18.0
LENS_ZMAX = 1.2
Z_HALFWIDTH = 0.05
SIGMA_E = 0.26
ELL_CUTS = dict(ell_min_cs=300, ell_max_cs=1800)

SRC_DZ_BASE = np.array([0.01, 0.01, 0.01, 0.02, 0.02])
LENS_DZ_BASE = 0.01
SCENARIOS = {"baseline": 1.0, "half": 0.5, "quarter": 0.25}
PROBES = ("cosmic_shear", "3x2pt")


# ----------------------------------------------------------------- inputs
def load_inputs():
    z_grid = np.linspace(0.01, 3.0, 150)
    qp_nz = qp.read(f"{NZDC}/nz_challenge_taskset_1_cardinal_nz_samples_1yr_wfd.hdf5")
    nz_realization = np.swapaxes(qp_nz.pdf(z_grid).reshape(5, 10, 150), 0, 1)
    nz_central = np.mean(qp_nz.pdf(z_grid).reshape(5, 10, 150), axis=1)
    nz_true = qp.read(f"{NZDC}/nz_challenge_taskset_1_cardinal_1yr_nz_true_wfd.hdf5").pdf(z_grid)
    bhat = tables_io.read(f"{NZDC}/nz_challenge_taskset_1_cardinal_1yr_bhat_wfd.hdf5")
    counts = np.bincount(bhat["bhat_for_wide_data"], minlength=5)[:5]
    neff_src_fid = counts * (3505864 / (107.4295 * 3600)) / 1000000

    cardinal_cov = FisherFlex(
        nz_source=nz_central, nz_realizations=nz_realization, z_grid=z_grid,
        neff_source=neff_src_fid, fsky=0.4, sigma_e=SIGMA_E,
        mode="cosmic_shear", nz_model="shift_stretch", y1=True, verbose=False,
    ).pz_prior_cov
    stretch_var = cardinal_cov[:, 1, 1].copy()

    lens_ref = FisherFlex(
        nz_source=nz_true, photoz_prior_cov=cardinal_cov, z_grid=z_grid,
        neff_source=neff_src_fid, fsky=0.4, sigma_e=SIGMA_E,
        mode="3x2pt", nz_model="shift_stretch", y1=True, verbose=False,
    )
    z_src = np.trapezoid(nz_true * z_grid, z_grid, axis=1) / np.trapezoid(nz_true, z_grid, axis=1)

    t = pq.read_table(MOCK, columns=["healpix", "mag_i_lsst", "redshift"])
    keep = np.isin(t["healpix"].to_numpy(), (10177, 10305, 10429))
    mag = t["mag_i_lsst"].to_numpy().astype(float)[keep]
    zm = t["redshift"].to_numpy().astype(float)[keep]
    o = np.argsort(zm)
    mag, zm = mag[o], zm[o]
    area_min2 = 3 * hp.nside2pixarea(32, degrees=True) * 3600
    lens_sorted = np.sort(mag[(zm > 0) & (zm < LENS_ZMAX)])
    grid = np.arange(23.0, 25.5, 0.001)
    m_lens_fid = float(grid[np.argmin(np.abs(np.searchsorted(lens_sorted, grid) / area_min2
                                             - NEFF_LENS_TOTAL))])

    return dict(z_grid=z_grid, nz_true=nz_true, neff_src_fid=neff_src_fid,
                stretch_var=stretch_var, neff_lens_fid=lens_ref.neff_lens.copy(),
                z_lens=lens_ref.z_eff_lens.copy(), z_src=z_src, mag=mag, zm=zm,
                m_lens_fid=m_lens_fid, mag_max=float(mag.max()))


def local_factor(inp, z_centre, m_fid, delta_m):
    lo, hi = np.searchsorted(inp["zm"], [z_centre - Z_HALFWIDTH, z_centre + Z_HALFWIDTH])
    m = np.sort(inp["mag"][lo:hi])
    return np.searchsorted(m, m_fid + delta_m) / np.searchsorted(m, m_fid)


def neff_at(inp, i_lim):
    dm = i_lim - M_SRC_FID
    f_s = np.array([local_factor(inp, z, M_SRC_FID, dm) for z in inp["z_src"]])
    f_l = np.array([local_factor(inp, z, inp["m_lens_fid"], dm) for z in inp["z_lens"]])
    return inp["neff_src_fid"] * f_s, inp["neff_lens_fid"] * f_l


# ----------------------------------------------------------------- forecasts
def build(inp, mode, area, ns, nl):
    cov0 = np.array([np.diag([SRC_DZ_BASE[b] ** 2, inp["stretch_var"][b]]) for b in range(5)])
    return FisherFlex(
        nz_source=inp["nz_true"], photoz_prior_cov=cov0, z_grid=inp["z_grid"],
        neff_source=ns, lens_neff=nl, fsky=area / FULL_SKY_DEG2, sigma_e=SIGMA_E,
        mode=mode, nz_model="shift_stretch", y1=True,
        lens_dz_prior=LENS_DZ_BASE if mode == "3x2pt" else None, verbose=False,
    )


def set_scenario(flex, inp, k):
    """Photo-z priors of scenario k (scale factor on every shift width)."""
    sd = SRC_DZ_BASE * k
    flex.pz_prior_cov = np.array([np.diag([sd[b] ** 2, inp["stretch_var"][b]]) for b in range(5)])
    for b in range(5):
        flex.prior_sigma[f"zbias{b + 1}"] = float(sd[b])
        flex.prior_sigma[f"zstretch{b + 1}"] = float(np.sqrt(inp["stretch_var"][b]))
    if flex.lens_dz_prior is not None:
        for b in range(flex.n_lens):
            flex.prior_sigma[f"lzbias{b + 1}"] = LENS_DZ_BASE * k


def metrics_at(flex, area):
    saved = flex.cov_blocks
    try:
        flex.cov_blocks = saved * (flex.fsky / (area / FULL_SKY_DEG2))
        res = flex.forecast(**ELL_CUTS)
    finally:
        flex.cov_blocks = saved
    return {"fom_om_s8": res.fom("omega_m", "sigma_8"),
            "fom_w0_wa": res.fom("w_0", "w_a"),
            "sigma_s8": res.s8()[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--quick", action="store_true", help="2 depths only (test)")
    args = ap.parse_args()
    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    t0 = time.time()
    inp = load_inputs()
    print(f"inputs loaded ({time.time() - t0:.0f} s); mock complete to i < {inp['mag_max']:.3f}, "
          f"lens reference depth i < {inp['m_lens_fid']:.3f}; n_proc={n_proc}", flush=True)
    if I_LIMS.max() > inp["mag_max"] + 1e-6:
        sys.exit("grid deeper than the mock")

    i_lims = I_LIMS[[0, -1]] if args.quick else I_LIMS
    rows = []
    for i_lim in i_lims:
        ns, nl = neff_at(inp, i_lim)
        for mode in PROBES:
            t1 = time.time()
            flex = build(inp, mode, AREA_REF, ns, nl).compute(parallel=True, n_proc=n_proc)
            for scen, k in SCENARIOS.items():
                set_scenario(flex, inp, k)
                for area in AREAS:
                    rows.append({"probe": mode, "scenario": scen, "prior_scale": k,
                                 "area": area, "i_lim": i_lim, "m5": i_lim + M5_TO_ILIM,
                                 "neff_src": ns.sum(), "neff_lens": nl.sum(),
                                 **metrics_at(flex, area)})
            print(f"i_lim={i_lim:.1f} {mode:>12}: compute {time.time() - t1:5.1f} s", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)       # checkpoint after each depth

    # ---- check the area rescaling against full recomputations
    print("\nrescaling check (full compute at the target area):", flush=True)
    df = pd.DataFrame(rows)
    worst = 0.0
    for i_lim, area in ((i_lims[0], 1000.0), (i_lims[-1], 18000.0)):
        ns, nl = neff_at(inp, i_lim)
        for mode in PROBES:
            flex = build(inp, mode, area, ns, nl).compute(parallel=True, n_proc=n_proc)
            for scen, k in SCENARIOS.items():
                set_scenario(flex, inp, k)
                res = flex.forecast(**ELL_CUTS)
                direct = {"fom_om_s8": res.fom("omega_m", "sigma_8"),
                          "fom_w0_wa": res.fom("w_0", "w_a"), "sigma_s8": res.s8()[1]}
                row = df[(df.probe == mode) & (df.scenario == scen) & np.isclose(df.area, area)
                         & np.isclose(df.i_lim, i_lim)].iloc[0]
                for key, v in direct.items():
                    worst = max(worst, abs(row[key] / v - 1))
    print(f"  max relative difference: {worst:.2e}", flush=True)
    if worst > 1e-8:
        sys.exit("area rescaling does not reproduce the full computation")
    print(f"done: {len(df)} rows in {time.time() - t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
