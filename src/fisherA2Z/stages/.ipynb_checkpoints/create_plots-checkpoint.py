#!/usr/bin/env python3
import json, yaml, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# === your Fisher stack (adjust import path/module name if needed)
from fisherA2Z.fisher import Fisher
from fisherA2Z.util import *

cosmo = ccl.Cosmology(Omega_c=0.2666, 
                       Omega_b=0.049, 
                       h=0.6727, 
                       sigma8=0.831, 
                       n_s=0.9645, 
                       transfer_function='eisenstein_hu')

# ---------- helpers ----------

def ellipse_points(center, cov2, nsig=2.0, n=360):
    """
    Return (x, y) arrays sampling the nsig contour of a 2D Gaussian with covariance cov2.
    """
    vals, vecs = np.linalg.eigh(cov2)
    # eigenvalues are variances along principal axes
    radii = nsig * np.sqrt(np.maximum(vals, 0))
    R = vecs  # columns are eigenvectors
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2, n)
    pts = (R @ (radii[:, None] * circle))
    pts[0, :] += center[0]
    pts[1, :] += center[1]
    return pts[0, :], pts[1, :]

def set_limits_from_content(ax, points_xy, pad_frac=0.12, min_span=1e-6):
    """
    points_xy: list of (xarray, yarray) or single (xarray, yarray).
    Sets xlim/ylim to tightly bound all points with a small fractional padding.
    """
    if isinstance(points_xy, tuple):
        points_xy = [points_xy]
    xs = np.concatenate([p[0] for p in points_xy]) if points_xy else np.array([])
    ys = np.concatenate([p[1] for p in points_xy]) if points_xy else np.array([])
    if xs.size == 0:
        return
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    # avoid zero span
    xr = max(xmax - xmin, min_span)
    yr = max(ymax - ymin, min_span)
    # padding
    xpad = xr * pad_frac
    ypad = yr * pad_frac
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

def gaussian_ellipse(ax, cov2, center, nsig=2, **kw):
    """
    Draw a nsig-sigma Gaussian ellipse from a 2x2 covariance matrix.
    Returns the Ellipse artist.
    """
    if cov2.shape != (2,2):
        raise ValueError("cov2 must be 2x2")
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # nsig scaling for a 2D Gaussian ellipse (sqrt of chi2 quantile ~ nsig)
    # For typical "2σ" look, scale by nsig
    width, height = 2 * nsig * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    e = Ellipse(xy=center, width=width, height=height, angle=angle, fill=False, **kw)
    ax.add_patch(e)
    return e

def two_by_two_cov_from_fisher(F, i, j):
    """
    Marginalized covariance for parameters (i,j) given Fisher F (invert full, then take submatrix).
    """
    C = np.linalg.inv(F)
    sub = C[[i,j]][:, [i,j]]
    return sub

def build_base_from_cfg(cfg, probe_label, ystage_key):
    """
    Recreate the base Fisher used in the forecast run so contours match.
    """
    bins = cfg.get("bins", [0,1,2,3,4])
    baseline_zvariance = cfg["baseline_zvariance"]
    prior_bias = cfg["prior_bias_by_1plusz"]
    prior_sig  = cfg["prior_sigma_by_1plusz"]
    save_deriv_template = cfg.get("save_deriv_template", "data/obj_deriv_{probe}_{y1}.pkl")
    overwrite_deriv = bool(cfg.get("overwrite_deriv", False))

    my_priors = {}
    for i in range(len(bins)):
        my_priors[f'zbias{i+1}'] = prior_bias[i]
        my_priors[f'zvariance{i+1}'] = prior_sig[i]
    if ystage_key=='y1':
        yboo = True
    else:
        yboo = False
    base = Fisher(
        cosmo,
        outliers=[0.0]*len(bins),
        zbias=[0.0]*len(bins),
        zvariance=baseline_zvariance[:len(bins)],
        save_deriv=save_deriv_template.format(probe=probe_label, y1=ystage_key),
        overwrite=overwrite_deriv,
        probe = probe_label,
        y1 = yboo
    )
    base.override_priors(my_priors)
    base.process()
    return base

def lcdm_index_subset(param_order, drop_names=("w_0","w_a")):
    """Return indices for a ΛCDM slice by dropping (w0, wa)."""
    return [k for k, name in enumerate(param_order) if name not in set(drop_names)]

def set_default_limits(ax, param_pair):
    """
    Apply your preferred axis windows when the pair matches (Ωm, σ8) or (w0, wa).
    """
    p0, p1 = param_pair
    if p0=="omega_m" and p1=="sigma_8":
        ax.set_xlim(0.285, 0.365)
        ax.set_ylim(0.75, 0.87)
    elif p0=="w_0" and p1=="w_a":
        ax.set_xlim(-1.15, -0.65)
        ax.set_ylim(-0.6, 0.5)

def set_default_limits_lcdm(ax, param_pair):
    if param_pair==("omega_m","sigma_8"):
        ax.set_xlim(0.292, 0.33)
        ax.set_ylim(0.81, 0.85)

# ---------- plotting for one scenario ----------

def plot_scenario(scen_dir: Path, cfg):
    """
    For a single scenario folder (e.g., forecast/bpz_equal_count_naive_stack),
    find all {probe}_{stage}_para_bias.json files and make the plots.
    """
    # find all bias files available
    pb_files = sorted(scen_dir.glob("*_para_bias.json"))
    lcdm_files = sorted(scen_dir.glob("*_para_bias_lcdm.json"))

    # Index by (probe, stage)
    def parse_key(p: Path):
        # expected name like "3x2pt_y1_para_bias.json"
        stem = p.stem
        parts = stem.split("_")
        # join all but last two tokens could be probe (handles probes with underscores)
        # but in our saving we used {probe}_{ystage_key}_para_bias.json
        probe = parts[0]
        stage = parts[1]
        return (probe, stage)

    pb_map = {parse_key(p): p for p in pb_files}
    lcdm_map = {parse_key(p): p for p in lcdm_files}

    # iterate common keys
    keys = sorted(set(pb_map.keys()) | set(lcdm_map.keys()))
    if not keys:
        print(f"[WARN] no bias files in {scen_dir}")
        return

    for (probe, stage) in keys:
        # Load biases (skip if missing)
        if (probe, stage) not in pb_map or (probe, stage) not in lcdm_map:
            print(f"[WARN] missing pair in {scen_dir.name}: {probe}, {stage}")
            continue
        para_bias = json.loads(pb_map[(probe, stage)].read_text())
        para_bias_lcdm = json.loads(lcdm_map[(probe, stage)].read_text())

        # Rebuild base Fisher to get matrix, params, labels, fiducials
        base = build_base_from_cfg(cfg, probe, stage)  # has .fisher, .param_order, .param_labels, .vals

        # ---- Figure 1: two panels (Ωm–σ8) and (w0–wa) ----
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        pairs = [("omega_m", "sigma_8"), ("w_0", "w_a")]
        widths = [0.001, 0.008]
        colors = plt.get_cmap('tab10').colors[:3]
        labels = ['Fiducial', 'Biased', 'Biased (LCDM)']

        for j, params in enumerate(pairs):
            p0, p1 = params
            i0 = base.param_order.index(p0)
            i1 = base.param_order.index(p1)

            # Use base Fisher for both contours (your earlier code did the same)
            cov2 = two_by_two_cov_from_fisher(base.fisher, i0, i1)

            # centers
            fid = (base.vals[p0], base.vals[p1])
            cen_biased   = (fid[0] + float(para_bias.get(p0,0.0)),   fid[1] + float(para_bias.get(p1,0.0)))
            cen_biased_l = (fid[0] + float(para_bias_lcdm.get(p0,0.0)), fid[1] + float(para_bias_lcdm.get(p1,0.0)))

            # markers
            ax[j].plot(*fid, 'X', color=colors[0], label=labels[0])
            ax[j].plot(*cen_biased, 'X', color=colors[1], label=labels[1])
            # arrow to "Biased" (full w0wa)
            ax[j].arrow(fid[0], fid[1],
                        cen_biased[0]-fid[0], cen_biased[1]-fid[1],
                        color='C1', width=0.0001, head_width=0.001)

            # contours (2σ)
            e0 = gaussian_ellipse(ax[j], cov2, center=fid,          nsig=2, lw=3.0, ec='black', fc='none', alpha=0.8)  # Fiducial
            e1 = gaussian_ellipse(ax[j], cov2, center=cen_biased,   nsig=2, lw=3.0, ec=colors[1], fc='none', alpha=0.8)  # Biased
            # e2 = gaussian_ellipse(ax[j], cov2, center=cen_biased_l, nsig=2, lw=3.0, ec=colors[2], fc='none', alpha=0.8)  # Biased (LCDM)

            ax[j].set_xlabel(base.param_labels[base.param_order.index(p0)], fontsize=22)
            ax[j].set_ylabel(base.param_labels[base.param_order.index(p1)], fontsize=22)
            # --- NEW: dynamic limits from everything we drew in this panel ---
            pts = []
            # ellipse point clouds
            x0, y0 = ellipse_points(fid,        cov2, nsig=2.0)
            x1, y1 = ellipse_points(cen_biased, cov2, nsig=2.0)
            pts += [(x0, y0), (x1, y1)]
            # fiducial + biased markers as tiny arrays to include them in bounds
            pts += [(np.array([fid[0]]), np.array([fid[1]])),
                    (np.array([cen_biased[0]]), np.array([cen_biased[1]]))]
            # If you include LCDM on this panel, also add its ellipse+point:
            # x2, y2 = ellipse_points(cen_biased_l, cov2, nsig=2.0)
            # pts += [(x2, y2), (np.array([cen_biased_l[0]]), np.array([cen_biased_l[1]]))]
            
            set_limits_from_content(ax[j], pts, pad_frac=0.12)
            
            ax[j].legend(handles=[e0, e1], labels=['Fiducial', 'Biased'], fontsize=12, loc="best")
            ax[j].grid(alpha=0.2)
        fig.suptitle(f"{scen_dir.name}  –  {probe.upper()}  {stage.upper()}", fontsize=16)
        fig.tight_layout()
        fig.savefig(scen_dir / f"{probe}_{stage}_w0wa_and_om_s8.png", dpi=180)
        plt.close(fig)

        # ---- Figure 2: ΛCDM (drop w0,wa), plot Ωm–σ8 with LCDM bias ----
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

        # slice Fisher to ΛCDM by dropping w0, wa
        keep = lcdm_index_subset(base.param_order, drop_names=("w_0","w_a"))
        F_lcdm = base.fisher[np.ix_(keep, keep)]
        # indices for the pair within the sliced matrix:
        # map param name -> index in sliced ordering
        name_by_keep = [base.param_order[i] for i in keep]
        i0 = name_by_keep.index("omega_m")
        i1 = name_by_keep.index("sigma_8")
        cov2_lcdm = two_by_two_cov_from_fisher(F_lcdm, i0, i1)

        fid = (base.vals["omega_m"], base.vals["sigma_8"])
        cen_l = (fid[0] + float(para_bias_lcdm.get("omega_m",0.0)),
                 fid[1] + float(para_bias_lcdm.get("sigma_8",0.0)))

        ax2.plot(*fid, 'X', color='k', label='Fiducial (LCDM)')
        ax2.plot(*cen_l, 'X', color='C1', label='Biased (LCDM)')
        ax2.arrow(fid[0], fid[1],
                  cen_l[0]-fid[0], cen_l[1]-fid[1],
                  color='C1', width=0.0001, head_width=0.001)

        e0 = gaussian_ellipse(ax2, cov2_lcdm, center=fid,  nsig=2, lw=3.0, ec='k',  fc='none', alpha=0.8)
        e1 = gaussian_ellipse(ax2, cov2_lcdm, center=cen_l, nsig=2, lw=3.0, ec='C1', fc='none', alpha=0.8)

        ax2.set_xlabel(base.param_labels[base.param_order.index("omega_m")], fontsize=18)
        ax2.set_ylabel(base.param_labels[base.param_order.index("sigma_8")], fontsize=18)
        set_default_limits_lcdm(ax2, ("omega_m","sigma_8"))
        pts = []
        x0, y0 = ellipse_points(fid,  cov2_lcdm, nsig=2.0)
        x1, y1 = ellipse_points(cen_l, cov2_lcdm, nsig=2.0)
        pts += [(x0, y0), (x1, y1),
                (np.array([fid[0]]),  np.array([fid[1]])),
                (np.array([cen_l[0]]), np.array([cen_l[1]]))]
        set_limits_from_content(ax2, pts, pad_frac=0.12)
        
        ax2.legend(handles=[e0, e1], labels=['Fiducial (LCDM)', 'Biased (LCDM)'], fontsize=10, loc="best")
        ax2.grid(alpha=0.2)
        ax2.legend(handles=[e0, e1], labels=['Fiducial (LCDM)', 'Biased (LCDM)'], fontsize=10, loc="best")
        ax2.grid(alpha=0.2)
        fig2.suptitle(f"{scen_dir.name}  –  {probe.upper()}  {stage.upper()} (ΛCDM)", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(scen_dir / f"{probe}_{stage}_lcdm_om_s8.png", dpi=180)
        plt.close(fig2)

        print(f"[PLOT] {scen_dir.name}: saved {probe}_{stage} figures")

def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    forecast_root = Path(cfg.get("forecast_dir", "forecast"))
    if not forecast_root.exists():
        print(f"[ERROR] forecast_dir not found: {forecast_root}")
        return

    # scenario folders = immediate subdirectories of forecast/
    scen_dirs = [p for p in forecast_root.iterdir() if p.is_dir()]
    if not scen_dirs:
        print(f"[WARN] no scenario subfolders in {forecast_root}")
        return

    for sd in sorted(scen_dirs):
        plot_scenario(sd, cfg)

# fishera2z/stages/prepare_fisher.py
def run(cfg_path: str | None = None):
    # If your script doesn’t need a config, ignore cfg_path
    # and call your existing main function.
    # Example: main() or main2() or whatever you named it
    from ._internal_prepare import main as _main  # if you split internal code
    _main()  # or _main(cfg_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_forecasts.py config.yml")
        sys.exit(1)
    main(sys.argv[1])