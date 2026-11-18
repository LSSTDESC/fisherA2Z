"""Lookup tables (one per prior scenario x probe x metric) and a PDF of all of them.

Usage:  python make_tables.py lookup_results.csv OUTDIR
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator

SCEN = {  # name -> caption label
    "baseline": r"prior $\times1$:  $\sigma(\Delta z_{\rm src})$ = 0.01, 0.01, 0.01, 0.02, 0.02;  "
                r"$\sigma(\Delta z_{\rm lens})$ = 0.01",
    "half":     r"prior $\times\frac{1}{2}$:  $\sigma(\Delta z_{\rm src})$ = 0.005, 0.005, 0.005, 0.01, 0.01;  "
                r"$\sigma(\Delta z_{\rm lens})$ = 0.005",
    "quarter":  r"prior $\times\frac{1}{4}$:  $\sigma(\Delta z_{\rm src})$ = 0.0025, 0.0025, 0.0025, 0.005, 0.005;  "
                r"$\sigma(\Delta z_{\rm lens})$ = 0.0025",
}
PROBE = {"cosmic_shear": "Cosmic shear", "3x2pt": "3x2pt"}
METRIC = {  # column -> (label, higher is better)
    "fom_om_s8": (r"FoM($\Omega_m$, $\sigma_8$)", True),
    "fom_w0_wa": (r"FoM($w_0$, $w_a$)", True),
    "sigma_s8":  (r"$\sigma(S_8)$", False),
}


def sig2(v):
    """Two significant digits, plain notation."""
    return f"{float(f'{v:.2g}'):g}"


def table(df, scen, probe, metric, ax_fig=None):
    d = df[(df.scenario == scen) & (df.probe == probe)]
    grid = d.pivot(index="i_lim", columns="area", values=metric).sort_index()   # shallow first
    areas, ilims = grid.columns.to_numpy(), grid.index.to_numpy()
    v = grid.to_numpy()
    label, higher_better = METRIC[metric]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    cmap = plt.cm.viridis if higher_better else plt.cm.viridis_r     # yellow = better for every metric
    norm = LogNorm(vmin=v.min(), vmax=v.max())
    im = ax.imshow(v, cmap=cmap, norm=norm, aspect="auto", origin="upper",
                   extent=(-0.5, len(areas) - 0.5, len(ilims) - 0.5, -0.5))
    for r in range(len(ilims)):
        for c in range(len(areas)):
            rgba = cmap(norm(v[r, c]))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(c, r, sig2(v[r, c]), ha="center", va="center", fontsize=6.6,
                    color="black" if lum > 0.5 else "white")

    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels([f"{a / 1000:g}k" for a in areas], fontsize=8)
    ax.set_xlabel(r"survey area  [deg$^2$]", fontsize=10)
    ax.set_yticks(range(len(ilims)))
    ax.set_yticklabels([f"{i:.1f}" for i in ilims], fontsize=8)
    ax.set_ylabel(r"$i_{\rm lim}$  (sample limiting magnitude)", fontsize=10)
    ax2 = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    ax2.set_yticks(range(len(ilims)))
    ax2.set_yticklabels([f"{i + 1:.1f}" for i in ilims], fontsize=8)
    ax2.set_ylabel(r"$i$-band coadd depth $m_5$  ($= i_{\rm lim} + 1$)", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, pad=0.09, fraction=0.035)
    cb.set_label(label, fontsize=10)
    decades = np.log10(v.max() / v.min())
    subs = np.arange(1, 10) if decades < 0.8 else (1, 2, 3, 5) if decades < 1.6 else (1, 2, 5)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    cb.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: sig2(x)))
    cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f"{PROBE[probe]}  ·  {label}\n{SCEN[scen]}", fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    return fig


def main():
    df = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    n = 0
    with PdfPages(os.path.join(out, "lsst_lookup_tables.pdf")) as pdf:
        for scen in SCEN:
            for probe in PROBE:
                for metric in METRIC:
                    fig = table(df, scen, probe, metric)
                    pdf.savefig(fig)
                    fig.savefig(os.path.join(out, "tables", f"{scen}_{probe}_{metric}.png"), dpi=150)
                    plt.close(fig)
                    n += 1
        info = pdf.infodict()
        info["Title"] = "LSST survey-strategy lookup tables"
    print(f"{n} tables -> {out}/lsst_lookup_tables.pdf and {out}/tables/")


if __name__ == "__main__":
    main()
