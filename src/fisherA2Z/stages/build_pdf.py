#!/usr/bin/env python3
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _fmt_float(x):
    try:
        return f"{float(x): .4e}"
    except Exception:
        return str(x)


def _load_nz_bias(scen_dir: Path):
    p = scen_dir / "nz_bias_vectors.json"
    if not p.exists():
        return None
    nz = json.loads(p.read_text())
    z_bias = np.array(nz.get("z_bias", []))
    sigmaz_bias = np.array(nz.get("sigmaz_bias", []))
    bins = nz.get("bins", list(range(len(z_bias)))) if "bins" in nz else list(range(len(z_bias)))
    return dict(bins=bins, z_bias=z_bias, sigmaz_bias=sigmaz_bias)


def _find_probe_stage_pairs(scen_dir: Path, default_probes, stages):
    found = set()
    for p in scen_dir.glob("*_para_bias.json"):
        parts = p.stem.split("_")  # {probe}_{stage}_para_bias
        if len(parts) >= 3:
            found.add((parts[0], parts[1]))
    if not found:
        found = {(pr, st) for pr in default_probes for st in stages}
    return sorted(found)


def _load_para_bias_maps(scen_dir: Path, probe_stage_pairs):
    """Return (full_bias_map, lcdm_bias_map, union_params_full, union_params_lcdm).
    Each *_bias_map is: { (probe,stage) : {param: value, ...}, ... }
    """
    full_map, lcdm_map = {}, {}
    params_full, params_lcdm = set(), set()

    for (probe, stage) in probe_stage_pairs:
        pb_file = scen_dir / f"{probe}_{stage}_para_bias.json"
        lcdm_file = scen_dir / f"{probe}_{stage}_para_bias_lcdm.json"
        if pb_file.exists():
            d = json.loads(pb_file.read_text())
            full_map[(probe, stage)] = d
            params_full.update(d.keys())
        if lcdm_file.exists():
            d = json.loads(lcdm_file.read_text())
            lcdm_map[(probe, stage)] = d
            params_lcdm.update(d.keys())

    return full_map, lcdm_map, sorted(params_full), sorted(params_lcdm)

def _wrap_text_page(pdf, title, lines, lines_per_page=55, fontsize=9):
    """
    Add one or more text pages to the PDF using a monospace font.
    lines: list[str] (already formatted, fixed-width preferred)
    """
    if not lines:
        return False
    # chunk into pages
    for i in range(0, len(lines), lines_per_page):
        chunk = lines[i:i+lines_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))  # landscape-ish letter
        ax.axis("off")
        # Title
        fig.suptitle(title, fontsize=12, y=0.98)
        # Draw the text block
        text = "\n".join(chunk)
        ax.text(
            0.02, 0.97, text,
            transform=ax.transAxes,
            va="top", ha="left",
            family="monospace",
            fontsize=fontsize,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)
    return True


def _fmt_float(x):
    try:
        return f"{float(x): .4e}"
    except Exception:
        return str(x)


def _make_param_table_page(pdf, scen_dir: Path, nz_bias, full_map, lcdm_map,
                           params_full, params_lcdm, probe_stage_pairs):
    """
    Text-only version that ALSO writes the same text into the PDF (monospace page[s]).

    Prints (stdout) and adds to PDF:
      - n(z) bias by tomographic bin
      - Parameter bias (w0–wa): one parameter per row, columns=probe-stage
      - Parameter bias (ΛCDM): one parameter per row, columns=probe-stage

    Returns True if any content was produced (printed & added to PDF), else False.
    """
    has_nz    = nz_bias is not None and len(nz_bias.get("z_bias", [])) > 0
    has_full  = bool(full_map) and len(params_full) > 0
    has_lcdm  = bool(lcdm_map) and len(params_lcdm) > 0

    if not (has_nz or has_full or has_lcdm):
        return False

    lines = []  # accumulate fixed-width lines for stdout + PDF

    def section_header(title):
        bar = "-" * len(title)
        return [title, bar]

    # 1) n(z) biases (if any)
    if has_nz:
        title = f"Scenario: {scen_dir.name} — n(z) Bias by Tomographic Bin"
        lines += section_header(title)

        bins        = nz_bias.get("bins", list(range(len(nz_bias["z_bias"]))))
        z_bias      = nz_bias["z_bias"]
        sigmaz_bias = nz_bias["sigmaz_bias"]

        # Column widths
        hdr0 = "bin"
        hdr1 = "z_bias (⟨z⟩ est − true)"
        hdr2 = "sigmaz_bias (σz/(1+z) est − true)"

        col0 = max(len(hdr0), max(len(str(b)) for b in bins)) + 2
        col1 = max(len(hdr1), max(len(_fmt_float(v)) for v in z_bias)) + 2
        col2 = max(len(hdr2), max(len(_fmt_float(v)) for v in sigmaz_bias)) + 2

        # Header row
        lines.append(
            f"{hdr0.ljust(col0)}{hdr1.ljust(col1)}{hdr2.ljust(col2)}"
        )
        # Rows
        for i, b in enumerate(bins):
            lines.append(
                f"{str(b).ljust(col0)}"
                f"{_fmt_float(z_bias[i]).ljust(col1)}"
                f"{_fmt_float(sigmaz_bias[i]).ljust(col2)}"
            )
        lines.append("")  # blank line separator

    # helper to format “one parameter per row” table
    def build_param_table_text(title, params, bias_map):
        if not params or not bias_map:
            return []
        # keep only pairs with data
        cols = [(p, s) for (p, s) in probe_stage_pairs if (p, s) in bias_map]
        if not cols:
            return []
        header = f"Scenario: {scen_dir.name} — {title}"
        out = section_header(header)

        # Figure column widths
        col0_hdr = "Parameter"
        col0 = max(len(col0_hdr), max(len(par) for par in params)) + 2

        col_headers = [f"{p.upper()}-{s.upper()}" for (p, s) in cols]
        col_w = []
        for (p, s), hdr in zip(cols, col_headers):
            w = len(hdr)
            for par in params:
                val = _fmt_float(bias_map[(p, s)].get(par, ""))
                w = max(w, len(val))
            col_w.append(w + 2)

        # Header row
        out.append(col0_hdr.ljust(col0) + "".join(h.ljust(w) for h, w in zip(col_headers, col_w)))
        # Rows
        for par in params:
            row = par.ljust(col0)
            for (p, s), w in zip(cols, col_w):
                row += _fmt_float(bias_map[(p, s)].get(par, "")).ljust(w)
            out.append(row)
        out.append("")  # blank line
        return out

    # 2) w0–wa table
    if has_full:
        lines += build_param_table_text("Cosmology Parameter Bias (w0–wa model)", params_full, full_map)

    # 3) ΛCDM table
    if has_lcdm:
        lines += build_param_table_text("Cosmology Parameter Bias (ΛCDM)", params_lcdm, lcdm_map)

    # Nothing accumulated? bail
    if not lines:
        return False

    # --- Print to stdout ---
    print("\n".join(lines))

    # --- Also add to the PDF as text pages (auto-paginated) ---
    _wrap_text_page(
        pdf=pdf,
        title=f"Scenario Text: {scen_dir.name}",
        lines=lines,
        lines_per_page=55,  # tweak if you want denser/sparser pages
        fontsize=9
    )
    return True

def _make_plots_page(pdf, scen_dir: Path, probe_stage_pairs):
    """
    Create a page of images for pairs that actually have both figures.
    Returns True if at least one row was plotted, else False (caller can skip page).
    """
    rows = []
    for (probe, stage) in probe_stage_pairs:
        left = scen_dir / f"{probe}_{stage}_w0wa_and_om_s8.png"
        right = scen_dir / f"{probe}_{stage}_lcdm_om_s8.png"
        if left.exists() or right.exists():
            rows.append((probe, stage, left if left.exists() else None, right if right.exists() else None))

    if not rows:
        return False

    fig_h = 4.5 * len(rows)
    fig, axes = plt.subplots(nrows=len(rows), ncols=2, figsize=(14, fig_h))
    if len(rows) == 1:
        axes = np.array([axes])

    fig.suptitle(f"Scenario Plots: {scen_dir.name}", fontsize=16, y=0.99)

    for r, (probe, stage, left, right) in enumerate(rows):
        axL, axR = axes[r, 0], axes[r, 1]
        # left
        axL.axis("off")
        if left is not None:
            img = plt.imread(left); axL.imshow(img)
            axL.set_title(f"{probe.upper()} {stage.upper()} — w0–wa & Ωm–σ8", fontsize=12)
        # right
        axR.axis("off")
        if right is not None:
            img = plt.imread(right); axR.imshow(img)
            axR.set_title(f"{probe.upper()} {stage.upper()} — ΛCDM (Ωm–σ8)", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)
    return True


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    forecast_root = Path(cfg.get("forecast_dir", "forecast"))
    if not forecast_root.exists():
        raise SystemExit(f"forecast_dir not found: {forecast_root}")

    scen_dirs = sorted([p for p in forecast_root.iterdir() if p.is_dir()])
    if not scen_dirs:
        raise SystemExit(f"No scenario subfolders in {forecast_root}")

    default_probes = cfg.get("fisher_probes", ["3x2pt", "ss"])
    stages = []
    if cfg.get("run_y1", True):  stages.append("y1")
    if cfg.get("run_y10", True): stages.append("y10")

    out_pdf = forecast_root / "forecast_report.pdf"
    with PdfPages(out_pdf) as pdf:
        for scen_dir in scen_dirs:
            # discover probe/stage
            print(scen_dir)
            pairs = _find_probe_stage_pairs(scen_dir, default_probes, stages)
            # load data
            nz_bias = _load_nz_bias(scen_dir)
            full_map, lcdm_map, params_full, params_lcdm = _load_para_bias_maps(scen_dir, pairs)

            # Build the parameter table page (skip if nothing)
            made_table = _make_param_table_page(
                pdf, scen_dir, nz_bias, full_map, lcdm_map, params_full, params_lcdm, pairs
            )

            # Build the plots page (skip if no images)
            made_plots = _make_plots_page(pdf, scen_dir, pairs)

            # If neither page was created, skip the scenario silently
            if not (made_table or made_plots):
                continue

    print(f"[REPORT] Wrote {out_pdf}")

from typing import Optional

def run(cfg_path: Optional[str] = None):
    if cfg_path is None:
        raise ValueError("build_pdf.run requires a config path")
    for fn_name in ("main", "main2", "main3", "run"):
        fn = globals().get(fn_name, None)
        if callable(fn):
            return fn(cfg_path)
    raise RuntimeError("build_pdf.py: no main/main2/main3/run(cfg) found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python build_report_pdf.py config.yml")
        raise SystemExit(2)
    main(sys.argv[1])
