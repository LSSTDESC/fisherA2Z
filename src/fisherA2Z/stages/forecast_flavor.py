#!/usr/bin/env python3
import os, json, glob, warnings
from pathlib import Path
import numpy as np
import yaml

# --- deps you said you have in your environment
# qp must be able to read the HDF5 (Ensemble)
import qp
# your Fisher & helpers:
# from fisher_module import Fisher, centroid_shift, centroid_shift_lcdm, Wrapper, cosmo
# (Assume these are importable in your PYTHONPATH)
from fisherA2Z.fisher import Fisher  # adjust import as needed
from fisherA2Z.util import *

cosmo = ccl.Cosmology(Omega_c=0.2666, 
                       Omega_b=0.049, 
                       h=0.6727, 
                       sigma8=0.831, 
                       n_s=0.9645, 
                       transfer_function='eisenstein_hu')


def get_z_sigmaz_mean_sigma(ens):
    """
    Compute ensemble mean z and sigma_z/(1+z) stats from a qp Ensemble with .pdf(z).
    Returns dict with zmean_val, zmean_sigma, zsigma_val, zsigma_sigma .
    """
    z = np.linspace(0, 3, 301)  # adjust if needed
    pdf_vals = ens.pdf(z)       # shape: (n_pdfs, n_z)
    dz = z[1] - z[0]
    # normalize over grid
    norm = np.sum(pdf_vals * dz, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    pdf_norm = pdf_vals / norm
    means = np.sum(pdf_norm * z, axis=1) * dz
    variances = np.sum(pdf_norm * (z - means[:, None])**2, axis=1) * dz
    sigmas = np.sqrt(variances) / (1.0 + means)
    return {
        "zmean_val": float(np.mean(means)),
        "zmean_sigma": float(np.std(means)),
        "zsigma_val": float(np.mean(sigmas)),
        "zsigma_sigma": float(np.std(sigmas)),
    }


def _read_qp_ensemble(h5_path):
    """
    Try reading a qp Ensemble from an HDF5 file path.
    """
    try:
        return qp.read(str(h5_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read qp Ensemble from {h5_path}: {e}")


def _find_estimated_single_nz(root_dir, algo, binning, b, stacking):
    """
    Your directory has multiple prefixes. We try common patterns. Returns a single path or None.
    """
    patterns = [
        f"single_NZ_summarize_{algo}_{binning}_bin{b}_{stacking}.hdf5",
        f"output_summarize_{algo}_{binning}_bin{b}_{stacking}.hdf5",
    ]
    for pat in patterns:
        cand = sorted(Path(root_dir).glob(pat))
        if cand:
            return cand[0]
    # last resort: wide search by suffix
    wide = sorted(Path(root_dir).glob(f"*{algo}_{binning}_bin{b}_{stacking}.hdf5"))
    return wide[0] if wide else None


def _find_truth_nz(root_dir, algo, binning, b):
    """
    True files follow: true_NZ_true_nz_{algo}_{binning}_bin{b}.hdf5
    """
    pat = f"true_NZ_true_nz_{algo}_{binning}_bin{b}.hdf5"
    cand = sorted(Path(root_dir).glob(pat))
    if cand:
        return cand[0]
    # fallback: try any 'true_nz' with same algo/binning/bin
    wide = sorted(Path(root_dir).glob(f"*true_nz*{algo}_{binning}_bin{b}.hdf5"))
    return wide[0] if wide else None


def compute_bias_vectors(root_dir, algo, binning, stacking, bins):
    """
    For a scenario (algo, binning, stacking), compute z_bias and sigmaz_bias arrays (len=bins).
    """
    z_bias = []
    sigmaz_bias = []
    for b in bins:
        est_file = _find_estimated_single_nz(root_dir, algo, binning, b, stacking)
        true_file = _find_truth_nz(root_dir, algo, binning, b)
        if est_file is None:
            warnings.warn(f"[MISS] estimated n(z) not found: {algo} / {binning} / bin{b} / {stacking}")
            z_bias.append(0.0); sigmaz_bias.append(0.0)
            continue
        if true_file is None:
            warnings.warn(f"[MISS] true n(z) not found: {algo} / {binning} / bin{b}")
            z_bias.append(0.0); sigmaz_bias.append(0.0)
            continue

        est_ens = _read_qp_ensemble(est_file)
        true_ens = _read_qp_ensemble(true_file)

        est_stats = get_z_sigmaz_mean_sigma(est_ens)
        tru_stats = get_z_sigmaz_mean_sigma(true_ens)  # same measure (mean z and sigma_z/(1+z))

        # bias = estimated - true
        z_bias.append(est_stats["zmean_val"] - tru_stats["zmean_val"])
        sigmaz_bias.append(est_stats["zsigma_val"] - tru_stats["zsigma_val"])

    return np.array(z_bias), np.array(sigmaz_bias)


def build_and_run_fisher(
    bins, prior_bias_by_1plusz, prior_sigma_by_1plusz,
    baseline_zvariance, z_bias, sigmaz_bias, outlier_bias_vec,
    save_deriv_template, overwrite_deriv, probe_label, ystage_label
):
    """
    Build base and biased Fisher, process, compute centroid-shift parameter biases.
    """
    # base Fisher (priors)
    my_priors = {}
    for i in range(len(bins)):
        my_priors[f'zbias{i+1}'] = prior_bias_by_1plusz[i]
        my_priors[f'zvariance{i+1}'] = prior_sigma_by_1plusz[i]


    if ystage_label=='y1':
        yboo = True
    else:
        yboo = False
    print(save_deriv_template.format(probe=probe_label, y1=ystage_label))
    base = Fisher(
        cosmo,
        outliers=[0.0]*len(bins),
        zbias=[0.0]*len(bins),
        zvariance=baseline_zvariance[:len(bins)],
        save_deriv=save_deriv_template.format(probe=probe_label, y1=ystage_label),
        probe = probe_label,
        y1 = yboo,
        overwrite=overwrite_deriv
    )
    base.override_priors(my_priors)
    base.process()

    # biased Fisher (apply the measured biases)
    biased = Fisher(
        cosmo,
        outliers=(np.array([0.0]*len(bins)) + outlier_bias_vec),
        zbias=np.array(z_bias),
        zvariance=(np.array(baseline_zvariance[:len(bins)]) + np.array(sigmaz_bias)),
        save_deriv=save_deriv_template.format(probe=probe_label, y1=ystage_label),
        probe = probe_label,
        y1 = yboo,
        overwrite=False
    )
    biased.process()

    # cosmology parameter bias estimates
    para_bias = centroid_shift(base, Wrapper(np.array(biased.ccl_cls), cosmic_shear=False))
    para_bias_lcdm = centroid_shift_lcdm(base, Wrapper(np.array(biased.ccl_cls), cosmic_shear=False))

    # convert to plain dict if needed
    def _todict(x):
        if hasattr(x, "items"):
            return {k: (float(v) if np.ndim(v)==0 else np.asarray(v).tolist()) for k, v in x.items()}
        return x
    return _todict(para_bias), _todict(para_bias_lcdm)


def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = Path(cfg["root_dir"])
    forecast_dir = Path(cfg.get("forecast_dir", "forecast"))
    forecast_dir.mkdir(parents=True, exist_ok=True)

    algorithms = cfg["algorithms"]
    binnings = cfg["binnings"]
    stackings = cfg["stackings"]
    bins = cfg.get("bins", [0,1,2,3,4])

    prior_bias_by_1plusz = cfg["prior_bias_by_1plusz"]
    prior_sigma_by_1plusz = cfg["prior_sigma_by_1plusz"]
    baseline_zvariance = cfg["baseline_zvariance"]

    outlier_bias_vec = np.array(cfg.get("outlier_bias", [0.0]*len(bins)))

    save_deriv_template = cfg.get("save_deriv_template", "data/obj_deriv_{probe}_{y1}.pkl")
    overwrite_deriv = bool(cfg.get("overwrite_deriv", False))

    # run both Y1 and Y10 by default
    y_stages = []
    if cfg.get("run_y1", True):  y_stages.append(("y1", "Y1"))
    if cfg.get("run_y10", True): y_stages.append(("y10", "Y10"))

    probes = cfg.get("fisher_probes", ["3x2pt", "ss"])  # labels only

    for algo in algorithms:
        for binning in binnings:
            for stacking in stackings:
                # compute per-bin biases for this scenario
                print(algo, binning, stacking)
                z_bias, sigmaz_bias = compute_bias_vectors(root_dir, algo, binning, stacking, bins)
                

                # temporary fix

                z_bias[0] = 0.0
                sigmaz_bias[0] = 0.0
                # z_bias = np.concatenate([z_bias, np.array([0])])
                # sigmaz_bias = np.concatenate([sigmaz_bias, np.array([0])])

                print(z_bias, sigmaz_bias)
                print(prior_bias_by_1plusz, prior_sigma_by_1plusz)

                
                # store the bias vectors for inspection
                scen_dir = forecast_dir / f"{algo}_{binning}_{stacking}"
                scen_dir.mkdir(parents=True, exist_ok=True)
                with open(scen_dir / "nz_bias_vectors.json", "w") as f:
                    json.dump({
                        "bins": bins,
                        "z_bias": z_bias.tolist(),
                        "sigmaz_bias": sigmaz_bias.tolist()
                    }, f, indent=2)

                # run all probe x ystage combinations
                for probe in probes:
                    for ystage_key, ystage_label in y_stages:

                        print(overwrite_deriv, probe, save_deriv_template)
                        
                        para_bias, para_bias_lcdm = build_and_run_fisher(
                            bins=bins,
                            prior_bias_by_1plusz=prior_bias_by_1plusz,
                            prior_sigma_by_1plusz=prior_sigma_by_1plusz,
                            baseline_zvariance=baseline_zvariance,
                            z_bias=z_bias,
                            sigmaz_bias=sigmaz_bias,
                            outlier_bias_vec=outlier_bias_vec,
                            save_deriv_template=save_deriv_template,
                            overwrite_deriv=overwrite_deriv,
                            probe_label=probe,
                            ystage_label=ystage_key
                        )

                        # save results
                        out_prefix = f"{probe}_{ystage_key}"
                        with open(scen_dir / f"{out_prefix}_para_bias.json", "w") as f:
                            json.dump(para_bias, f, indent=2)
                        with open(scen_dir / f"{out_prefix}_para_bias_lcdm.json", "w") as f:
                            json.dump(para_bias_lcdm, f, indent=2)

                        print(f"[OK] {algo}/{binning}/{stacking} -> {probe} {ystage_key} biases saved in {scen_dir}")

from typing import Optional

def run(cfg_path: Optional[str] = None):
    if cfg_path is None:
        raise ValueError("forecast_flavor.run requires a config path")
    # Try the common entry names in your script:
    for fn_name in ("main", "main2", "run"):
        fn = globals().get(fn_name, None)
        if callable(fn):
            return fn(cfg_path)
    raise RuntimeError("forecast_flavor.py: no main/main2/run(cfg) found")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_forecasts.py config.yml")
        sys.exit(1)
    main(sys.argv[1])
