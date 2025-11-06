# fishera2z/cli.py
from __future__ import annotations
import argparse
import sys
import os

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="forecast-pz",
        description="Run Fisher A2Z forecast stages via a single CLI."
    )
    g = p.add_argument_group("Stages (choose one or combine)")
    g.add_argument("--prepare-fisher", action="store_true",
                   help="Precompute all Fisher matrices")
    g.add_argument("--forecast-flavor", action="store_true",
                   help="Predict cosmological bias per scenario")
    g.add_argument("--create-plots", action="store_true",
                   help="Generate all plots per scenario")
    g.add_argument("--build-pdf", action="store_true",
                   help="Summarize results into a PDF report")
    g.add_argument("--all", action="store_true",
                   help="Run all stages in order")

    p.add_argument("config", nargs="?", help="Path to config.yml (required for most stages)")
    p.add_argument("-q", "--quiet", action="store_true", help="Reduce verbosity")
    p.add_argument("--matplotlib-backend", default="Agg",
                   help="Matplotlib backend to use (default: Agg)")

    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    # Set a non-interactive backend by default (good for HPC)
    os.environ.setdefault("MPLBACKEND", args.matplotlib_backend)

    # Resolve which stages to run
    run_prepare  = args.prepare_fisher or args.all
    run_forecast = args.forecast_flavor or args.all
    run_plots    = args.create_plots or args.all
    run_pdf      = args.build_pdf or args.all

    if not any([run_prepare, run_forecast, run_plots, run_pdf]):
        print("No stages selected. Use --prepare-fisher, --forecast-flavor, --create-plots, --build-pdf, or --all.")
        return 2

    # Config required?
    need_cfg = any([run_forecast, run_plots, run_pdf])
    if need_cfg and not args.config:
        print("error: config.yml is required for this stage. Example: forecast-pz --forecast-flavor config.yml")
        return 2

    cfg_path = args.config

    try:
        if run_prepare:
            from fisherA2Z.stages import prepare_fisher
            if not args.quiet:
                print("[forecast-pz] Stage: prepare_fisher")
            prepare_fisher.run(cfg_path)  # pass cfg if your code needs it

        if run_forecast:
            from fisherA2Z.stages import forecast_flavor
            if not args.quiet:
                print("[forecast-pz] Stage: forecast_flavor")
            forecast_flavor.run(cfg_path)

        if run_plots:
            from fisherA2Z.stages import create_plots
            if not args.quiet:
                print("[forecast-pz] Stage: create_plots")
            create_plots.run(cfg_path)

        if run_pdf:
            from fisherA2Z.stages import build_pdf
            if not args.quiet:
                print("[forecast-pz] Stage: build_pdf")
            build_pdf.run(cfg_path)

    except KeyboardInterrupt:
        print("\n[forecast-pz] Aborted by user.")
        return 130
    except Exception as e:
        print(f"[forecast-pz] ERROR: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
