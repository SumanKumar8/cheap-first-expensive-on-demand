#!/usr/bin/env python3
"""Run the Cheap-First / Expensive-on-Demand experiment pipeline.

Examples
--------
From a terminal:
    python run_pipeline.py

From a Jupyter notebook:
    !python run_pipeline.py

Use a custom artifacts directory:
    python run_pipeline.py --artifacts /path/to/Hierarchical-Model-SNN-main/artifacts

Optionally install dependencies first:
    python run_pipeline.py --install
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(script: str, *args: str) -> int:
    """Run a Python script and show the tail of its output."""
    script_path = SCRIPT_DIR / script
    command = [PYTHON, str(script_path), *args]

    print("\n»", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout[-1500:])

    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-1200:], file=sys.stderr)

    return result.returncode


def require_success(script: str, *args: str) -> None:
    """Run a script and stop the pipeline if it fails."""
    return_code = run(script, *args)
    if return_code != 0:
        raise SystemExit(return_code)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_artifacts = os.environ.get(
        "SNN_ARTIFACTS",
        str((SCRIPT_DIR / "../Hierarchical-Model-SNN-main/artifacts").resolve()),
    )

    parser = argparse.ArgumentParser(
        description="Run the Cheap-First / Expensive-on-Demand pipeline."
    )
    parser.add_argument(
        "--artifacts",
        default=default_artifacts,
        help="Path containing FashionMNIST_Experiments, MNIST_Experiments, and SVHN_Experiments.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install packages from requirements.txt before running the pipeline.",
    )
    return parser.parse_args(argv)


def print_generated_outputs() -> None:
    """List generated figures and print generated Markdown tables."""
    figures_pattern = str(SCRIPT_DIR / "results/figures/*.png")
    tables_pattern = str(SCRIPT_DIR / "results/tables/*.md")

    figures = sorted(glob.glob(figures_pattern))
    print("\n=== Generated figures ===")
    if figures:
        for figure in figures:
            print(Path(figure).relative_to(SCRIPT_DIR))
    else:
        print("No PNG figures found.")

    tables = sorted(glob.glob(tables_pattern))
    print("\n=== Generated tables ===")
    if tables:
        for table in tables:
            table_path = Path(table)
            print(f"\n--- {table_path.name} ---")
            print(table_path.read_text(encoding="utf-8"))
    else:
        print("No Markdown tables found.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    artifacts_path = Path(args.artifacts).expanduser().resolve()
    if not artifacts_path.is_dir():
        print(
            "SNN artifacts folder not found:\n"
            f"  {artifacts_path}\n\n"
            "Set it with either:\n"
            "  python run_pipeline.py --artifacts /path/to/artifacts\n"
            "or the SNN_ARTIFACTS environment variable.",
            file=sys.stderr,
        )
        return 2

    os.environ["SNN_ARTIFACTS"] = str(artifacts_path)
    print("artifacts:", artifacts_path)

    if args.install:
        requirements = SCRIPT_DIR / "requirements.txt"
        if not requirements.is_file():
            print(f"requirements.txt not found at {requirements}", file=sys.stderr)
            return 2
        print("\n=== Installing dependencies ===")
        install_result = subprocess.run(
            [PYTHON, "-m", "pip", "install", "-q", "-r", str(requirements)],
            cwd=SCRIPT_DIR,
            check=False,
        )
        if install_result.returncode != 0:
            return install_result.returncode

    print("\n=== 1. Build residual pools ===")
    for dataset in ("fashionmnist", "mnist", "svhn"):
        require_success("build_pools.py", dataset, "0", "9")
        require_success("build_pools.py", dataset, "merge")

    print("\n=== 2. E1 detectors, E2 conditioning, E3 cascade cost ===")
    for dataset in ("fashionmnist", "mnist", "svhn"):
        for metric in ("isi", "cv"):
            require_success("real_experiments.py", "e13", dataset, metric)
        require_success("real_experiments.py", "e2", dataset)

    print("\n=== 3. Stage-2 VP/VR, fault breakdown, drift, statistics ===")
    require_success("e4_stage2_vpvr.py")
    require_success("fault_breakdown.py", "derivable")
    require_success("e5_drift.py")
    require_success("stats_support.py")

    print("\n=== 4. Aggregate report and verification ===")
    require_success("make_report.py")
    require_success("verify_all.py")

    print_generated_outputs()
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
