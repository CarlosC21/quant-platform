from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_platform import __version__
from quant_platform.runner.run import run_from_config


# ============================================================
# Example registry
# ============================================================

EXAMPLES = {
    "stat-arb": "configs/stat_arb_exec.yaml",
    "delta-hedge": "examples/delta_hedge_demo.py",
    "walk-forward": "scripts/run_walk_forward_example.py",
}


# ============================================================
# utils
# ============================================================


def _print(msg: str):
    print(f"[quantp] {msg}")


# ============================================================
# report generator
# ============================================================


def generate_report(run_dir: Path) -> None:
    """Generate a human-readable summary from a run directory."""

    equity_fp = run_dir / "equity.csv"
    # draw_fp = run_dir / "drawdowns.csv"
    risk_fp = run_dir / "risk_metrics.json"

    if not equity_fp.exists():
        raise FileNotFoundError(f"Missing equity.csv in {run_dir}")

    equity = pd.read_csv(equity_fp)
    if "equity" in equity.columns:
        equity_series = equity.set_index("timestamp")["equity"]
    else:
        equity_series = equity.iloc[:, 0]

    # drawdowns = pd.read_csv(draw_fp).iloc[:, 0] if draw_fp.exists() else None

    risk_metrics = json.loads(risk_fp.read_text()) if risk_fp.exists() else {}

    _print("=== REPORT ===")
    _print(f"Run directory: {run_dir}")
    _print("")
    _print(f"Final equity: {equity_series.iloc[-1]:.2f}")

    if risk_metrics:
        for k, v in risk_metrics.items():
            _print(f"{k:20s} {v:.4f}")

    # optionally plot (future extension)


# ============================================================
# command handlers
# ============================================================


def cmd_run(args: argparse.Namespace) -> None:
    config_path = args.config
    save_dir = args.save_dir
    _print(f"Running backtest: {config_path}")

    result = run_from_config(config_path, save_dir)
    _print("Backtest complete.")
    _print(f"Final equity: {result.equity_curve.iloc[-1]:.2f}")


def cmd_examples_list(args: argparse.Namespace) -> None:
    _print("Available examples:")
    for name in EXAMPLES:
        print(f"  - {name}")


def cmd_examples_run(args: argparse.Namespace) -> None:
    name = args.name
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example: {name}")

    example_path = EXAMPLES[name]
    _print(f"Running example '{name}' -> {example_path}")

    if example_path.endswith(".yaml") or example_path.endswith(".yml"):
        run_from_config(example_path, save_dir="runs/example_" + name)
    else:
        # Execute python script example
        _print("Executing script example via Python")
        exec(open(example_path).read(), {})


def cmd_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    generate_report(run_dir)


def cmd_version(args: argparse.Namespace) -> None:
    print(__version__)


# ============================================================
# top-level CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantp", description="Quant Research Platform CLI"
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---------------------------------------------------------
    # run
    # ---------------------------------------------------------
    p_run = subparsers.add_parser("run", help="Run a backtest from config")
    p_run.add_argument("config", type=str, help="Path to YAML config file")
    p_run.add_argument("--save-dir", type=str, default=None)
    p_run.set_defaults(func=cmd_run)

    # ---------------------------------------------------------
    # examples list / run
    # ---------------------------------------------------------
    p_ex = subparsers.add_parser("examples", help="List or run examples")
    ex_sub = p_ex.add_subparsers(dest="ex_cmd")

    p_ex_list = ex_sub.add_parser("list", help="List all examples")
    p_ex_list.set_defaults(func=cmd_examples_list)

    p_ex_run = ex_sub.add_parser("run", help="Run an example")
    p_ex_run.add_argument("name", type=str, help="Example name")
    p_ex_run.set_defaults(func=cmd_examples_run)

    # ---------------------------------------------------------
    # report
    # ---------------------------------------------------------
    p_rep = subparsers.add_parser("report", help="Generate report from run directory")
    p_rep.add_argument("run_dir", type=str)
    p_rep.set_defaults(func=cmd_report)

    # ---------------------------------------------------------
    # version
    # ---------------------------------------------------------
    p_ver = subparsers.add_parser("version", help="Print version")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
