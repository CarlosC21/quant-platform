from __future__ import annotations

import argparse
from pathlib import Path

from quant_platform.runner.run import run_from_config
from quant_platform import __version__


# ============================================================
# Example registry
# ============================================================

EXAMPLES = {
    "stat-arb": {
        "description": "Stat-arb pair trading example with execution engine.",
        "config": "examples/stat_arb_example.json",
    },
    "delta-hedge": {
        "description": "Delta hedge simulation example.",
        "config": "examples/delta_hedge_example.json",
    },
    "walk-forward": {
        "description": "Walk-forward ML pipeline example.",
        "config": "examples/walk_forward_example.json",
    },
}


# ============================================================
# Command: run
# ============================================================


def cmd_run(args):
    print(f"[quantp] Running backtest: {args.config}")
    result = run_from_config(args.config, save_dir=args.save_dir)

    print("\n========== Backtest Complete ==========")
    print(f"Final equity: {result.equity_curve.iloc[-1]:.4f}")
    print(f"Sharpe: {result.risk_metrics.get('sharpe'):.4f}")
    print(f"Max drawdown: {result.risk_metrics.get('max_drawdown'):.4f}")
    print("=======================================\n")


# ============================================================
# Command: examples list
# ============================================================


def cmd_examples_list(args):
    print("[quantp] Available examples:")
    for name, meta in EXAMPLES.items():
        print(f"  - {name} : {meta['description']}")


# ============================================================
# Command: examples run <name>
# ============================================================


def cmd_examples_run(args):
    name = args.example

    if name not in EXAMPLES:
        raise ValueError(f"Unknown example: {name}")

    cfg_path = Path(EXAMPLES[name]["config"])
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Example config not found: {cfg_path}. "
            f"Ensure example files are included in the package."
        )

    print(f"[quantp] Running example '{name}' … using {cfg_path}")
    result = run_from_config(cfg_path, save_dir=args.save_dir)

    print("\n========== Example Complete ==========")
    print(f"Final equity: {result.equity_curve.iloc[-1]:.4f}")
    print("======================================\n")


# ============================================================
# Command: report
# ============================================================


def cmd_report(args):
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"[quantp] Loading report from {run_dir}")

    eq = run_dir / "equity_curve.csv"
    pos = run_dir / "positions.csv"
    trades = run_dir / "trades.json"

    if eq.exists():
        print("  ✓ equity_curve.csv")
    if pos.exists():
        print("  ✓ positions.csv")
    if trades.exists():
        print("  ✓ trades.json")
    print("[quantp] Report complete.")


# ============================================================
# Command: version
# ============================================================


def cmd_version(args):
    print(__version__)


# ============================================================
# Main CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(prog="quantp")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a backtest from a config file")
    p_run.add_argument("--config", required=True, help="Path to config JSON/YAML")
    p_run.add_argument(
        "--save-dir", required=False, default=None, help="Directory to save results"
    )
    p_run.set_defaults(func=cmd_run)

    # ------------------------------------------------------------------
    # examples
    # ------------------------------------------------------------------
    p_ex = sub.add_parser("examples", help="List or run examples")
    ex_sub = p_ex.add_subparsers(dest="examples_cmd", required=True)

    p_list = ex_sub.add_parser("list", help="List available examples")
    p_list.set_defaults(func=cmd_examples_list)

    p_run_ex = ex_sub.add_parser("run", help="Run an example")
    p_run_ex.add_argument("example", help="Example name")
    p_run_ex.add_argument("--save-dir", default=None)
    p_run_ex.set_defaults(func=cmd_examples_run)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    p_rep = sub.add_parser("report", help="Show summary for a completed run")
    p_rep.add_argument("--run-dir", required=True, help="Path to saved run directory")
    p_rep.set_defaults(func=cmd_report)

    # ------------------------------------------------------------------
    # version
    # ------------------------------------------------------------------
    p_ver = sub.add_parser("version", help="Show version")
    p_ver.set_defaults(func=cmd_version)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
