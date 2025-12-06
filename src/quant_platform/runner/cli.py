from __future__ import annotations
import argparse
from quant_platform.runner.run import run_from_config
from quant_platform.examples.stat_arb_backtest_with_execution import run_example


def main():
    parser = argparse.ArgumentParser(prog="quantp", description="Quant Platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------------
    # run
    # -----------------------------
    run_p = subparsers.add_parser("run", help="Run a backtest from config")
    run_p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    run_p.add_argument(
        "--save-dir",
        required=False,
        default=None,
        help="Optional directory to save results",
    )

    # -----------------------------
    # examples
    # -----------------------------
    ex_p = subparsers.add_parser("examples", help="Run built-in examples")
    ex_p.add_argument("example", choices=["stat-arb-basic"], help="Example name")

    # -----------------------------
    # report
    # -----------------------------
    rep_p = subparsers.add_parser("report", help="Generate a simple text report")
    rep_p.add_argument("--run-dir", required=True, help="Directory containing results")

    # -----------------------------
    # version
    # -----------------------------
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    # =============================
    # COMMAND HANDLING
    # =============================

    if args.command == "run":
        result = run_from_config(args.config)
        print("Backtest complete.")
        print(f"Final equity: {result.equity_curve.iloc[-1]:.4f}")
        print(f"Sharpe: {result.risk_metrics.get('sharpe'):.4f}")
        print(f"Max Drawdown: {result.risk_metrics.get('max_drawdown'):.4f}")

    elif args.command == "examples":
        if args.example == "stat-arb-basic":
            run_example()

    elif args.command == "report":
        print(f"[TODO] Reporting from: {args.run_dir}")

    elif args.command == "version":
        from quant_platform import __version__

        print(f"quant_platform version {__version__}")


if __name__ == "__main__":
    main()
