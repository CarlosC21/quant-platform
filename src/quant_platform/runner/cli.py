from __future__ import annotations

import argparse

from quant_platform.runner.run import run_from_config


def main():
    parser = argparse.ArgumentParser(description="Quant Platform Backtest Runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to backtest config YAML/JSON",
    )
    parser.add_argument(
        "--save-dir",
        required=False,
        default=None,
        help="Optional directory to save results",
    )

    args = parser.parse_args()

    result = run_from_config(args.config, save_dir=args.save_dir)
    print("Backtest complete.")
    print(f"Final equity: {result.equity_curve.iloc[-1]:.4f}")
    print(f"Sharpe: {result.risk_metrics.get('sharpe'):.4f}")
    print(f"Max Drawdown: {result.risk_metrics.get('max_drawdown'):.4f}")


if __name__ == "__main__":
    main()
