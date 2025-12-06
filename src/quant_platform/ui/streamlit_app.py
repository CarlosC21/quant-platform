from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from quant_platform.runner.config.loader import load_config
from quant_platform.runner.strategy_factory import (
    STRATEGY_REGISTRY,
)
from quant_platform.runner.strategy_meta import get_strategy_param_schema
from quant_platform.runner.run import run_from_config, _load_market_data
from quant_platform.ui.data_validation import validate_market_data


# ============================================================
# Sidebar — Load config
# ============================================================


def sidebar_load_config() -> Path | None:
    st.sidebar.header("⚙️ Load Backtest Configuration")

    uploaded = st.sidebar.file_uploader(
        "Upload JSON/YAML config",
        type=["json", "yaml", "yml"],
    )

    if uploaded is None:
        return None

    suffix = uploaded.name.split(".")[-1]
    tmp_path = Path(f"uploaded_config.{suffix}")
    tmp_path.write_bytes(uploaded.read())
    return tmp_path


# ============================================================
# Strategy parameter editing (PARAM_SCHEMA-aware)
# ============================================================


def render_strategy_param_editor(
    strategy_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Render Streamlit widgets for a strategy using PARAM_SCHEMA if available.
    Falls back gracefully if no schema is defined.
    """
    st.subheader("🔧 Strategy Parameters")

    cls = STRATEGY_REGISTRY.get(strategy_name)
    if cls is None:
        st.error(f"Strategy '{strategy_name}' not found in registry.")
        return params

    schema = get_strategy_param_schema(cls)
    new_params = dict(params)

    if not schema:
        st.info("Strategy has no PARAM_SCHEMA; raw params will be used.")
        for k, v in params.items():
            new_params[k] = st.text_input(k, value=str(v))
        return new_params

    for name, info in schema.items():
        default = new_params.get(name, info.default)

        # INT
        if info.type == "int":
            value = int(default) if default is not None else 0
            number_kwargs: Dict[str, Any] = {"value": value}
            if isinstance(info.min, (int, float)):
                number_kwargs["min_value"] = int(info.min)
            if isinstance(info.max, (int, float)):
                number_kwargs["max_value"] = int(info.max)

            new_params[name] = st.number_input(
                label=info.description or name,
                step=1,
                **number_kwargs,
            )

        # FLOAT
        elif info.type == "float":
            value = float(default) if default is not None else 0.0
            number_kwargs = {"value": value}
            if isinstance(info.min, (int, float)):
                number_kwargs["min_value"] = float(info.min)
            if isinstance(info.max, (int, float)):
                number_kwargs["max_value"] = float(info.max)

            new_params[name] = st.number_input(
                label=info.description or name,
                **number_kwargs,
            )

        # BOOL
        elif info.type == "bool":
            val = bool(default) if default is not None else False
            new_params[name] = st.checkbox(info.description or name, value=val)

        # CHOICE
        elif info.type == "choice" and info.choices:
            choices = list(info.choices)
            if default in choices:
                index = choices.index(default)
            else:
                index = 0
            new_params[name] = st.selectbox(
                info.description or name,
                options=choices,
                index=index,
            )

        # STRING / FALLBACK
        else:
            new_params[name] = st.text_input(
                info.description or name,
                value=str(default) if default is not None else "",
            )

    return new_params


# ============================================================
# Main dashboard
# ============================================================


def main() -> None:
    st.set_page_config(page_title="Quant Platform Dashboard", layout="wide")
    st.title("📈 Quant Research & Trading Platform")
    st.markdown(
        "Upload a **config** → preview **market data** → tweak **strategy params** → run an **execution-aware backtest**."
    )

    # ---------------------
    # Load config
    # ---------------------
    cfg_path = sidebar_load_config()
    if cfg_path is None:
        st.info("Upload a config file (JSON/YAML) in the sidebar to begin.")
        return

    cfg = load_config(cfg_path)

    # ---------------------
    # Market data preview
    # ---------------------
    st.header("📊 Market Data Preview")

    if cfg.data_source is None:
        st.error("Config is missing `data_source` pointing to CSV/Parquet.")
        return

    try:
        df = _load_market_data(cfg.data_source)
        validate_market_data(df)
    except Exception as e:
        st.error(f"Market data error: {e}")
        return

    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Rows: {len(df)}")

    # ---------------------
    # Strategy configuration
    # ---------------------
    st.header("🎯 Strategy Configuration")

    params = dict(cfg.strategy.params)
    strat_name = params.get("name")
    if strat_name is None:
        st.error("Config missing `strategy.params.name`.")
        return

    st.write(f"**Strategy:** `{strat_name}`")

    new_params = render_strategy_param_editor(strat_name, params)
    cfg.strategy.params = new_params

    # ---------------------
    # Run backtest
    # ---------------------
    st.header("🚀 Run Backtest")

    if st.button("Run Backtest"):
        # Write updated config to a temp JSON so run_from_config sees changes
        runtime_cfg_path = Path("ui_runtime_config.json")
        runtime_cfg_path.write_text(cfg.model_dump_json(indent=2))

        with st.spinner("Running backtest…"):
            result = run_from_config(runtime_cfg_path)

        st.success("Backtest completed.")

        # --------------------------------------
        # Equity curve
        # --------------------------------------
        st.subheader("💵 Equity Curve")
        eq_df = result.equity_curve.to_frame(name="equity")
        fig_eq = px.line(eq_df, y="equity", title="Equity Curve")
        st.plotly_chart(fig_eq, use_container_width=True)

        # --------------------------------------
        # Drawdowns
        # --------------------------------------
        st.subheader("📉 Drawdowns")
        dd_df = result.drawdowns.to_frame(name="drawdown")
        fig_dd = px.area(dd_df, y="drawdown", title="Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

        # --------------------------------------
        # Risk metrics
        # --------------------------------------
        st.subheader("📊 Risk Metrics")
        st.json(result.risk_metrics)

        # --------------------------------------
        # Trades
        # --------------------------------------
        if hasattr(result, "trade_log"):
            st.subheader("📝 Trades")
            st.dataframe(pd.DataFrame(result.trade_log), use_container_width=True)

        # --------------------------------------
        # Positions over time
        # --------------------------------------
        if result.positions_ts is not None:
            st.subheader("📦 Positions Over Time")
            st.dataframe(result.positions_ts.head(), use_container_width=True)

        # --------------------------------------
        # Portfolio snapshots
        # --------------------------------------
        if hasattr(result, "portfolio_snapshots"):
            st.subheader("📘 Portfolio Snapshots")
            snap_df = pd.DataFrame(result.portfolio_snapshots)
            st.dataframe(snap_df.head(), use_container_width=True)


if __name__ == "__main__":
    main()
