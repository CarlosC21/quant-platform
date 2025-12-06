import streamlit as st
import pandas as pd
import tempfile
import yaml
import json

from quant_platform.runner.run import run_from_config
from quant_platform.runner.strategy_factory import (
    STRATEGY_REGISTRY,
    autodiscover_strategies,
)

st.set_page_config(page_title="Quant Platform", layout="wide")

st.title("📈 Quant Research & Trading Platform")
st.markdown("Upload data → configure → run execution-aware backtest.")


# ==============================================================
# SESSION STATE
# ==============================================================

if "config_path" not in st.session_state:
    st.session_state["config_path"] = None

if "uploaded_csv_path" not in st.session_state:
    st.session_state["uploaded_csv_path"] = None

if "symbols" not in st.session_state:
    st.session_state["symbols"] = []

if "selected_strategy" not in st.session_state:
    st.session_state["selected_strategy"] = None


# ==============================================================
# STEP 1 — UPLOAD MARKET DATA
# ==============================================================

st.header("Step 1 — Upload Market Data (CSV)")

data_file = st.file_uploader("Upload CSV", type=["csv"])
df = None

if data_file:
    try:
        df = pd.read_csv(data_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["timestamp", "symbol"])

        st.session_state["symbols"] = sorted(df["symbol"].unique())

        st.success(
            f"Loaded market data. Symbols detected: {st.session_state['symbols']}"
        )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        df.to_csv(tmp.name, index=False)
        st.session_state["uploaded_csv_path"] = tmp.name

        st.dataframe(df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"CSV load error: {e}")
        st.stop()


# ==============================================================
# STEP 2 — STRATEGY AUTODISCOVERY
# ==============================================================

st.header("Step 2 — Strategy Discovery")
st.write("You may load additional strategy modules dynamically.")

modules_input = st.text_input(
    "Module paths (comma-separated)",
    placeholder="e.g. my_strats.mean_reversion, my_strats.options",
)

if st.button("🔍 Discover Strategies"):
    if modules_input.strip():
        modules = [m.strip() for m in modules_input.split(",")]
        autodiscover_strategies(modules)
        st.success(f"Imported modules: {modules}")
    else:
        st.warning("Enter at least one module path.")


# Show registry summary
with st.expander("📦 Registered Strategies"):
    st.json(list(STRATEGY_REGISTRY.keys()))


# ==============================================================
# STEP 3 — STRATEGY SELECTION & PARAMS
# ==============================================================

st.header("Step 3 — Strategy Configuration")

# Strategy dropdown
strategy_names = sorted(STRATEGY_REGISTRY.keys())
selected = st.selectbox("Choose a strategy", ["<select>"] + strategy_names)

if selected != "<select>":
    st.session_state["selected_strategy"] = selected
else:
    st.session_state["selected_strategy"] = None


# Dynamic parameter editor
strategy_params = {}
if st.session_state["selected_strategy"]:
    st.subheader("Strategy Parameters")

    # Give a JSON editor for flexibility
    params_json = st.text_area(
        "Params (JSON dict)",
        value="{}",
        height=150,
        placeholder='{"lookback": 20, "vol_target": 0.1}',
    )

    try:
        strategy_params = json.loads(params_json)
    except Exception:
        st.warning("JSON could not be parsed. Using empty parameters.")
        strategy_params = {}


# ==============================================================
# STEP 4 — CONFIG FILE (AUTO-GENERATED)
# ==============================================================

st.header("Step 4 — Auto-Generated Config")

config_preview = None

if st.session_state["uploaded_csv_path"] and st.session_state["selected_strategy"]:
    config_preview = {
        "name": f"{st.session_state['selected_strategy']}_run",
        "strategy": {
            "params": {"name": st.session_state["selected_strategy"], **strategy_params}
        },
        "data_source": st.session_state["uploaded_csv_path"],
        "initial_cash": 100000,
        "execution": {
            "latency_seconds": 0.0,
            "slippage_bps": 0.0,
            "cost_bps": 0.0,
        },
        "save": {
            "directory": "runs/ui_output",
            "save_equity_curve": True,
            "save_positions": True,
            "save_trades": True,
        },
    }

    st.code(yaml.safe_dump(config_preview), language="yaml")

    tmp_cfg = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
    yaml.safe_dump(config_preview, tmp_cfg)
    tmp_cfg.close()

    st.session_state["config_path"] = tmp_cfg.name


else:
    st.info("Select a strategy first and upload data.")


# ==============================================================
# STEP 5 — RUN BACKTEST
# ==============================================================

st.header("Step 5 — Run Backtest")

run_btn = st.button("🚀 Run Backtest")

if run_btn:
    if not st.session_state["config_path"]:
        st.error("No config available. Complete steps above.")
        st.stop()

    st.info("Running backtest...")

    try:
        result = run_from_config(st.session_state["config_path"])
    except Exception as e:
        st.error(f"Backtest error: {e}")
        st.stop()

    st.success("Backtest complete!")

    st.subheader("📈 Equity Curve")
    st.line_chart(result.equity_curve)

    st.subheader("📉 Drawdowns")
    st.line_chart(result.drawdowns)

    st.subheader("📊 Risk Metrics")
    st.json(result.risk_metrics)

    if getattr(result, "positions_ts", None) is not None:
        st.subheader("📘 Positions Over Time")
        st.dataframe(result.positions_ts, use_container_width=True)
