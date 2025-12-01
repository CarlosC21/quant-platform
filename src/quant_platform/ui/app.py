import streamlit as st
import pandas as pd
import tempfile
import yaml
from quant_platform.runner.run import run_from_config

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


# ==============================================================
# STEP 1 — UPLOAD MARKET DATA
# ==============================================================

st.header("Step 1 — Upload Market Data (CSV)")

data_file = st.file_uploader("Upload CSV", type=["csv"])

df = None

if data_file:
    try:
        df = pd.read_csv(data_file)

        # Force timestamp parsing
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y")

        df = df.sort_values(["timestamp", "symbol"])
        st.session_state["symbols"] = sorted(df["symbol"].unique())

        st.success(f"Loaded data. Detected symbols = {st.session_state['symbols']}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        df.to_csv(tmp.name, index=False)
        st.session_state["uploaded_csv_path"] = tmp.name

        st.dataframe(df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"CSV load error: {e}")
        st.stop()


# ==============================================================
# STEP 2 — CONFIGURATION
# ==============================================================

st.header("Step 2 — Strategy Configuration")

config_file = st.file_uploader("Upload YAML/JSON", type=["yaml", "json"])
use_default = st.button("Use Default Stat-Arb Config")

default_template = """
name: stat_arb_run

strategy:
  params:
    name: stat_arb_exec
    y_symbol: AAA
    x_symbol: BBB
    vol_target: 0.10
    dollar_neutral: true

data_source: PLACEHOLDER

execution:
  latency_seconds: 0.0
  slippage_bps: 0.0
  cost_bps: 0.0

initial_cash: 100000

save:
  directory: runs/stat_arb_output
  save_equity_curve: true
  save_positions: true
  save_trades: true
"""

with st.expander("📘 Default Template"):
    st.code(default_template, language="yaml")


# ----- HANDLE DEFAULT CONFIG -----

if use_default:
    try:
        cfg = yaml.safe_load(default_template)

        # Inject detected symbols
        if len(st.session_state["symbols"]) >= 2:
            y, x = st.session_state["symbols"][:2]
            cfg["strategy"]["params"]["y_symbol"] = y
            cfg["strategy"]["params"]["x_symbol"] = x
            st.info(f"Auto-selected pair: {y} / {x}")

        # Inject CSV path
        if st.session_state["uploaded_csv_path"]:
            cfg["data_source"] = st.session_state["uploaded_csv_path"]

        # TEMP YAML FILE MUST BE IN TEXT MODE
        tmp_cfg = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
        yaml.safe_dump(cfg, tmp_cfg)
        tmp_cfg.close()

        st.session_state["config_path"] = tmp_cfg.name
        st.success("Default config loaded. Ready to run.")

    except Exception as e:
        st.error(f"Default config error: {e}")


# ----- HANDLE USER CONFIG -----

if config_file:
    suffix = ".yaml" if config_file.name.endswith(".yaml") else ".json"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb")
    tmp.write(config_file.read())
    tmp.close()

    st.session_state["config_path"] = tmp.name
    st.success("Custom config loaded.")


# ==============================================================
# STEP 3 — RUN BACKTEST
# ==============================================================

st.header("Step 3 — Run Backtest")
run_btn = st.button("🚀 Run Backtest")

if run_btn:
    if st.session_state["uploaded_csv_path"] is None:
        st.error("Upload CSV first.")
        st.stop()

    if st.session_state["config_path"] is None:
        st.error("Upload config or use default.")
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
