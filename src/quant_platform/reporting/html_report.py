from __future__ import annotations
from pathlib import Path

HTML_TEMPLATE = """
<html>
<head>
<title>Backtest Report</title>
<style>
body { font-family: Arial; margin: 40px; }
h1 { color: #333; }
table { border-collapse: collapse; width: 70%; margin-bottom: 40px; }
td, th { border: 1px solid #ccc; padding: 8px; }
</style>
</head>
<body>

<h1>Backtest Report</h1>

<h2>Performance Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Final Equity</td><td>{final_equity}</td></tr>
<tr><td>Max Drawdown (%)</td><td>{max_dd}</td></tr>
<tr><td>Sharpe Ratio</td><td>{sharpe}</td></tr>
</table>

<h2>Equity Curve (first 10 rows)</h2>
{equity_table}

</body>
</html>
"""


def generate_html_report(result, path: str | Path):
    eq = result.equity_curve
    dd = result.drawdowns["drawdown_pct"].min()

    sharpe = result.metrics.get("sharpe", "N/A")

    html = HTML_TEMPLATE.format(
        final_equity=eq.iloc[-1],
        max_dd=dd,
        sharpe=sharpe,
        equity_table=eq.head().to_html(),
    )

    path = Path(path)
    path.write_text(html)
    return path
