# examples/delta_hedge_demo.py
import numpy as np
from src.quant_platform.options.models.black_scholes import delta_hedge_simulator
from src.quant_platform.options.models.local_vol import LocalVolSurface, LocalVolOption

np.random.seed(42)

# -------------------------------
# Simulation parameters
# -------------------------------
S0 = 100
K = 100
T = 1.0
r = 0.05
sigma_bs = 0.2
n_steps = 50
n_paths = 1000
option_types = ["call", "put"]

# Simulate spot paths (lognormal)
dt = T / n_steps
S_paths = S0 * np.exp(
    np.cumsum(sigma_bs * np.sqrt(dt) * np.random.randn(n_paths, n_steps), axis=1)
)

# -------------------------------
# Local Vol surface (flat for demo)
# -------------------------------
strikes = np.array([80, 100, 120])
maturities = np.array([0.5, 1.0, 1.5])
vol_matrix = np.array([[0.18, 0.19, 0.20], [0.20, 0.20, 0.20], [0.22, 0.21, 0.20]])
lv_surface = LocalVolSurface(strikes, maturities, vol_matrix)

# -------------------------------
# Run delta-hedge simulations
# -------------------------------
results = []

for opt_type in option_types:
    bs_pnls = []
    lv_pnls = []

    for path in S_paths:
        # Black-Scholes
        pnl_bs = delta_hedge_simulator(
            path, K, T, r, sigma_bs, option_type=opt_type, dt=dt
        )
        bs_pnls.append(pnl_bs)

        # Local Vol (use scalar sigma at (K,T) as placeholder)
        lv_option = LocalVolOption(
            S=S0, K=K, T=T, r=r, local_vol_surface=lv_surface, option_type=opt_type
        )
        pnl_lv = delta_hedge_simulator(
            path, K, T, r, lv_option.sigma, option_type=opt_type, dt=dt
        )
        lv_pnls.append(pnl_lv)

    results.append(
        {
            "option_type": opt_type,
            "bs_mean_pnl": np.mean(bs_pnls),
            "bs_std_pnl": np.std(bs_pnls),
            "lv_mean_pnl": np.mean(lv_pnls),
            "lv_std_pnl": np.std(lv_pnls),
        }
    )

# -------------------------------
# Print results
# -------------------------------
print("\nDelta-Hedge Simulation Results (1000 paths)")
print("Option | BS P&L (mean ± std) | LV P&L (mean ± std)")
print("-" * 55)
for r in results:
    print(
        f"{r['option_type']:>4}   | "
        f"{r['bs_mean_pnl']:+.4f} ± {r['bs_std_pnl']:.4f} | "
        f"{r['lv_mean_pnl']:+.4f} ± {r['lv_std_pnl']:.4f}"
    )
