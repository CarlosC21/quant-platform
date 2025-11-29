import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from quant_platform.data.feature_store.store import FeatureStore
from quant_platform.ml.hyperparam_search import WalkForwardHyperparamSearch

# Step 1: Load / generate features
data = pd.DataFrame(
    {
        "date": pd.date_range("2023-01-01", periods=50, freq="D"),
        "feat1": np.random.randn(50),
        "feat2": np.random.randn(50),
        "target": np.random.randn(50),
    }
)

fs = FeatureStore()
fs.save_features("example_features", data)

# Step 2: Define model and hyperparameter grid
param_grid = {"alpha": [0.1, 1.0, 10.0]}
base_model = Ridge()

# Step 3: Run hyperparameter search with walk-forward CV
search = WalkForwardHyperparamSearch(
    base_model=base_model,
    param_grid=param_grid,
    feature_store=fs,
    feature_name="example_features",
    target="target",
    val_window=5,
    embargo_days=1,
)

results = search.run()
print("Best params:", search.best_params())

# Optional: inspect metrics
for res in results:
    print(res["params"], "-> avg RMSE:", res["avg_rmse"])
