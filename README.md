# A-FIS — A-Subsethood Fuzzy Inference System

A Python library implementing A-FIS (A-subsethood based Fuzzy Inference System), with rule learning, inference, regression, and visualization.

---

## Package Structure

```
afis/
├── core/
│   ├── afis_utils.py      # Membership functions, fuzzy rule structures, defuzzification
│   ├── A_FIS.py           # Main inference algorithm
│   ├── A_vee_B.py         # Supremum ν(A ∨ B) — analytical and numerical
│   └── wangmendel.py      # Wang-Mendel rule learning
├── visualization/
│   └── plotting.py        # Rule base plots, supremum visualization, diagnostics
├── regression/
│   ├── regressor.py       # AFISRegressor, k-fold evaluation, benchmarking
│   └── utils.py           # Correlation analysis, metrics, result plots
└── examples/
    ├── inference/          # 1D, ND, Gaussian, and supremum notebooks
    └── regression/         # Regression application notebook
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install numpy scipy scikit-learn pandas plotly tqdm jupyterlab
```

---

## Quick Start

### Inference

```python
from afis.core import FuzzySet, InferiorBorder, SuperiorBorder, Triangular, format_FN_N_Dim
from afis.visualization import create_rule_base, run_afis, plot_results

A1 = FuzzySet("A1", InferiorBorder(0, 5))
A2 = FuzzySet("A2", Triangular(0, 5, 10))
A3 = FuzzySet("A3", SuperiorBorder(5, 10))
Y1 = FuzzySet("Y1", InferiorBorder(0, 5))
Y2 = FuzzySet("Y2", Triangular(0, 5, 10))
Y3 = FuzzySet("Y3", SuperiorBorder(5, 10))

rule_base = create_rule_base(
    antecedents=[[A1, A2, A3]],
    consequents=[Y1, Y2, Y3],
    input_ranges=[(0, 10)],
    output_range=(0, 10),
    rules=[([0], 0), ([1], 1), ([2], 2)]
)

x = format_FN_N_Dim([3.0])
output, U, y_crisp, _ = run_afis(x, rule_base)
plot_results(rule_base, x, output, U, y_crisp)
```

### Regression

```python
import numpy as np
from afis.regression import AFISRegressor, generate_rule_base, evaluate_kfold

X, y = ...  # your data

config = {
    'agg_method': 'avg',        # 'avg' | 'min' | 'max' | 'product' | ['weighted_avg', weights]
    't_norm_type': 'product',   # 'product' | 'min' | 'luka' | ['hamacher', param]
    'imp_params': ['luka', 1],  # implication type and parameter
    'k_max': 10,                # neighbor search ceiling (or fixed k with 'k_fixed')
    'p_norm': 1,                # Minkowski distance order
}

rule_base = generate_rule_base(X, y, n_fuzzy_partitions=5, n_rules=30)

model = AFISRegressor(config)
model.fit(X, y, rule_base, X_val=X_val, y_val=y_val, optimize_k=True)
predictions = model.predict(X_test)
model.save('model.pkl')
```

### K-Fold Cross-Validation

```python
results = evaluate_kfold(
    X, y,
    rule_base_generator=lambda X, y: generate_rule_base(X, y, 5, 30),
    config=config,
    n_splits=5,
    n_repetitions=3,
)
print(f"RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
```

---

## Membership Functions

| Class | Parameters | Shape |
|---|---|---|
| `Triangular` | `ini, top, end` | Triangle |
| `Trapezoidal` | `ini, top1, top2, end` | Trapezoid |
| `InferiorBorder` | `top, end` | Left-saturated ramp |
| `SuperiorBorder` | `ini, top` | Right-saturated ramp |
| `Gaussian` | `center, sigma` | Gaussian bell |

---

## Key API

### `afis.core`

| Symbol | Description |
|---|---|
| `A_FIS(input, rule_base, ...)` | Main inference function |
| `format_FN_N_Dim(x, ftype)` | Format inputs for A_FIS |
| `fuzzy_imp(a, b, type, param)` | Fuzzy implication I(a, b) |
| `nu_A_vee_B_auto(A, B, U)` | Supremum area ν(A ∨ B), auto-dispatch |
| `wangmendel.generate_rule_base(...)` | Generate rule base from data |

### `afis.visualization`

| Symbol | Description |
|---|---|
| `create_rule_base(...)` | Build a `FuzzyRuleBase` from lists |
| `run_afis(input, rule_base, ...)` | Run inference and defuzzify |
| `plot_results(...)` | Plot antecedents, output, and defuzzified value |
| `plot_supremum(...)` | Visualize ν(X ∨ A) and SVI |
| `show_svi_table(...)` | Print SVI and S_A per rule |
| `compute_activation_curves(...)` | Sweep input and compute S_A curves |

### `afis.regression`

| Symbol | Description |
|---|---|
| `AFISRegressor` | Full regression model: `fit`, `predict`, `save`, `load` |
| `generate_rule_base(...)` | Wang-Mendel rule generation |
| `evaluate_kfold(...)` | K-fold cross-validation |
| `benchmark_method(...)` | Repeated K-fold with held-out test sets |
| `compute_metrics(y_true, y_pred)` | Returns RMSE, MAE, R² |

### `AFISRegressor` config reference

| Key | Default | Description |
|---|---|---|
| `k_max` | `10` | Upper bound for neighbor search; used as fixed k when `optimize_k=False` |
| `k_fixed` | — | If set, skips k optimization and uses this value directly |
| `p_norm` | `1` | Minkowski distance order for neighbor lookup |
| `agg_method` | `'avg'` | Aggregation across input dimensions |
| `t_norm_type` | `'product'` | T-norm for consequent scaling |
| `imp_params` | `['luka', 1]` | Implication type and parameter |
| `param_range` | `(0.1, 50.0)` | Search range for parametric implication optimization |
| `param_step` | `0.5` | Step size for implication parameter grid search |
| `disc` | `100` | Discretization points for numerical integration |

---

## Credits

Fuzzy set classes and Wang-Mendel rule learning are based on the original work of **Renato Lopes Moura**, used here with modifications:
[https://github.com/renatolm/wang-mendel](https://github.com/renatolm/wang-mendel)
