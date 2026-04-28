# A-FIS — A-Subsethood Fuzzy Inference System

A Python library implementing the A-FIS (A-subsethood based Fuzzy Inference System), including rule learning, inference, regression, and visualization tools.

---

## Package Structure

```
afis/
├── core/
│   ├── afis_utils.py      # Membership functions, fuzzy rule structures, defuzzification
│   ├── A_FIS.py           # Main inference algorithm (A_FIS)
│   ├── A_vee_B.py         # Supremum ν(A ∨ B) — analytical and numerical
│   └── wangmendel.py      # Fuzzy rule learning (Wang-Mendel algorithm)
├── visualization/
│   └── plotting.py        # Rule base plots, supremum visualization, diagnostics
├── regression/
│   ├── regressor.py       # AFISRegressor, k-fold evaluation, benchmarking
│   └── utils.py           # Correlation analysis, metrics, result plots
└── examples/
    ├── inference/         # 1D, ND, Gaussian, and supremum notebooks
    └── regression/        # Regression application notebook
```

---

## Installation

### Using the existing virtual environment (recommended)

A `venv` is already set up in the repo root using **Python 3.14**. Activate it before running any notebooks or scripts:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

To deactivate when done:

```bash
deactivate
```

### Creating a fresh virtual environment

If you need to recreate the environment from scratch:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install numpy scipy scikit-learn pandas plotly tqdm jupyterlab
```

### Verifying the environment

```bash
python -c "import afis; print(afis.__version__)"
```

> All notebooks in `afis/examples/` are intended to be run with the `venv` kernel selected in JupyterLab.

---

## Quick Start

### 1D Inference

```python
from afis.core import FuzzySet, FuzzyRuleBase, FuzzyRule, Triangular, Trapezoidal
from afis.visualization import create_rule_base, run_afis, plot_results

# Define antecedents and consequents
A1 = FuzzySet("A1", Triangular(0, 0, 5))
A2 = FuzzySet("A2", Triangular(0, 5, 10))
A3 = FuzzySet("A3", Triangular(5, 10, 10))

Y1 = FuzzySet("Y1", Triangular(0, 0, 5))
Y2 = FuzzySet("Y2", Triangular(0, 5, 10))
Y3 = FuzzySet("Y3", Triangular(5, 10, 10))

rule_base = create_rule_base(
    antecedents=[[A1, A2, A3]],
    consequents=[Y1, Y2, Y3],
    input_ranges=[(0, 10)],
    output_range=(0, 10),
    rules=[([0], 0), ([1], 1), ([2], 2)]
)

# Crisp input
from afis.core import format_FN_N_Dim
x_input = format_FN_N_Dim([3.0])

output, U, y_crisp, max_values = run_afis(x_input, rule_base)
plot_results(rule_base, Trapezoidal(3, 3, 3, 3), output, U, y_crisp)
```

### Regression

```python
import numpy as np
from afis.regression import AFISRegressor, generate_rule_base, evaluate_kfold

X = np.random.rand(200, 3)
y = X[:, 0] + 2 * X[:, 1] - X[:, 2]

rule_base = generate_rule_base(X, y, n_fuzzy_partitions=5, n_rules=30)

config = {
    'agg_method': 'avg',
    't_norm_type': 'product',
    'imp_params': ['luka', 1],
    'k_max': 10,
}

model = AFISRegressor(config)
model.fit(X, y, rule_base)
predictions = model.predict(X)
```

### K-Fold Cross-Validation

```python
results = evaluate_kfold(
    X, y,
    rule_base_generator=lambda X, y: generate_rule_base(X, y, 5, 30),
    config=config,
    n_splits=5
)
print(f"Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
```

---

## Membership Functions

| Class | Parameters | Shape |
|---|---|---|
| `Triangular` | `ini, top, end` | Triangle |
| `Trapezoidal` | `ini, top1, top2, end` | Trapezoid |
| `InferiorBorder` | `top, end` | Left-open plateau |
| `SuperiorBorder` | `ini, top` | Right-open plateau |
| `Gaussian` | `center, sigma` | Gaussian bell |

---

## Key API

### `afis.core`

| Symbol | Description |
|---|---|
| `A_FIS(input, rule_base, ...)` | Main inference function |
| `format_FN_N_Dim(x, ftype)` | Format inputs for A_FIS |
| `nu_A_vee_B(A, B, U)` | Analytical supremum area ν(A ∨ B) |
| `nu_A_vee_B_auto(A, B, U)` | Auto-dispatch (analytical or numerical) |
| `fuzzy_imp(a, b, type, param)` | Fuzzy implication I(a, b) |
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
| `AFISRegressor` | Full regression model (fit, predict, save, load) |
| `generate_rule_base(...)` | Wang-Mendel rule generation |
| `evaluate_kfold(...)` | K-fold cross-validation |
| `benchmark_method(...)` | Repeated K-fold with held-out test sets |
| `compute_metrics(y_true, y_pred)` | RMSE, MAE, R² |

---

## Credits

The fuzzy set classes (`afis/core/afis_utils.py`) and Wang-Mendel rule learning algorithm (`afis/core/wangmendel.py`) are the original work of **Renato Lopes Moura**, used here with minor modifications:

- GitHub: [https://github.com/renatolm/wang-mendel](https://github.com/renatolm/wang-mendel)
